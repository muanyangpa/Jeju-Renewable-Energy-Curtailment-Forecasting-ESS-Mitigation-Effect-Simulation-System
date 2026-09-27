"""KIM 일사량 예보의 정확도를 정방향으로 축적해 검증한다.

[왜 이 방식인가]
원래 계획은 테스트 구간(2025년)에 KIM 과거 예보를 넣어 정보 손실 재측정을 다시 돌리는
것이었다. 실측으로 확인해보니 **API가 과거 예보를 보관하지 않는다** — 2026-09-27 기준
응답이 오는 가장 오래된 분석시각이 하루 전이었다. 그래서 소급 측정은 불가능하고, 남은
방법은 하나다: 오늘부터 예보를 쌓고, 며칠 뒤 ASOS 실측이 올라오면 맞춰보는 것.

  collect  거래일 하루치 예보를 CSV에 덧붙인다 (KIM 값 + 지금 쓰는 추정값 둘 다)
  report   ASOS 실측이 있는 날에 대해 두 경로의 오차를 비교한다

매일 한 번 collect를 돌려야 한다. 예보를 놓친 날은 되돌릴 수 없다(보관이 없으므로).

  .venv/bin/python scripts/kim_validation_log.py collect            # 내일 거래일
  .venv/bin/python scripts/kim_validation_log.py collect 2026-09-29
  .venv/bin/python scripts/kim_validation_log.py report

[읽는 법]
ASOS 실측을 정답으로 두고 두 경로의 MAE를 비교한다. KIM이 더 낮으면 교체가 옳았다는
직접 증거가 되고, 정보 손실 +2.99%p 중 얼마를 회수하는지 추정할 근거가 된다. 표본이
며칠뿐이면 그 수치를 결론으로 쓸 수 없다 — 날 수를 항상 함께 보고한다.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from app.services import kim_forecast as K

LAT, LON = 33.5141, 126.5297
LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "kim_forecast_log.csv")
COLS = ["target_date", "hour", "tmfc", "kim_mj", "kim_inst_mj", "est_mj",
        "wind80", "temp_c", "collected_at"]


def collect(target: dt.date) -> None:
    rows = K.fetch_day(target, lat=LAT, lon=LON)
    if len(rows) != 24:
        print(f"❌ 24시간 중 {len(rows)}개만 받았습니다 — 기록하지 않습니다.")
        return
    tmfc = K.default_tmfc(target)

    # 같은 날 추정값도 함께 남긴다. 나중에 두 경로를 같은 실측에 대고 비교하려면
    # '그때 추정했다면 얼마였는가'가 기록돼 있어야 한다 — 사후 재현은 예보가 사라져 불가능하다.
    est = {}
    try:
        from app.services.kma_forecast import fetch
        est = {w["hour"]: w["solar_rad"] for w in fetch(target, lat=LAT, lon=LON, use_kim=False)}
    except Exception as e:                                        # noqa: BLE001
        print(f"⚠ 추정값을 함께 받지 못했습니다({type(e).__name__}) — KIM 값만 기록합니다.")

    new = pd.DataFrame([{
        "target_date": target.isoformat(), "hour": r.hour, "tmfc": tmfc,
        "kim_mj": r.ghi_mj, "kim_inst_mj": r.inst_mj, "est_mj": est.get(r.hour, np.nan),
        "wind80": r.wind80, "temp_c": r.temp_c,
        "collected_at": dt.datetime.now().isoformat(timespec="seconds")} for r in rows])[COLS]

    if os.path.exists(LOG):
        old = pd.read_csv(LOG)
        dup = (old["target_date"] == target.isoformat()) & (old["tmfc"] == tmfc)
        if dup.any():
            print(f"이미 기록돼 있습니다 ({target} / tmfc={tmfc}) — 덮어씁니다.")
            old = old[~dup]
        new = pd.concat([old, new], ignore_index=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new.to_csv(LOG, index=False)
    days = new["target_date"].nunique()
    print(f"✅ {target} 24시간 기록 (tmfc={tmfc}) · 일적산 KIM {sum(r.ghi_mj for r in rows):.2f} MJ/m²")
    print(f"   누적 {days}일 / {len(new)}행 -> {LOG}")
    print(f"   ASOS 실측은 보통 며칠 뒤 올라옵니다. 그 뒤 report로 비교하세요.")


def report() -> None:
    if not os.path.exists(LOG):
        sys.exit(f"기록이 없습니다: {LOG}\n먼저 collect를 며칠 돌려야 합니다.")
    log = pd.read_csv(LOG)
    log["dt"] = (pd.to_datetime(log["target_date"])
                 + pd.to_timedelta(log["hour"], unit="h"))
    print(f"예보 기록 {log['target_date'].nunique()}일 / {len(log)}행 "
          f"({log['target_date'].min()} ~ {log['target_date'].max()})")

    from app.data_prep import load_asos
    obs = load_asos("184")[["dt", "solar_rad"]].rename(columns={"solar_rad": "실측"})
    j = log.merge(obs, on="dt", how="inner").dropna(subset=["실측"])
    if j.empty:
        print("\nASOS 실측과 겹치는 시간이 아직 없습니다. 관측자료가 올라온 뒤 다시 실행하세요.")
        print("  (기상자료개방포털 ASOS 시간자료를 data/에 내려놓아야 합니다)")
        return

    day = j[(j["실측"] > 0.02) | (j["kim_mj"] > 0.02)]
    print(f"\n실측과 겹치는 {j['target_date'].nunique()}일 / 주간 {len(day)}시간")
    if j["target_date"].nunique() < 10:
        print("  ⚠ 표본이 적습니다 — 아래 수치는 경향만 읽고 결론으로 쓰지 마세요.")

    print(f"\n{'경로':10} {'MAE':>8} {'NMAE':>8} {'편향':>8} {'상관':>7} {'일적산비':>9}")
    for name, col in (("KIM 예보", "kim_mj"), ("KIM 순간값", "kim_inst_mj"), ("추정(기존)", "est_mj")):
        if col not in day or day[col].isna().all():
            print(f"{name:10} {'기록 없음':>8}")
            continue
        d = day[col] - day["실측"]
        mae = d.abs().mean()
        nmae = 100 * mae / day["실측"].mean() if day["실측"].mean() else np.nan
        print(f"{name:10} {mae:8.3f} {nmae:7.1f}% {d.mean():+8.3f} "
              f"{day[col].corr(day['실측']):7.3f} {day[col].sum()/day['실측'].sum():9.3f}")
    print("\nMAE·NMAE가 낮을수록 좋다. KIM이 추정보다 낮으면 교체가 옳았다는 직접 증거다.")

    # 일별 일적산 대조 — 특정 날짜만 크게 틀렸는지 확인한다
    dd = j.groupby("target_date")[["kim_mj", "est_mj", "실측"]].sum().round(2)
    dd.columns = ["KIM", "추정", "실측"]
    print(f"\n일적산 (MJ/m²)\n{dd.to_string()}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "collect"
    if mode == "collect":
        t = (dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2
             else dt.date.today() + dt.timedelta(days=1))
        collect(t)
    elif mode == "report":
        report()
    else:
        sys.exit(f"알 수 없는 모드: {mode} (collect | report)")
