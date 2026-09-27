"""KIM 일사량 예보를 실제로 받아 검증한다. 인증키가 있는 환경에서 실행해야 한다.

  .venv/bin/python scripts/verify_kim_radiation.py [거래일 YYYY-MM-DD]

세 가지를 확인한다.
  1) 분석시각 가용성 — 기본값(D-2 18 UTC)이 실제로 서비스되는지, 과거 분석시각은 얼마나
     남아 있는지. 과거 보관이 없으면 '과거 일사량 예보 vs 실측' 비교(정보손실 재측정)가
     아예 불가능하므로, 이 결과가 다음 단계를 결정한다.
  2) 시간별 일사량 — 직달+산란 합성값과 ACSWDNB 누적차분이 일치하는지.
  3) 추정 모델과의 차이 — 지금 쓰는 radiation_estimator가 얼마나 벗어나 있었는지.
또한 받은 원문을 scratchpad에 저장해 오프라인 테스트 샘플로 쓸 수 있게 한다.
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


def availability(target: dt.date) -> str | None:
    """가장 최신부터 거슬러 올라가며 실제로 응답이 오는 분석시각을 찾는다."""
    key = os.environ.get("KMA_AUTH_KEY")
    if not key:
        sys.exit("KMA_AUTH_KEY 환경변수가 없습니다.")
    ctx = K._ctx()
    base = dt.datetime.combine(target, dt.time(0), tzinfo=K.KST).astimezone(dt.timezone.utc)
    print("1) 분석시각 가용성  (거래일 24시간을 hf 22~45로 덮는 조건 = base로부터 6~48h 전)")
    first_ok = None
    for back in (6, 12, 18, 24, 30, 36, 42, 48, 72, 120, 240):
        t = base - dt.timedelta(hours=back)
        t = t.replace(hour=t.hour // 6 * 6, minute=0)
        tmfc = t.strftime("%Y%m%d%H")
        hf = K.hf_for(target, 13, tmfc)          # 13시(정오 부근) 기준으로 시험
        try:
            v = K._one(key, ctx, tmfc, hf, LAT, LON, 30.0)
        except Exception as e:                                    # noqa: BLE001
            print(f"   {tmfc} UTC (hf={hf:3})  ✗ {type(e).__name__}")
            continue
        ok = "SWDDIR2" in v
        print(f"   {tmfc} UTC (hf={hf:3})  {'✓' if ok else '✗ 값 없음'}"
              + (f"  직달 {v['SWDDIR2']:.1f} W/m²" if ok else ""))
        if ok and first_ok is None and hf <= 45:
            first_ok = tmfc
    print(f"   -> 입찰마감 전에 쓸 수 있는 가장 최신 분석시각: {first_ok or '없음'}")
    return first_ok


def hourly(target: dt.date, tmfc: str) -> list[K.KimHour]:
    print(f"\n2) 시간별 일사량  (tmfc={tmfc} UTC, 제주 {LAT},{LON}, KIM 국지 1.5km)")
    errs: list[str] = []
    rows = K.fetch_day(target, lat=LAT, lon=LON, tmfc=tmfc, errors=errs)
    if errs:
        print(f"   수신 실패 {len(errs)}건 — 첫 건: {errs[0]}")
    print("   시각  직달W  산란W   합성MJ  누적차분MJ  차이")
    for r in rows:
        d = r.ghi_mj - r.acc_mj if r.acc_mj == r.acc_mj else float("nan")
        print(f"   {r.hour:3}  {r.direct_w:6.1f} {r.diffuse_w:6.1f}  "
              f"{r.ghi_mj:6.3f}  {r.acc_mj:9.3f}  {d:+.3f}")
    g = np.array([r.ghi_mj for r in rows])
    a = np.array([r.acc_mj for r in rows])
    m = ~np.isnan(a)
    print(f"   일적산 합성 {g.sum():.2f} MJ/m²  ·  누적차분 {a[m].sum():.2f} MJ/m²")
    if m.sum():
        print(f"   두 경로 차이: 평균 {np.mean(g[m]-a[m]):+.4f}  최대절대 "
              f"{np.max(np.abs(g[m]-a[m])):.4f} MJ/m²")
    print(f"   허브높이(80m) 풍속 {min(r.wind80 for r in rows):.1f}~"
          f"{max(r.wind80 for r in rows):.1f} m/s")
    return rows


def versus_estimator(target: dt.date, rows: list[K.KimHour]) -> None:
    from app.services.kma_forecast import fetch
    print("\n3) 지금 쓰는 추정값과의 차이")
    try:
        est = fetch(target, lat=LAT, lon=LON, use_kim=False)
    except Exception as e:                                        # noqa: BLE001
        print(f"   단기예보를 못 받아 비교를 건너뜁니다: {type(e).__name__}: {e}")
        return
    e = pd.Series({w["hour"]: w["solar_rad"] for w in est})
    k = pd.Series({r.hour: r.ghi_mj for r in rows})
    j = pd.DataFrame({"추정": e, "KIM": k}).dropna()
    day = j[j.max(axis=1) > 0.02]
    print(f"   일적산  추정 {j['추정'].sum():.2f}  KIM {j['KIM'].sum():.2f} MJ/m² "
          f"({(j['추정'].sum()/max(j['KIM'].sum(),1e-9)-1)*100:+.1f}%)")
    if len(day):
        d = day["추정"] - day["KIM"]
        print(f"   주간 {len(day)}시간  편향 {d.mean():+.3f}  MAE {d.abs().mean():.3f} "
              f"MJ/m²  상관 {day['추정'].corr(day['KIM']):.3f}")
    print(j.round(3).to_string())


def save_sample(target: dt.date, tmfc: str) -> None:
    """오프라인 테스트 샘플 저장. 위치는 scratchpad — 저장소에 원문을 넣지 않는다."""
    key, ctx = os.environ["KMA_AUTH_KEY"], K._ctx()
    out = ["# KIM 국지 1.5km 제주 지점 응답 원문 (오프라인 테스트용)",
           f"# tmfc={tmfc} UTC  lat={LAT} lon={LON}"]
    need = sorted({K.hf_for(target, h, tmfc) for h in range(1, 25)}
                  | {K.hf_for(target, h, tmfc) - 1 for h in range(1, 25)})
    import urllib.parse, urllib.request
    for hf in need:
        q = urllib.parse.urlencode({
            "group": K.GROUP, "nwp": K.NWP, "data": K.DATA, "name": ",".join(K.NAMES),
            "tmfc": tmfc, "hf": str(hf), "disp": "A", "help": "1",
            "lat": f"{LAT}", "lon": f"{LON}", "authKey": key})
        try:
            with urllib.request.urlopen(f"{K.ENDPOINT}?{q}", timeout=30, context=ctx) as r:
                body = r.read().decode("cp949", "replace")
        except Exception as ex:                                   # noqa: BLE001
            print(f"   hf={hf} 저장 실패: {type(ex).__name__}")
            continue
        out += [f"#hf={hf}", body.rstrip()]
    path = os.path.join(
        os.environ.get("CLAUDE_SCRATCHPAD", "/tmp"), f"kim_sample_{target:%Y%m%d}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n4) 오프라인 테스트 샘플 저장: {path}  ({os.path.getsize(path)/1024:.0f} KB)")
    print("   인증키는 이 파일에 들어가지 않습니다 (응답 본문만 저장).")


if __name__ == "__main__":
    tgt = (dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1
           else dt.date.today() + dt.timedelta(days=1))
    print(f"거래일 {tgt}\n")
    tmfc = availability(tgt) or K.default_tmfc(tgt)
    rows = hourly(tgt, tmfc)
    if len(rows) == 24:
        versus_estimator(tgt, rows)
        save_sample(tgt, tmfc)
    else:
        print(f"\n24시간을 못 채웠습니다({len(rows)}개) — 운영에서는 추정값으로 폴백합니다.")
