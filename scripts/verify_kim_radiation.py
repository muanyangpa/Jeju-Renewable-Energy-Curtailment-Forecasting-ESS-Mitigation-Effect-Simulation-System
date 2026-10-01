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
    """가장 최신부터 거슬러 올라가며 실제로 응답이 오는 분석시각을 찾는다.

    **이 함수가 답할 수 없는 것**: "입찰마감(D-1 11시 KST) 시점에 그 자료가 나와 있었는가".
    지금 응답이 온다는 것은 '실행 시각까지는 생산됐다'는 뜻일 뿐이다. 마감 시점의 가용성은
    생산 지연을 알아야 하고, 그래서 아래에서 현재 최신 분석시각 대비 경과 시간을 함께 찍는다.
    """
    key = os.environ.get("KMA_AUTH_KEY")
    if not key:
        sys.exit("KMA_AUTH_KEY 환경변수가 없습니다.")
    ctx = K._ctx()
    base = dt.datetime.combine(target, dt.time(0), tzinfo=K.KST).astimezone(dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    print(f"1) 분석시각 가용성   (실행 시각 {now:%Y-%m-%d %H:%M} UTC)")
    print("   전체 24시간을 덮으려면 24시(=D+1 00시 KST)의 hf가 48 이하여야 한다.")
    ok_list: list[tuple[str, int, float]] = []
    for back in (6, 12, 18, 24, 30, 36, 42, 48, 72, 120, 240):
        t = base - dt.timedelta(hours=back)
        t = t.replace(hour=t.hour // 6 * 6, minute=0)
        tmfc = t.strftime("%Y%m%d%H")
        hf13, hf24 = K.hf_for(target, 13, tmfc), K.hf_for(target, 24, tmfc)
        covers = "전체" if hf24 <= 48 else f"부족(24시 hf={hf24})"
        try:
            v = K._one(key, ctx, tmfc, hf13, LAT, LON, 30.0)
        except Exception as e:                                    # noqa: BLE001
            print(f"   {tmfc} UTC  ✗ {type(e).__name__}")
            continue
        if "ACSWDNB" not in v:
            print(f"   {tmfc} UTC  ✗ 값 없음 (미생산 또는 보관기간 경과)")
            continue
        age = (now - t.replace(tzinfo=dt.timezone.utc)).total_seconds() / 3600
        print(f"   {tmfc} UTC  ✓  분석 후 {age:5.1f}h 경과 · 커버리지 {covers}")
        if hf24 <= 48:
            ok_list.append((tmfc, hf24, age))

    if not ok_list:
        print("   -> 전체 24시간을 덮는 분석시각이 없습니다.")
        return None
    newest = ok_list[0]
    oldest = ok_list[-1]
    print(f"\n   보관 깊이: 가장 오래된 응답이 분석 후 {oldest[2]:.0f}h "
          f"({oldest[2]/24:.1f}일) — 과거 예보 아카이브는 "
          f"{'있습니다' if oldest[2] > 24 * 30 else '사실상 없습니다'}")
    print(f"   가장 최신: {newest[0]} UTC (분석 후 {newest[2]:.1f}h)")
    print("   ⚠ 이 결과는 '지금 받을 수 있다'만 말한다. 입찰마감(D-1 11시 KST) 시점의 가용성은\n"
          "     생산 지연을 알아야 하므로, 마감 직전 시각에 한 번 더 실행해 확인할 것.")
    # 기본값(D-2 18 UTC)이 살아 있으면 그것을 쓴다 — 마감 전 가용성이 가장 확실하다.
    dflt = K.default_tmfc(target)
    if any(t == dflt for t, _, _ in ok_list):
        print(f"   -> 사용: {dflt} (모듈 기본값, 마감 전 가용성이 가장 안전)")
        return dflt
    print(f"   -> 사용: {newest[0]} (기본값 {dflt}는 응답 없음)")
    return newest[0]


def hourly(target: dt.date, tmfc: str) -> list[K.KimHour]:
    print(f"\n2) 시간별 일사량  (tmfc={tmfc} UTC, 제주 {LAT},{LON}, KIM 국지 1.5km)")
    errs: list[str] = []
    rows = K.fetch_day(target, lat=LAT, lon=LON, tmfc=tmfc, errors=errs)
    if errs:
        print(f"   수신 실패 {len(errs)}건 — 첫 건: {errs[0]}")
    print("   시각  직달W  산란W   채택MJ  순간값MJ    차이   ← 채택 = ACSWDNB 시간차분")
    for r in rows:
        print(f"   {r.hour:3}  {r.direct_w:6.1f} {r.diffuse_w:6.1f}  "
              f"{r.ghi_mj:6.3f}  {r.inst_mj:7.3f}  {r.inst_gap:+7.3f}")
    g = np.array([r.ghi_mj for r in rows])
    i_ = np.array([r.inst_mj for r in rows])
    print(f"   일적산 채택 {g.sum():.2f} MJ/m²  ·  순간값 경로 {i_.sum():.2f} MJ/m²")
    print(f"   두 경로 차이: 평균 {np.mean(i_-g):+.4f}  최대절대 {np.max(np.abs(i_-g)):.4f} MJ/m²")
    print("   (순간값은 그 시각의 스냅샷이라 시간평균이 아니다. 차이가 큰 시간은 구름 변동이\n"
          "    심해 시간평균 자체의 대표성이 낮다는 신호다.)")
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
