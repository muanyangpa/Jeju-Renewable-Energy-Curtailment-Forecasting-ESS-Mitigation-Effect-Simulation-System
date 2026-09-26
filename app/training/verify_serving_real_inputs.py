"""실측 입력을 /predict에 그대로 넣어 서빙 동작을 검증한다.

[왜 필요한가]
evaluate_pipeline은 '순위 지표'를 재고, smoke_test는 '계약'을 본다. 둘 다 못 보는 것이
두 가지 있다 — 드리프트 경고가 언제 울리는지, 운영 임계값이 실제로 몇 %를 경보로 띄우는지.
이 둘은 합성 입력으로는 알 수 없다. 실제로 이 스크립트가 두 개의 실제 버그를 잡았다.

  ① 드리프트 경고의 범위를 p1~p99로 잡았더니 학습 구간에서 21.2%의 날이 경고를 받았다.
     p1~p99는 정의상 학습 데이터의 2%가 범위 밖이고 피처를 OR로 합치면 그게 누적된다.
     min~max로 고쳐 학습 구간 0.0%가 됐다.
  ② 운영 임계값 0.03의 발동률이 모델별로 3.6%~33.8%(10배)로 갈렸다. 0.03은
     classifier_wind_demand 기준으로 고른 값인데 다른 모델에도 공통으로 쓰고 있었다.

[입력] ASOS 실측 관측치와 실측 수요. 하루전 예보가 아니라 관측치이므로 이 결과도
운영 성능이 아니라 상한이다(README '서비스 경로 평가'와 같은 조건).

[수요 데이터 상한] 2024-07-01에서 끝난다. 그래서 수요를 쓰는 경로(권장 경로 포함)는
2024년 상반기까지만 검증할 수 있고, 2025~2026년은 수요미포함 모델만 돌 수 있다.

실행: python -m app.training.verify_serving_real_inputs
"""
from __future__ import annotations

import os
from collections import Counter
from datetime import date, timedelta

import pandas as pd

from app.data_prep import build_labeled_hourly, load_asos, load_asos_multi, load_demand_actual
from app.model_io import MODELS_DIR
from app.schemas import PredictRequest
from app.services.predict import predict

# 경로별 운영 임계값 — select_thresholds.py가 고른 값을 /predict가 응답에 담아준다.
# 공통 0.03을 쓰던 시절에는 경로별 발동률이 3.6%~31.8%로 9배 벌어졌다.
SOLAR_FIELDS = ("solar_rad", "temp", "cloud")

WINDOWS = (
    ("학습 구간", "2021-01-01", "2022-07-01"),
    ("보정 구간", "2022-07-01", "2023-01-01"),
    ("테스트 2023", "2023-01-01", "2024-01-01"),
    ("2024 상반기", "2024-01-01", "2024-07-01"),
    ("2025", "2025-01-01", "2026-01-01"),
    ("2026", "2026-01-01", "2026-09-20"),
)

# (발전원, 수요, 태양광 기상값) — /predict의 모델 자동 전환 경로 전부
PATHS = (
    ("wind", True, True),    # 권장 경로 (crossp)
    ("wind", True, False),
    ("wind", False, False),
    ("solar", True, False),
    ("solar", False, False),
)


def _sources():
    return (load_asos_multi(("184", "185", "188")).set_index("dt"),
            load_asos("184").set_index("dt"),
            load_demand_actual().set_index("dt")["demand_mw"])


def day_request(asos, demand, energy_type: str, d: date,
                with_demand: bool, with_solar_weather: bool) -> PredictRequest | None:
    """하루치 실측 관측을 요청으로 만든다. 값이 하나라도 없으면 None (그 날은 건너뛴다).

    h시 = target_date 00:00 + h시간 — 학습·서빙과 같은 규약이다(24시는 다음날 00:00).
    """
    base = pd.Timestamp(d)
    weather, dem = [], []
    for h in range(1, 25):
        t = base + pd.Timedelta(hours=int(h))
        if t not in asos.index:
            return None
        r = asos.loc[t]
        w: dict = {"hour": h}
        if energy_type == "wind":
            if pd.isna(r.wind_speed):
                return None
            w["wind_speed"] = float(r.wind_speed)
            if with_solar_weather:
                if any(pd.isna(r[c]) for c in SOLAR_FIELDS):
                    return None
                w.update({c: float(r[c]) for c in SOLAR_FIELDS})
        else:
            if any(pd.isna(r[c]) for c in SOLAR_FIELDS):
                return None
            w.update({c: float(r[c]) for c in SOLAR_FIELDS})
        weather.append(w)
        if with_demand:
            if t not in demand.index or pd.isna(demand.loc[t]):
                return None
            dem.append(float(demand.loc[t]))
    kw = {"demand_forecast_mw": dem} if with_demand else {}
    return PredictRequest(energy_type=energy_type, region="제주", target_date=d,
                          weather=weather, **kw)


def run_window(sources, energy_type, with_demand, with_solar_weather, start, end) -> dict | None:
    asos_w, asos_s, demand = sources
    asos = asos_w if energy_type == "wind" else asos_s
    d, d1 = date.fromisoformat(start), date.fromisoformat(end)
    n = skipped = warned = 0
    feats, models, probs = Counter(), Counter(), []
    clf_warned = [0]
    thr, thr_reliable = None, None
    while d < d1:
        req = day_request(asos, demand, energy_type, d, with_demand, with_solar_weather)
        if req is None:
            skipped += 1
        else:
            res = predict(req)
            n += 1
            models[res.model_used] += 1
            probs += [h.curtailment_probability for h in res.hourly]
            thr = res.operational_threshold
            thr_reliable = res.operational_threshold_reliable
            note = res.note or ""
            if "입력이 학습 분포" in note:
                warned += 1
                body = note.split("—")[1].split(".")[0]
                for part in body.split("/"):
                    tok = part.strip().split()
                    if len(tok) >= 2:
                        feats[f"{tok[0]}:{tok[1]}"] += 1
                # 분류기 단계만 따로 센다 — 회귀 가드가 쓰는 값이다(아래 main 주석 참고)
                if "파생->분류기" in body:
                    clf_warned[0] += 1
        d += timedelta(days=1)
    if not n:
        return None
    p = pd.Series(probs)
    return {"energy_type": energy_type, "demand": with_demand, "solar_weather": with_solar_weather,
            "model_used": max(models, key=models.get), "window": f"{start}~{end}",
            "n_days": n, "n_skipped": skipped,
            "drift_warn_days": warned, "drift_warn_pct": round(warned / n * 100, 2),
            "classifier_drift_pct": round(clf_warned[0] / n * 100, 2),
            "drift_features": ";".join(f"{k}={v}" for k, v in feats.items()),
            "mean_proba": round(float(p.mean()), 4),
            "p95_proba": round(float(p.quantile(0.95)), 4),
            "op_threshold": thr, "op_threshold_reliable": thr_reliable,
            "pct_over_op_threshold": (round(float((p >= thr).mean()) * 100, 1)
                                      if thr is not None else None)}


def main() -> pd.DataFrame:
    sources = _sources()
    base_rate = {et: build_labeled_hourly(et).pipe(
        lambda x: x[(x.dt >= "2023-01-01") & (x.dt < "2024-01-01")]).is_curtailed.mean() * 100
        for et in ("wind", "solar")}
    print(f"2023년 실제 제어율: 풍력 {base_rate['wind']:.2f}% / 태양광 {base_rate['solar']:.2f}%")
    print("(아래 '임계 초과'가 이 값보다 크게 높으면 그 모델에 0.03은 맞는 임계값이 아니다)\n")

    rows = []
    for et, dm, sw in PATHS:
        label = f"{et}/{'수요O' if dm else '수요X'}/{'태양광기상O' if sw else '태양광기상X'}"
        print(f"[{label}]")
        for name, a, b in WINDOWS:
            r = run_window(sources, et, dm, sw, a, b)
            if r is None:
                print(f"  {name:12} — 실측 입력이 없어 건너뜀")
                continue
            rows.append({"path": label, "span": name, **r})
            print(f"  {name:12} n={r['n_days']:4}일 skip={r['n_skipped']:3}  "
                  f"드리프트 전체 {r['drift_warn_pct']:5.1f}% (분류기 {r['classifier_drift_pct']:4.1f}%)  "
                  f"확률평균 {r['mean_proba']:.4f}  "
                  f"임계 {r['op_threshold']} 초과 {r['pct_over_op_threshold']:5.1f}%"
                  f"{'' if r['op_threshold_reliable'] else '(임계값 신뢰불가)'}"
                  f"  {r['drift_features']}")
        print()

    out = pd.DataFrame(rows)
    path = os.path.join(MODELS_DIR, "serving_real_input_verification.csv")
    out.to_csv(path, index=False)
    print(f"저장: {path}")

    # 회귀 가드: 기저 학습 구간에서 경고가 거의 울리지 않아야 한다.
    #
    # 정확히 0%가 아닌 이유 — 범위(train_feature_ranges)는 '실측 발전량'으로 만들지만 서빙은
    # '컨버터 예측 발전량'을 넣는다. 컨버터 오차만큼 파생 피처(capacity_factor·penetration)가
    # 학습 범위를 살짝 벗어날 수 있다. 관측된 잔여는 2% 미만이다.
    #
    # 보정 구간(2022 하반기)은 기저 학습 범위 밖 데이터이므로 여기서 울리는 것은 정상이다 —
    # 가드에 넣지 않는다.
    # 가드는 '분류기 단계'만 본다. 컨버터는 학습 구간이 다르므로(태양광 컨버터는 2024-03~12
    # 10개월뿐 — 신규 발전량 데이터셋이 2024년부터 시작하고 2025년을 테스트로 빼기 때문)
    # 분류기 학습 구간의 기상값이 컨버터 범위를 벗어나는 것은 정상이고 실제 신호다.
    GUARD_PCT = 2.0
    bad = out[(out.span == "학습 구간") & (out.classifier_drift_pct > GUARD_PCT)]
    if len(bad):
        print(f"\n⚠ 기저 학습 구간에서 드리프트 경고가 {GUARD_PCT}%를 넘었다 — 감지기가 과민하다. "
              "train_classifier.feature_ranges를 확인할 것 "
              "(p1~p99를 쓰면 정의상 2%가 범위 밖이 되어 이 값이 20%대로 뛴다):")
        print(bad[["path", "span", "drift_warn_pct", "drift_features"]].to_string(index=False))
    else:
        print(f"\n✅ 기저 학습 구간에서 '분류기 단계' 드리프트 경고 "
              f"{out[out.span=='학습 구간'].classifier_drift_pct.max():.1f}% (<= {GUARD_PCT}%) — "
              f"감지기가 학습 데이터에서 울리지 않는다")
        conv = out[out.span == "학습 구간"].drift_warn_pct.max()
        if conv > GUARD_PCT:
            print(f"   (같은 구간의 '컨버터 단계' 경고는 {conv:.1f}%다 — 컨버터 학습 구간이 다르기 "
                  f"때문이고 오탐이 아니다. 태양광 컨버터는 2024-03~12 10개월만 학습했다.)")

    # 운영 임계값 점검: 0.03은 classifier_wind_demand 기준으로 고른 값이다.
    t23 = out[out.span == "테스트 2023"]
    print("\n[경로별 운영 임계값의 실측 발동률 — 2023]")
    for _, r in t23.iterrows():
        rate = base_rate["wind" if r.energy_type == "wind" else "solar"]
        if r.pct_over_op_threshold is None:
            print(f"  {r.path:26} 임계값 미선정 — select_thresholds.py를 실행하세요")
            continue
        ratio = r.pct_over_op_threshold / rate
        flag = (f"  ⚠ 실제 제어율의 {ratio:.1f}배" if ratio > 2.0 else
                f"  ⚠ 임계값 신뢰불가" if not r.op_threshold_reliable else "")
        print(f"  {r.path:26} 임계 {r.op_threshold}: {r.pct_over_op_threshold:5.1f}% "
              f"(실제 {rate:.2f}%, {ratio:.2f}배){flag}")
    return out


if __name__ == "__main__":
    main()
