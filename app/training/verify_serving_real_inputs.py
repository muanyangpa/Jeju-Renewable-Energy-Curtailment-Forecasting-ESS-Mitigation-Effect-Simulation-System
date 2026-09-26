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

OP_THRESHOLD = 0.03  # README '운영 임계값' — classifier_wind_demand 기준으로 고른 값
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
    while d < d1:
        req = day_request(asos, demand, energy_type, d, with_demand, with_solar_weather)
        if req is None:
            skipped += 1
        else:
            res = predict(req)
            n += 1
            models[res.model_used] += 1
            probs += [h.curtailment_probability for h in res.hourly]
            note = res.note or ""
            if "입력이 학습 분포" in note:
                warned += 1
                feats[note.split("—")[1].split()[0]] += 1
        d += timedelta(days=1)
    if not n:
        return None
    p = pd.Series(probs)
    return {"energy_type": energy_type, "demand": with_demand, "solar_weather": with_solar_weather,
            "model_used": max(models, key=models.get), "window": f"{start}~{end}",
            "n_days": n, "n_skipped": skipped,
            "drift_warn_days": warned, "drift_warn_pct": round(warned / n * 100, 2),
            "drift_features": ";".join(f"{k}:{v}" for k, v in feats.items()),
            "mean_proba": round(float(p.mean()), 4),
            "p95_proba": round(float(p.quantile(0.95)), 4),
            "pct_over_op_threshold": round(float((p >= OP_THRESHOLD).mean()) * 100, 1)}


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
                  f"드리프트경고 {r['drift_warn_pct']:5.1f}%  "
                  f"확률평균 {r['mean_proba']:.4f}  임계 0.03 초과 {r['pct_over_op_threshold']:5.1f}%"
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
    GUARD_PCT = 2.0
    bad = out[(out.span == "학습 구간") & (out.drift_warn_pct > GUARD_PCT)]
    if len(bad):
        print(f"\n⚠ 기저 학습 구간에서 드리프트 경고가 {GUARD_PCT}%를 넘었다 — 감지기가 과민하다. "
              "train_classifier.feature_ranges를 확인할 것 "
              "(p1~p99를 쓰면 정의상 2%가 범위 밖이 되어 이 값이 20%대로 뛴다):")
        print(bad[["path", "span", "drift_warn_pct", "drift_features"]].to_string(index=False))
    else:
        print(f"\n✅ 기저 학습 구간 드리프트 경고 {out[out.span=='학습 구간'].drift_warn_pct.max():.1f}% "
              f"(<= {GUARD_PCT}%) — 감지기가 학습 데이터에서 울리지 않는다")

    # 운영 임계값 점검: 0.03은 classifier_wind_demand 기준으로 고른 값이다.
    t23 = out[out.span == "테스트 2023"]
    print("\n[운영 임계값 0.03의 모델별 발동률 — 2023]")
    for _, r in t23.iterrows():
        rate = base_rate["wind" if r.energy_type == "wind" else "solar"]
        ratio = r.pct_over_op_threshold / rate
        flag = "  ⚠ 실제 제어율의 " + f"{ratio:.1f}배" if ratio > 2.0 else ""
        print(f"  {r.path:26} {r.pct_over_op_threshold:5.1f}% (실제 {rate:.2f}%){flag}")
    print("  -> 발동률이 모델마다 크게 다르면 0.03을 공통으로 쓸 수 없다. 모델별 재선정이 필요하다.")
    return out


if __name__ == "__main__":
    main()
