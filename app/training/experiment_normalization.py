"""발전량 피처 정규화 실험 — 설비 증설로 인한 분포 이동에 강건해지는가? [신규]

문제:
  제주 태양광 설비가 계속 늘어 발전량 절대값(MWh)의 분포가 해마다 위로 밀린다.
  분류기 학습 구간(<2022-07)의 최대값을 2023 테스트가 329시간(3.8%) 초과하는데,
  RandomForest는 외삽을 못 해 학습 범위 경계에서 포화된다 — 큰 발전량 시간들이
  서로 구분되지 않아 순위가 무너진다. 2026년에는 더 벌어진다.

비교하는 피처 구성:
  A 현행      : generation_mwh (절대 MWh)
  B 이용률    : capacity_factor = 발전량 / 설비용량대리
  C 이용률+침투: capacity_factor + penetration(= 발전량 / 수요)
  D 전부      : 위 셋 모두

  이용률만 쓰면(B) 범위는 안정되지만 '잉여 규모' 정보를 잃는다. 침투율(C)이 그
  정보를 스케일 안정적인 형태로 되살린다. D는 절대값까지 남겨 비교 기준으로 둔다.

같이 보는 것: 테스트 구간에서 학습 범위를 벗어난 행의 비율(포화 위험 지표).

실행: python -m app.training.experiment_normalization
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from app.data_prep import (add_normalized_features, add_time_features, build_labeled_hourly,
                           load_demand_actual, load_generation_actual)
from app.metrics import classification_report
from app.model_io import MODELS_DIR
from app.training.train_classifier import TEST_END, TEST_START, TRAIN_END, make_model

TIME = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
VARIANTS = {
    "A 현행(절대 MWh)": ["generation_mwh"],
    "B 이용률": ["capacity_factor"],
    "C 이용률+침투율": ["capacity_factor", "penetration"],
    "D 전부": ["generation_mwh", "capacity_factor", "penetration"],
}


def build(energy_type: str) -> pd.DataFrame:
    df = add_time_features(build_labeled_hourly(energy_type))
    df = add_normalized_features(df, load_generation_actual(energy_type), load_demand_actual())
    if energy_type == "wind":
        pass  # demand_mw는 add_normalized_features에서 이미 병합됨(풍력 정식 피처)
    return df


def out_of_range_pct(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> float:
    """테스트 행 중 어느 피처든 학습 범위를 벗어난 비율 — 포화 위험 지표."""
    lo, hi = train[cols].min(), train[cols].max()
    outside = ((test[cols] < lo) | (test[cols] > hi)).any(axis=1)
    return float(outside.mean() * 100)


def run(energy_type: str) -> list[dict]:
    df = build(energy_type)
    extra = ["demand_mw"] if energy_type == "wind" else []
    rows = []
    label = {"solar": "태양광", "wind": "풍력"}[energy_type]
    print(f"\n=== {label} ===")
    for name, feats in VARIANTS.items():
        cols = feats + TIME + extra
        d = df.dropna(subset=cols + ["is_curtailed"])
        train = d[d["dt"] < TRAIN_END]
        test = d[(d["dt"] >= TEST_START) & (d["dt"] < TEST_END)]
        if train.empty or test.empty:
            continue
        m = make_model()
        m.fit(train[cols], train["is_curtailed"])
        rep = classification_report(test["is_curtailed"].to_numpy(),
                                    m.predict_proba(test[cols])[:, 1])
        oor = out_of_range_pct(train, test, feats)
        rows.append({"energy_type": energy_type, "variant": name, "features": "+".join(feats),
                     "out_of_range_pct": round(oor, 2), "n_train": len(train), **rep})
        print(f"  {name:<16} AUC={rep['auc']:.4f} PR-AUC={rep['pr_auc']:.4f} "
              f"top5={rep['top5_capture']:.4f} Brier={rep['brier']:.4f} | 학습범위 밖 {oor:5.2f}%")
    best = max(rows, key=lambda r: r["pr_auc"])
    print(f"  -> PR-AUC 최고: {best['variant']} ({best['pr_auc']:.4f})")
    return rows


if __name__ == "__main__":
    results = run("solar") + run("wind")
    out = os.path.join(MODELS_DIR, "normalization_experiment.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n저장: {out}")


# ---------------------------------------------------------------------------
# 컨버터(날씨 -> 발전량) 타깃 정규화 실험
# ---------------------------------------------------------------------------
# 컨버터는 분류기와 문제의 방향이 반대다. 입력(날씨)은 드리프트하지 않고 '타깃'인
# 발전량이 증설로 커진다. RandomForest는 학습에서 본 타깃 최대값을 넘는 값을 예측할 수
# 없으므로, 2023 컨버터 예측이 263MWh를 한 번도 넘지 못했다(실측은 336MWh까지).
#
#   A 현행   : 타깃 = 발전량(MWh)
#   B 정규화 : 타깃 = 이용률, 예측 후 설비용량대리를 곱해 MWh로 환산
#     B1 시점별 대리지표 — 매 시점의 인과적(과거만) 대리지표. 지속 재학습 환경의 상한
#     B2 고정 대리지표  — 학습 끝 시점 값 하나로 고정. /predict의 실제 동작과 동일

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

from app.data_prep import capacity_proxy, load_asos, load_asos_multi  # noqa: E402
from app.training.train_converter import SOLAR_FEATURES, WIND_FEATURES, _nmae  # noqa: E402

CONV_SPLIT = "2023-01-01"


def run_converter(energy_type: str) -> list[dict]:
    weather = load_asos_multi(("184", "185", "188")) if energy_type == "wind" else load_asos("184")
    weather = add_time_features(weather)
    gen = load_generation_actual(energy_type)
    df = weather.merge(gen, on="dt", how="inner")
    df["capacity_proxy_mwh"] = df["dt"].map(capacity_proxy(gen))
    df = df.dropna(subset=["capacity_proxy_mwh", "generation_mwh"])
    df["capacity_factor"] = df["generation_mwh"] / df["capacity_proxy_mwh"]

    feats = SOLAR_FEATURES if energy_type == "solar" else WIND_FEATURES
    df = df.dropna(subset=feats)
    train = df[df["dt"] < CONV_SPLIT]
    test = df[df["dt"] >= CONV_SPLIT]
    y = test["generation_mwh"].to_numpy()
    label = {"solar": "태양광", "wind": "풍력"}[energy_type]

    def mk():
        return RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)

    preds = {}
    m = mk().fit(train[feats], train["generation_mwh"])
    preds["A 현행(MWh 타깃)"] = np.clip(m.predict(test[feats]), 0, None)

    mb = mk().fit(train[feats], train["capacity_factor"])
    cf = np.clip(mb.predict(test[feats]), 0, None)
    preds["B1 이용률 타깃/시점별 대리"] = cf * test["capacity_proxy_mwh"].to_numpy()
    frozen = float(train["capacity_proxy_mwh"].iloc[-1])
    preds["B2 이용률 타깃/고정 대리"] = cf * frozen

    train_max = float(train["generation_mwh"].max())
    actual_over = float((y > train_max).mean() * 100)
    print(f"\n=== {label} 컨버터 === 학습 타깃 최대 {train_max:.0f}MWh, "
          f"2023 실측이 이를 넘는 시간 {actual_over:.2f}% (고정 대리지표={frozen:.0f}MWh)")

    rows = []
    for name, p in preds.items():
        corr = float(np.corrcoef(y, p)[0, 1])
        nmae = _nmae(y, p)
        over = float((p > train_max).mean() * 100)
        rows.append({"energy_type": energy_type, "variant": name, "corr": round(corr, 4),
                     "nmae_pct": round(nmae, 2), "pred_max_mwh": round(float(p.max()), 1),
                     "pred_over_train_max_pct": round(over, 2),
                     "actual_max_mwh": round(float(y.max()), 1)})
        print(f"  {name:<24} corr={corr:.4f} NMAE={nmae:6.2f}%  "
              f"예측최대={p.max():6.1f}MWh  학습최대 초과={over:5.2f}%")
    print(f"  (실측 최대 {y.max():.0f}MWh — 예측이 여기 못 미치면 외삽 실패)")
    return rows


if __name__ == "__main__" and os.environ.get("CONVERTER_ONLY"):
    res = run_converter("solar") + run_converter("wind")
    out = os.path.join(MODELS_DIR, "converter_normalization_experiment.csv")
    pd.DataFrame(res).to_csv(out, index=False)
    print(f"\n저장: {out}")
