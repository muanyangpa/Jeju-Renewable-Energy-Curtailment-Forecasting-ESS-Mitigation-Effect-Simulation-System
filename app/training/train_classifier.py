"""
출력제어 확률 예측 분류모델(RandomForestClassifier) 학습.

계획서 04장 01번 기능과 동일한 스펙:
RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=6)
입력: 발전량 · 시_sin/cos · 월_sin/cos (+ 풍력은 수요 실측을 추가 피처로 정식 채택, 05장 근거)
태양광은 수요 피처를 정식 채택하지 않는다 — 검증 가능한 표본 부족(05장).

실행: python -m app.training.train_classifier
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from app.data_prep import add_time_features, load_curtailment, load_demand_actual, load_generation_actual

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

BASE_FEATURES = ["generation_mwh", "hour_sin", "hour_cos", "month_sin", "month_cos"]


def _top5_capture(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """상위5%포착률 — 문서 전체에서 쓰는 핵심 지표(05장 정의 그대로)."""
    n = len(y_true)
    k = max(1, int(np.ceil(n * 0.05)))
    top_idx = np.argsort(-y_score)[:k]
    return float(y_true[top_idx].sum() / max(1, y_true.sum()))


def train_one(energy_type: str, use_demand: bool) -> dict:
    gen = load_generation_actual(energy_type)
    gen = add_time_features(gen)
    curt = load_curtailment(energy_type)[["dt", "is_curtailed"]]
    df = gen.merge(curt, on="dt", how="inner")

    features = list(BASE_FEATURES)
    if use_demand:
        demand = load_demand_actual()
        df = df.merge(demand, on="dt", how="inner")
        features.append("demand_mw")

    df["label"] = df["is_curtailed"].astype(int)

    # 리키지-프리 시간분할: 2023년을 테스트로 사용 (07장 ESS 계산과 같은 기준 연도)
    train = df[df["dt"] < "2023-01-01"]
    test = df[df["dt"] >= "2023-01-01"]

    model = RandomForestClassifier(class_weight="balanced", n_estimators=200, max_depth=6,
                                    random_state=42, n_jobs=-1)
    model.fit(train[features], train["label"])

    proba = model.predict_proba(test[features])[:, 1]
    y_true = test["label"].to_numpy()
    auc = roc_auc_score(y_true, proba) if y_true.sum() > 0 else float("nan")
    top5 = _top5_capture(y_true, proba)

    os.makedirs(MODELS_DIR, exist_ok=True)
    suffix = "_demand" if use_demand else ""
    out_path = os.path.join(MODELS_DIR, f"classifier_{energy_type}{suffix}.joblib")
    joblib.dump({"model": model, "features": features}, out_path)

    metrics = {"energy_type": energy_type, "use_demand": use_demand, "auc": round(float(auc), 4),
               "top5_capture": round(top5, 4), "n_train": len(train), "n_test": len(test),
               "n_pos_test": int(y_true.sum())}
    print(f"[classifier:{energy_type}{suffix}] AUC={metrics['auc']} top5={metrics['top5_capture']} "
          f"(test n={metrics['n_test']}, pos={metrics['n_pos_test']}) -> {out_path}")
    return metrics


if __name__ == "__main__":
    results = [
        train_one("solar", use_demand=False),
        train_one("wind", use_demand=False),
        train_one("wind", use_demand=True),  # 정식 채택 모델 (05장)
    ]
    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "classifier_metrics.csv"), index=False)
