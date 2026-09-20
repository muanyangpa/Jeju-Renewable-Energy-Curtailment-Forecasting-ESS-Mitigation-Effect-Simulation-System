"""
날씨->발전량 변환모델(RandomForestRegressor) 학습.

계획서 02장/05장에 정의된 ②단계 모델. H장에서 확인된 교훈(시간 피처 없으면 상관계수 0.54,
추가하면 0.90 이상)을 그대로 반영해 hour_sin/cos, month_sin/cos를 입력에 포함한다.

실행: python -m app.training.train_converter
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from app.data_prep import add_time_features, load_asos, load_asos_multi, load_generation_actual

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

SOLAR_FEATURES = ["solar_rad", "temp", "cloud", "hour_sin", "hour_cos", "month_sin", "month_cos"]
WIND_FEATURES = ["wind_speed", "hour_sin", "hour_cos", "month_sin", "month_cos"]


def _nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.mean(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return mean_absolute_error(y_true, y_pred) / denom * 100


def train_one(energy_type: str) -> dict:
    if energy_type == "wind":
        # I장 재검증 결과: 3지점(제주184·고산185·성산188) 평균이 단일지점보다 위치오차를 상쇄해
        # 더 정확하다(corr 0.656->0.763, NMAE 15.58%->12.03%). 풍력은 3지점 평균을 사용.
        weather = load_asos_multi(("184", "185", "188"))
        station = "184+185+188(평균)"
    else:
        weather = load_asos("184")  # 태양광은 일사량 기준 지점 하나로 충분(H장 검증)
        station = "184"
    weather = add_time_features(weather)
    gen = load_generation_actual(energy_type)

    df = weather.merge(gen, on="dt", how="inner").dropna()
    features = SOLAR_FEATURES if energy_type == "solar" else WIND_FEATURES

    # 리키지-프리: 2023년을 테스트로, 그 이전을 학습으로 (계획서 전체와 동일한 시간분할 원칙)
    train = df[df["dt"] < "2023-01-01"]
    test = df[df["dt"] >= "2023-01-01"]

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train[features], train["generation_mwh"])

    pred = model.predict(test[features])
    corr = np.corrcoef(test["generation_mwh"], pred)[0, 1]
    nmae = _nmae(test["generation_mwh"].to_numpy(), pred)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"converter_{energy_type}.joblib")
    joblib.dump({"model": model, "features": features, "station": station}, out_path)

    metrics = {"energy_type": energy_type, "station": station, "corr": round(float(corr), 4), "nmae_pct": round(float(nmae), 2),
               "n_train": len(train), "n_test": len(test)}
    print(f"[converter:{energy_type}] corr={metrics['corr']} NMAE={metrics['nmae_pct']}% "
          f"(train={metrics['n_train']}, test={metrics['n_test']}) -> {out_path}")
    return metrics


if __name__ == "__main__":
    results = [train_one("solar"), train_one("wind")]
    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "converter_metrics.csv"), index=False)
