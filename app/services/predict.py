"""
추론 서비스 — 학습된 컨버터/분류기/회귀모델을 로드해 실제 예측을 수행한다.
app/training/*.py 로 학습된 .joblib 아티팩트를 그대로 사용한다.
"""
from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from app.schemas import EnergyType, PredictRequest, HourlyPrediction, PredictResponse

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

SOLAR_NOTE = (
    "태양광은 전력거래소가 출력제어 제어량(MWh)을 공식적으로 산정하지 않아(계획서 07장) "
    "expected_curtailment_mwh를 제공하지 않습니다. curtailment_probability(발생 확률)만 사용하세요."
)


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 없음 — 먼저 `python -m app.training.train_converter` 등으로 학습을 실행하세요"
        )
    return joblib.load(path)


def _time_features(hour_1_24: int, month: int) -> dict:
    hour0 = 0 if hour_1_24 == 24 else hour_1_24
    return {
        "hour_sin": np.sin(2 * np.pi * hour0 / 24),
        "hour_cos": np.cos(2 * np.pi * hour0 / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }


def predict(req: PredictRequest) -> PredictResponse:
    energy_type: EnergyType = req.energy_type
    converter = _load(f"converter_{energy_type}")
    month = req.target_date.month

    weather_by_hour = {w.hour: w for w in req.weather}

    # ---- ②단계: 날씨 -> 발전량 예측치 ----
    gen_forecast: dict[int, float] = {}
    for h in range(1, 25):
        w = weather_by_hour[h]
        tf = _time_features(h, month)
        if energy_type == "solar":
            values = {"solar_rad": w.solar_rad or 0.0, "temp": w.temp or 0.0, "cloud": w.cloud or 0.0, **tf}
        else:
            values = {"wind_speed": w.wind_speed or 0.0, **tf}
        row = pd.DataFrame([values])[converter["features"]]
        pred = converter["model"].predict(row)[0]
        gen_forecast[h] = max(0.0, float(pred))

    # ---- ③단계: 발전량 -> 출력제어 확률 ----
    use_demand = energy_type == "wind" and req.demand_forecast_mw is not None
    clf_name = "classifier_wind_demand" if use_demand else f"classifier_{energy_type}"
    classifier = _load(clf_name)

    demand_by_hour = {}
    if use_demand:
        demand_by_hour = {h: v for h, v in zip(range(1, 25), req.demand_forecast_mw)}

    # ---- 풍력 전용: ④ 계산에 쓰이는 제어량(MWh) 회귀 ----
    regressor = None
    if energy_type == "wind":
        try:
            regressor = _load("curtailment_regressor_wind")
        except FileNotFoundError:
            regressor = None

    hourly: list[HourlyPrediction] = []
    for h in range(1, 25):
        tf = _time_features(h, month)
        feat_row = {
            "generation_mwh": gen_forecast[h],
            "hour_sin": tf["hour_sin"], "hour_cos": tf["hour_cos"],
            "month_sin": tf["month_sin"], "month_cos": tf["month_cos"],
        }
        if use_demand:
            feat_row["demand_mw"] = demand_by_hour[h]

        x = pd.DataFrame([feat_row])[classifier["features"]]
        proba = float(classifier["model"].predict_proba(x)[0][1])

        expected_mwh = None
        if energy_type == "wind" and regressor is not None:
            xr = pd.DataFrame([feat_row])[regressor["features"]]
            expected_mwh = max(0.0, float(regressor["model"].predict(xr)[0]))

        hourly.append(HourlyPrediction(
            hour=h, generation_forecast_mwh=round(gen_forecast[h], 2),
            curtailment_probability=round(proba, 4),
            expected_curtailment_mwh=round(expected_mwh, 2) if expected_mwh is not None else None,
        ))

    return PredictResponse(
        energy_type=energy_type, region=req.region, target_date=req.target_date,
        hourly=hourly, note=(SOLAR_NOTE if energy_type == "solar" else None),
    )
