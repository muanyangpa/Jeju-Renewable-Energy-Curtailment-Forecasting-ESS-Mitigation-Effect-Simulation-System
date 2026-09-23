"""
추론 서비스 — 학습된 컨버터/분류기/회귀모델을 로드해 실제 예측을 수행한다.
app/training/*.py 로 학습된 .joblib 아티팩트를 그대로 사용한다.

[수정 사항]
- 누락된 기상값을 0으로 바꾸던 처리 제거 (schemas.py에서 MISSING_REQUIRED_FIELD로 거부)
- 시간/월 피처를 학습과 동일한 기준(각 시간 구간의 '끝 시각' dt)으로 계산
  -> 24시는 다음날 00:00이므로 월말 24시의 month는 다음 달로 계산된다(학습 데이터와 일치)
- 24시간을 한 번에 DataFrame으로 추론 (행마다 DataFrame 생성하던 방식 제거)
- 실제 사용된 분류모델을 응답의 model_used로 반환
- 아티팩트 로드 시 scikit-learn 버전 불일치 경고 (app/model_io.py)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from app.data_prep import add_time_features
from app.model_io import load_artifact
from app.schemas import EnergyType, HourlyPrediction, PredictRequest, PredictResponse

SOLAR_NOTE = (
    "태양광은 전력거래소가 출력제어 제어량(MWh)을 공식적으로 산정하지 않아(계획서 07장) "
    "expected_curtailment_mwh를 제공하지 않습니다. curtailment_probability(발생 확률)만 사용하세요."
)


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return load_artifact(name)


def _optional(name: str) -> dict | None:
    try:
        return _load(name)
    except FileNotFoundError:
        return None


def predict(req: PredictRequest) -> PredictResponse:
    energy_type: EnergyType = req.energy_type
    converter = _load(f"converter_{energy_type}")

    # 학습 데이터와 같은 기준: h시 = target_date 00:00 + h시간 (24시 -> 다음날 00:00)
    base = datetime.combine(req.target_date, datetime.min.time())
    weather = sorted(req.weather, key=lambda w: w.hour)
    df = pd.DataFrame([w.model_dump() for w in weather])
    df["dt"] = [base + timedelta(hours=w.hour) for w in weather]
    df = add_time_features(df)

    # ---- ②단계: 날씨 -> 발전량 예측치 ----
    df["generation_mwh"] = np.clip(converter["model"].predict(df[converter["features"]]), 0, None)

    # ---- ③단계: 발전량 -> 출력제어 확률 ----
    use_demand = energy_type == "wind" and req.demand_forecast_mw is not None
    clf_name = "classifier_wind_demand" if use_demand else f"classifier_{energy_type}"
    classifier = _load(clf_name)
    if use_demand:
        df["demand_mw"] = req.demand_forecast_mw
    proba = classifier["model"].predict_proba(df[classifier["features"]])[:, 1]

    # ---- 풍력 전용: ④ 계산에 쓰이는 제어량(MWh) 회귀 ----
    # 회귀모델은 수요 피처가 필수라 수요예측이 없으면 제어량을 제공하지 않는다
    expected = [None] * 24
    if energy_type == "wind" and use_demand:
        regressor = _optional("curtailment_regressor_wind")
        if regressor is not None:
            expected = np.clip(regressor["model"].predict(df[regressor["features"]]), 0, None).tolist()

    hourly = [
        HourlyPrediction(
            hour=int(h),
            generation_forecast_mwh=round(float(g), 2),
            curtailment_probability=round(float(p), 4),
            expected_curtailment_mwh=round(float(e), 2) if e is not None else None,
        )
        for h, g, p, e in zip(df["hour"], df["generation_mwh"], proba, expected)
    ]

    note = SOLAR_NOTE if energy_type == "solar" else None
    if energy_type == "wind" and not use_demand:
        note = ("demand_forecast_mw가 없어 수요 미포함 모델(classifier_wind)을 사용했고, "
                "제어량 회귀모델은 수요 피처가 필요해 expected_curtailment_mwh를 제공하지 않습니다.")

    return PredictResponse(
        energy_type=energy_type, region=req.region, target_date=req.target_date,
        hourly=hourly, model_used=clf_name, note=note,
    )
