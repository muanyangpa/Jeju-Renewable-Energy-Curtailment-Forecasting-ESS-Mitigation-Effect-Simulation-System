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

ESS_WARNING = (
    "expected_curtailment_mwh = curtailment_probability x E[제어량|제어 발생]로 계산한 '기댓값'입니다. "
    "총합 추정에는 쓸 수 있으나 개별 시간의 제어량 크기가 아니므로 /ess/simulate의 "
    "hourly_curtailment_mwh로 넣지 마세요 — 흡수율이 크게 과대평가됩니다(README '알려진 한계' 참고)."
)

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

    # ---- ③단계: 발전량 -> 출력제어 확률 (+ 풍력은 제어량 2단계 추정) ----
    # /predict는 isotonic 보정된 분류기만 로드한다 (보정 전 모델을 서빙하는 경로는 없다).
    # 보정 구간은 기저 학습·2023 테스트 어디에도 쓰지 않은 2022-07-01~2023-01-01이며,
    # 아티팩트 메타데이터(calibration_period)에 기록돼 있다.
    use_demand = energy_type == "wind" and req.demand_forecast_mw is not None
    if use_demand:
        df["demand_mw"] = req.demand_forecast_mw

    clf_name = ("classifier_wind_demand_calibrated" if use_demand
                else f"classifier_{energy_type}_calibrated")
    classifier = _load(clf_name)
    proba = classifier["model"].predict_proba(df[classifier["features"]])[:, 1]

    # 풍력 + 수요예측이면 같은 확률에 조건부 제어량을 곱해 기댓값을 만든다.
    # 두 값이 같은 확률을 쓰므로 expected / probability = 조건부 제어량이 성립한다.
    expected = [None] * 24
    if use_demand:
        stage2 = _optional("curtailment_stage2_wind")
        if stage2 is not None:
            conditional = np.clip(stage2["model"].predict(df[stage2["features"]]), 0, None)
            # 주의: 기댓값이라 개별 시간의 제어량 크기가 아니다 — ESS 계산 금지(ESS_WARNING)
            expected = (proba * conditional).tolist()

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
        note = ("demand_forecast_mw가 없어 수요 미포함 모델(classifier_wind_calibrated)을 사용했고, "
                "제어량 조건부 회귀는 수요 피처가 필요해 expected_curtailment_mwh를 제공하지 않습니다.")
    elif use_demand and expected[0] is not None:
        note = ESS_WARNING
    elif use_demand:
        note = ("조건부 회귀 아티팩트(curtailment_stage2_wind)가 없어 확률만 반환하고 "
                "expected_curtailment_mwh는 null입니다. "
                "`python -m app.training.train_curtailment_regressor`로 학습하세요.")

    return PredictResponse(
        energy_type=energy_type, region=req.region, target_date=req.target_date,
        hourly=hourly, model_used=clf_name, note=note,
    )
