"""
Backend(Java/Spring Boot) <-> AI 서버(Python) REST 계약.

이 파일이 곧 API 계약 문서다 — 09장 리스크 대응 전략("API 계약을 코드보다 문서로 먼저 고정")에 따라,
필드명·타입·단위·에러코드를 여기 한 곳에서만 정의하고 Backend 쪽에는 이 스키마를 그대로 전달한다.

검증 책임 분담 (09장 반영 그대로):
  - Backend: 필드 존재 여부·기본 타입만 확인 (JSON 파싱 레벨)
  - AI 서버: 도메인 규칙 검증 — "weather 배열이 정확히 24개, 시간 중복 없음" 등
    위반 시 HTTP 422 + error_code="INVALID_HOUR_SET"
  - AI 서버: 발전원별 필수 기상값 누락 검증 [신규]
    위반 시 HTTP 422 + error_code="MISSING_REQUIRED_FIELD"
    (수정 전에는 누락값을 조용히 0으로 바꿔 예측했다 — 예: 풍속 null -> 풍속 0 -> '제어 없음'으로 오예측)

에러 메시지 규약: ValueError("<ERROR_CODE>:<사람이 읽는 메시지>") — main.py가 코드와 메시지를 분리한다.
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

EnergyType = Literal["solar", "wind"]

# 발전원별 컨버터 필수 입력 (app/training/train_converter.py의 피처와 일치해야 함)
REQUIRED_WEATHER_FIELDS: dict[str, tuple[str, ...]] = {
    "solar": ("solar_rad", "temp", "cloud"),
    "wind": ("wind_speed",),
}


class WeatherHour(BaseModel):
    """시간별 기상예보 1건. hour는 1~24(24시=자정, 원본 공공데이터 관례와 동일)."""
    hour: int = Field(..., ge=1, le=24, description="1~24시")
    solar_rad: Optional[float] = Field(None, description="일사량(MJ/m2) — 태양광 필수")
    temp: Optional[float] = Field(None, description="기온(°C)")
    cloud: Optional[float] = Field(None, description="전운량(0~10) — 태양광 권장")
    wind_speed: Optional[float] = Field(None, description="풍속(m/s) — 풍력 필수")


class PredictRequest(BaseModel):
    energy_type: EnergyType
    region: str = Field(..., description="예: '남원읍' — 참고용, 모델 선택에는 미사용(province-wide 모델)")
    target_date: date
    weather: list[WeatherHour] = Field(..., description="정확히 24개, 1~24시 각 1회")
    demand_forecast_mw: Optional[list[float]] = Field(
        None, description="24개(선택). 있으면 침투율(발전량/수요) 포함 모델을 사용한다 — 태양광·풍력 모두 정확도가 올라간다(태양광 PR-AUC 0.639->0.711). 풍력은 수요 자체도 정식 피처다(05장). 생략하면 미포함 모델로 자동 전환되고 풍력의 expected_curtailment_mwh는 null이 된다"
    )

    @field_validator("weather")
    @classmethod
    def _check_24_hours(cls, v: list[WeatherHour]) -> list[WeatherHour]:
        if len(v) != 24:
            raise ValueError("INVALID_HOUR_SET:weather 배열은 정확히 24개여야 합니다")
        hours = sorted(h.hour for h in v)
        if hours != list(range(1, 25)):
            raise ValueError("INVALID_HOUR_SET:weather 배열의 시간이 1~24시 각 1회가 아닙니다(중복 또는 누락)")
        return v

    @model_validator(mode="after")
    def _check_demand_length(self) -> "PredictRequest":
        if self.demand_forecast_mw is not None and len(self.demand_forecast_mw) != 24:
            raise ValueError("INVALID_HOUR_SET:demand_forecast_mw는 24개여야 합니다")
        return self

    @model_validator(mode="after")
    def _check_required_weather(self) -> "PredictRequest":
        required = REQUIRED_WEATHER_FIELDS[self.energy_type]
        missing = sorted(
            {f"{w.hour}시.{f}" for w in self.weather for f in required if getattr(w, f) is None},
            key=lambda s: (int(s.split("시")[0]), s),
        )
        if missing:
            preview = ", ".join(missing[:5]) + (f" 외 {len(missing) - 5}건" if len(missing) > 5 else "")
            raise ValueError(
                f"MISSING_REQUIRED_FIELD:{self.energy_type} 예측에는 {', '.join(required)}가 모든 시간에 필요합니다 "
                f"(누락: {preview})"
            )
        return self


class HourlyPrediction(BaseModel):
    hour: int
    generation_forecast_mwh: float
    curtailment_probability: float = Field(
        ..., description="출력제어 발생 확률 0~1. sigmoid 보정된 값이라 보정 전 모델과 크기가 "
                         "다르다. 0.5 같은 고정 임계값을 쓰지 말 것(README '운영 임계값' 참고). "
                         "산출 모델은 model_used 참고"
    )
    expected_curtailment_mwh: Optional[float] = Field(
        None,
        description=(
            "제어량 기댓값 = curtailment_probability x E[제어량|제어 발생] (2단계 모델). "
            "같은 확률을 쓰므로 expected_curtailment_mwh / curtailment_probability = 조건부 제어량이 된다. "
            "풍력 + demand_forecast_mw가 있을 때만 값이 존재하고, 태양광은 07장 사유로 null 고정. "
            "⚠ 이 값은 '기댓값'이라 개별 시간의 제어량 크기가 아니다 — /ess/simulate의 "
            "hourly_curtailment_mwh로 넘기지 말 것. 2023년 검증에서 ESS 흡수율이 "
            "실측 36.8% 대비 80.6%로 과대평가됐다(README '알려진 한계')."
        ),
    )


class PredictResponse(BaseModel):
    energy_type: EnergyType
    region: str
    target_date: date
    hourly: list[HourlyPrediction]
    model_used: str = Field(
        ..., description="curtailment_probability를 산출한 sigmoid 보정 모델 이름. "
                         "demand_forecast_mw가 있으면 'classifier_{solar|wind}_demand_calibrated_sigmoid', "
                         "없으면 'classifier_{solar|wind}_calibrated_sigmoid'. "
                         "값을 하드코딩해 분기하지 말 것 — 보정 방식이 바뀌면 이름도 바뀐다"
    )
    note: Optional[str] = Field(
        None, description="태양광 응답에는 expected_curtailment_mwh가 null인 이유를 항상 포함"
    )


class EssSimulateRequest(BaseModel):
    hourly_curtailment_mwh: list[float] = Field(..., description="시간별 출력제어량(MWh), 24개 또는 임의 길이")
    rated_power_mw: float = Field(22.5, description="ESS 정격출력(MW) — 04장 03번 슬라이더 값")
    method: Literal["hourly_capped", "naive_upper_bound"] = "hourly_capped"


class EssSimulateResponse(BaseModel):
    rated_power_mw: float
    method: str
    total_curtailment_mwh: float
    total_absorbed_mwh: float
    absorption_rate: float


class ErrorResponse(BaseModel):
    error_code: str
    message: str
