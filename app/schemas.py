"""
Backend(Java/Spring Boot) <-> AI 서버(Python) REST 계약.

이 파일이 곧 API 계약 문서다 — 09장 리스크 대응 전략("API 계약을 코드보다 문서로 먼저 고정")에 따라,
필드명·타입·단위·에러코드를 여기 한 곳에서만 정의하고 Backend 쪽에는 이 스키마를 그대로 전달한다.

검증 책임 분담 (09장 반영 그대로):
  - Backend: 필드 존재 여부·기본 타입만 확인 (JSON 파싱 레벨)
  - AI 서버: 도메인 규칙 검증 — "weather 배열이 정확히 24개, 시간 중복 없음" 등
    위반 시 HTTP 422 + error_code="INVALID_HOUR_SET"
"""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

EnergyType = Literal["solar", "wind"]


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
        None, description="24개, 풍력 전용(수요 정식 채택 피처, 05장). 태양광은 무시됨"
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


class HourlyPrediction(BaseModel):
    hour: int
    generation_forecast_mwh: float
    curtailment_probability: float
    expected_curtailment_mwh: Optional[float] = Field(
        None, description="풍력만 값 존재. 태양광은 07장 사유로 null 고정"
    )


class PredictResponse(BaseModel):
    energy_type: EnergyType
    region: str
    target_date: date
    hourly: list[HourlyPrediction]
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
