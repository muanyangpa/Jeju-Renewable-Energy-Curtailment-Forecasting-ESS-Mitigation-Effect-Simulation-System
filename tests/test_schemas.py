"""API 계약 — schemas.py가 발행하는 도메인 에러코드. 데이터·아티팩트 불필요."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import EssSimulateRequest, PredictRequest

W = [{"hour": h, "wind_speed": 6.0, "solar_rad": 1.0, "temp": 20.0, "cloud": 3.0}
     for h in range(1, 25)]


def _code(exc: ValidationError) -> str:
    """ValueError("<CODE>:<message>") 규약 — main.py가 이 앞부분으로 error_code를 만든다."""
    return str(exc).split("Value error, ")[1].split(":")[0]


def test_24시간_정상():
    r = PredictRequest(energy_type="wind", region="한림읍", target_date="2026-03-15", weather=W)
    assert len(r.weather) == 24


@pytest.mark.parametrize("weather,code", [
    (W[:23], "INVALID_HOUR_SET"),
    (W[:23] + [W[0]], "INVALID_HOUR_SET"),        # 중복
])
def test_시간_집합_위반(weather, code):
    with pytest.raises(ValidationError) as e:
        PredictRequest(energy_type="wind", region="x", target_date="2026-03-15", weather=weather)
    assert _code(e.value) == code


def test_발전원별_필수_기상값():
    no_wind = [{**w, "wind_speed": None} for w in W]
    with pytest.raises(ValidationError) as e:
        PredictRequest(energy_type="wind", region="x", target_date="2026-03-15", weather=no_wind)
    assert _code(e.value) == "MISSING_REQUIRED_FIELD"

    no_rad = [{**w, "solar_rad": None} for w in W]
    with pytest.raises(ValidationError) as e:
        PredictRequest(energy_type="solar", region="x", target_date="2026-03-15", weather=no_rad)
    assert _code(e.value) == "MISSING_REQUIRED_FIELD"


def test_풍력에는_일사량이_필수가_아니다():
    """교차 피처 도입 후에도 태양광 기상값은 optional이어야 한다 — 없으면 기존 모델로 전환된다."""
    wind_only = [{"hour": h, "wind_speed": 6.0} for h in range(1, 25)]
    PredictRequest(energy_type="wind", region="x", target_date="2026-03-15", weather=wind_only)


def test_수요예측_길이():
    with pytest.raises(ValidationError) as e:
        PredictRequest(energy_type="wind", region="x", target_date="2026-03-15",
                       weather=W, demand_forecast_mw=[500.0] * 23)
    assert _code(e.value) == "INVALID_HOUR_SET"


def test_ess_기본값은_저장용량_제약_방식이다():
    """기본 method가 바뀌면 문서의 대표값(52.5%)과 API 응답이 어긋난다."""
    r = EssSimulateRequest(hourly_curtailment_mwh=[10.0, 20.0])
    assert r.method == "storage_constrained"
    assert r.rated_power_mw == 65.0
    assert r.round_trip_efficiency == 0.90 and (r.soc_min, r.soc_max) == (0.10, 0.90)


@pytest.mark.parametrize("kw,code", [
    ({"soc_min": 0.9, "soc_max": 0.1}, "INVALID_SOC_RANGE"),
    ({"energy_capacity_mwh": 0.0}, "INVALID_ENERGY_CAPACITY"),
    ({"hour_of_day": [1]}, "LENGTH_MISMATCH"),
])
def test_ess_검증(kw, code):
    with pytest.raises(ValidationError) as e:
        EssSimulateRequest(hourly_curtailment_mwh=[10.0, 20.0], **kw)
    assert _code(e.value) == code
