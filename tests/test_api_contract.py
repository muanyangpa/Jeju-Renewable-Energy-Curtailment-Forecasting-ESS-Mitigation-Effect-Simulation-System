"""Backend ↔ AI 서버 REST 계약 — docs/통신규격_확정서_v1.1.md가 약속한 동작을 고정한다.

계약 문서와 구현이 갈리면 연동이 조용히 깨진다. 문서에 적은 응답 모양·에러코드·조건부 필수
규칙을 여기서 검증하므로, 이 파일이 깨지면 문서도 함께 고쳐야 한다.

Spring 쪽에서 같은 케이스를 호출해 응답을 비교하면 연동 테스트가 된다.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import needs_artifacts

client = TestClient(app, raise_server_exceptions=False)
FULL = [{"hour": h, "temp": 15.0, "cloud": 4.0, "wind_speed": 6.0, "solar_rad": 0.5}
        for h in range(1, 25)]


def body(**kw) -> dict:
    return {"energy_type": "wind", "region": "제주", "target_date": "2026-10-01",
            "weather": FULL, **kw}


def post(**kw):
    return client.post("/predict", json=body(**kw))


class TestErrorShape:
    """모든 4xx는 {error_code, message} 한 가지 모양이다."""

    @pytest.mark.parametrize("payload,code", [
        ({"weather": FULL[:23]}, "INVALID_HOUR_SET"),
        ({"weather": FULL[:23] + [FULL[0]]}, "INVALID_HOUR_SET"),
        ({"demand_forecast_mw": [600.0] * 12}, "INVALID_HOUR_SET"),
        ({"region": "   "}, "MISSING_REQUIRED_FIELD"),
        ({"energy_type": "windd"}, "VALIDATION_ERROR"),
    ])
    def test_에러코드와_모양(self, payload, code):
        r = post(**payload)
        assert r.status_code == 422
        j = r.json()
        assert set(j) == {"error_code", "message"}, j
        assert j["error_code"] == code, j
        assert j["message"], "메시지가 비어 있으면 릴레이해도 쓸모가 없다"

    def test_형식오류는_필드_위치를_알려준다(self):
        """pydantic 기본 메시지는 'Field required'뿐이라 어느 필드인지 알 수 없다 (v1.1에서 추가)."""
        r = client.post("/predict", json={"energy_type": "wind", "region": "제주",
                                          "target_date": "2026-10-01"})
        assert r.status_code == 422
        assert "위치: weather" in r.json()["message"], r.json()


@needs_artifacts
class TestConditionalRequired:
    """Optional은 '널을 보내도 된다'가 아니다 — energy_type이 필수를 결정한다."""

    @pytest.mark.parametrize("energy_type,drop", [
        ("wind", "wind_speed"),
        ("solar", "solar_rad"), ("solar", "temp"), ("solar", "cloud"),
    ])
    def test_발전원별_필수값이_null이면_거부(self, energy_type, drop):
        w = [{**x, drop: None} for x in FULL]
        r = post(energy_type=energy_type, weather=w)
        assert r.status_code == 422
        j = r.json()
        assert j["error_code"] == "MISSING_REQUIRED_FIELD"
        assert drop in j["message"], j

    @pytest.mark.parametrize("energy_type,drop", [
        ("wind", "solar_rad"), ("wind", "cloud"), ("solar", "wind_speed"),
    ])
    def test_그_발전원에_불필요한_값은_null이어도_통과(self, energy_type, drop):
        r = post(energy_type=energy_type, weather=[{**x, drop: None} for x in FULL])
        assert r.status_code == 200, r.json()

    def test_누락_위치를_시각과_필드로_알려준다(self):
        w = FULL[:12] + [{**x, "wind_speed": None} for x in FULL[12:]]
        r = post(weather=w)
        assert "13시.wind_speed" in r.json()["message"], r.json()

    def test_null을_0으로_대체하지_않는다(self):
        """풍속 null을 0으로 바꾸면 '제어 없음'으로 오예측한다 — 실제로 겪어 고친 동작이다."""
        r = post(weather=[{**x, "wind_speed": None} for x in FULL])
        assert r.status_code == 422, "조용히 통과하면 안 된다"


@needs_artifacts
class TestRegion:
    """자유 문자열, 모델은 쓰지 않고 그대로 echo. 빈 값만 거부."""

    @pytest.mark.parametrize("v", ["제주", "남원읍", "제주특별자치도 서귀포시 남원읍", "JEJU"])
    def test_어떤_문자열이든_받고_그대로_돌려준다(self, v):
        r = post(region=v)
        assert r.status_code == 200 and r.json()["region"] == v

    @pytest.mark.parametrize("v", ["", "   ", "x" * 101])
    def test_빈_값과_과길이는_거부(self, v):
        assert post(region=v).status_code == 422

    def test_region이_예측을_바꾸지_않는다(self):
        """제주 전역 단일 모델이므로 지역으로 분기하지 않는다."""
        a = post(region="제주").json()
        b = post(region="남원읍").json()
        assert [h["curtailment_probability"] for h in a["hourly"]] == \
               [h["curtailment_probability"] for h in b["hourly"]]


@needs_artifacts
class TestResponseShape:
    def test_필수_키가_모두_있다(self):
        j = post().json()
        assert {"energy_type", "region", "target_date", "hourly", "model_used"} <= set(j)
        assert len(j["hourly"]) == 24
        assert {"hour", "generation_forecast_mwh", "curtailment_probability"} <= set(j["hourly"][0])

    def test_태양광은_expected_curtailment가_null이다(self):
        """07장 사유 — 태양광은 제어량이 집계되지 않아 2단계 모델이 없다."""
        j = post(energy_type="solar", demand_forecast_mw=[600.0] * 24).json()
        assert all(h["expected_curtailment_mwh"] is None for h in j["hourly"])

    def test_태양광_기상값을_함께_주면_침투율_모델로_전환된다(self):
        with_solar = post(demand_forecast_mw=[600.0] * 24).json()["model_used"]
        without = post(demand_forecast_mw=[600.0] * 24,
                       weather=[{"hour": x["hour"], "wind_speed": x["wind_speed"]} for x in FULL]
                       ).json()["model_used"]
        assert "crossp" in with_solar and "crossp" not in without


def test_health는_200이다():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
