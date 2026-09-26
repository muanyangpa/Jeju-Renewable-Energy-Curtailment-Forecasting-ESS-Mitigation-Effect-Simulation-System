"""아티팩트 정합성 — 학습된 .joblib이 없으면 skip한다.

README가 명시한 불변식을 코드로 고정한다. 문서에만 적혀 있으면 재학습 때 조용히 깨진다.
"""
from __future__ import annotations

import pytest

from app.model_io import load_artifact
from tests.conftest import needs_artifacts

pytestmark = needs_artifacts

SERVED = (
    "classifier_solar_calibrated_sigmoid",
    "classifier_solar_demand_calibrated_sigmoid",
    "classifier_wind_calibrated_sigmoid",
    "classifier_wind_demand_calibrated_sigmoid",
    "classifier_wind_demand_crossp_calibrated_sigmoid",
)


@pytest.mark.parametrize("name", SERVED)
def test_서빙_분류기는_보정본이고_대리지표를_갖는다(name):
    a = load_artifact(name)
    meta = a.get("meta") or {}
    assert meta.get("calibrated") is True, f"{name}: 보정 안 된 모델을 서빙할 수 없다"
    assert meta.get("capacity_proxy_mwh"), f"{name}: capacity_proxy_mwh가 없다"


@pytest.mark.parametrize("energy_type", ["solar", "wind"])
def test_컨버터와_분류기의_설비용량_대리지표가_같다(energy_type):
    """README '설비용량 대리지표의 정합성' — 컨버터가 곱하는 상수와 분류기가 나누는 상수가
    다르면 capacity_factor가 그 비율만큼 왜곡된다. 문서에만 적혀 있던 불변식을 고정한다.
    """
    conv = (load_artifact(f"converter_{energy_type}").get("meta") or {})
    if conv.get("target") != "capacity_factor":
        pytest.skip(f"{energy_type} 컨버터는 MWh 타깃이라 대리지표를 곱하지 않는다")
    clf = (load_artifact(f"classifier_{energy_type}_calibrated_sigmoid").get("meta") or {})
    assert conv["capacity_proxy_mwh"] == clf["capacity_proxy_mwh"], (
        f"{energy_type}: 컨버터 {conv['capacity_proxy_mwh']} != 분류기 {clf['capacity_proxy_mwh']} — "
        "둘을 같은 실행에서 재학습하세요")


def test_교차_피처_모델만_합산_침투율을_쓴다():
    cross = load_artifact("classifier_wind_demand_crossp_calibrated_sigmoid")
    plain = load_artifact("classifier_wind_demand_calibrated_sigmoid")
    assert "total_penetration" in cross["features"]
    assert "total_penetration" not in plain["features"]


def test_태양광_서빙_모델에는_교차_피처가_없다():
    """태양광은 교차 피처로 top5가 나빠졌다(0.8475 -> 0.8333) — 실수로 서빙되면 안 된다."""
    a = load_artifact("classifier_solar_demand_calibrated_sigmoid")
    assert "total_penetration" not in a["features"]


def test_stage2의_stage1이_서빙_모델과_같다():
    """expected = probability x 조건부가 성립하려면 stage1이 /predict가 쓰는 모델이어야 한다."""
    from app.services.predict import CROSS_ARTIFACT
    from app.training.train_curtailment_regressor import STAGE1_NAME
    assert STAGE1_NAME == CROSS_ARTIFACT
