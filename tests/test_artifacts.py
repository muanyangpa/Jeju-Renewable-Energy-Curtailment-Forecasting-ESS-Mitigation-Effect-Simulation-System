"""아티팩트 정합성 — 학습된 .joblib이 없으면 skip한다.

README가 명시한 불변식을 코드로 고정한다. 문서에만 적혀 있으면 재학습 때 조용히 깨진다.
"""
from __future__ import annotations

import pytest

from app.model_io import load_artifact
from app.serving_config import SERVED_FAMILY, artifact_name, features_for, path_key
from tests.conftest import needs_artifacts

pytestmark = needs_artifacts

# serving_config에서 유도한다 — 하드코딩하면 모델군을 바꿀 때마다 테스트가 깨진다.
PATHS = (("solar", False, False), ("solar", True, False),
         ("wind", False, False), ("wind", True, False), ("wind", True, True))
SERVED = tuple(artifact_name(*p) for p in PATHS)


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
    cross = load_artifact(artifact_name("wind", True, True))
    plain = load_artifact(artifact_name("wind", True, False))
    assert "total_penetration" in cross["features"]
    assert "total_penetration" not in plain["features"]


def test_태양광_서빙_모델에는_교차_피처가_없다():
    """태양광은 교차 피처로 top5가 나빠졌다(0.8475 -> 0.8333) — 실수로 서빙되면 안 된다."""
    assert "total_penetration" not in load_artifact(artifact_name("solar", True, False))["features"]


@pytest.mark.parametrize("path", PATHS)
def test_서빙_아티팩트의_모델군이_설정과_일치한다(path):
    """serving_config가 lr을 가리키는데 rf 아티팩트가 로드되면 조용히 다른 모델이 서빙된다."""
    key = path_key(*path)
    meta = load_artifact(artifact_name(*path)).get("meta") or {}
    assert meta.get("model_family") == SERVED_FAMILY[key], (
        f"{key}: 아티팩트 {meta.get('model_family')} != 설정 {SERVED_FAMILY[key]} — 재학습하세요")
    assert meta.get("served_by_predict") is True, f"{key}: served_by_predict가 True가 아니다"


def test_죽은_피처가_서빙_모델에서_제외됐다():
    """순열 중요도가 음수였던 capacity_factor·penetration이 crossp 경로에 남아 있으면 안 된다."""
    feats = load_artifact(artifact_name("wind", True, True))["features"]
    for f in ("capacity_factor", "penetration"):
        assert f not in feats, f"crossp 경로에 죽은 피처 {f}가 남아 있다"
    assert "total_penetration" in feats


def test_서빙_모델에_신뢰구간이_저장돼_있다():
    """소수점 4자리 점추정만 저장하면 문서가 존재하지 않는 정밀도를 인용하게 된다."""
    meta = load_artifact(artifact_name("wind", True, True)).get("meta") or {}
    ci = meta.get("test_report_ci")
    assert ci, "test_report_ci가 없다 — train_classifier를 재실행하세요"
    for m in ("auc", "pr_auc", "top5_capture"):
        assert ci[m]["lo"] <= ci[m]["point"] <= ci[m]["hi"], f"{m}: 점추정이 구간 밖"


def test_stage2의_stage1이_서빙_모델과_같다():
    """expected = probability x 조건부가 성립하려면 stage1이 /predict가 쓰는 모델이어야 한다."""
    from app.services.predict import CROSS_ARTIFACT
    from app.training.train_curtailment_regressor import STAGE1_NAME
    assert STAGE1_NAME == CROSS_ARTIFACT


def test_경로별_임계값_파일이_모든_서빙_경로를_덮는다():
    """select_thresholds.py를 돌리지 않고 재학습만 하면 임계값이 옛 확률 척도에 남는다.

    파일이 아예 없으면 skip한다(아직 안 돌린 상태). 있으면 다섯 경로가 모두 있어야 한다.
    """
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "models", "operational_thresholds.json")
    if not os.path.exists(path):
        pytest.skip("operational_thresholds.json이 없습니다 — "
                    "`python -m app.training.select_thresholds`를 실행하세요")
    with open(path, encoding="utf-8") as f:
        thr = json.load(f)
    missing = [n for n in SERVED if n not in thr]
    assert not missing, f"임계값이 없는 서빙 경로: {missing}"
    for name, v in thr.items():
        assert 0.0 < v["threshold"] < 1.0, f"{name}: 임계값 {v['threshold']}이 (0,1) 밖"
        assert "reliable" in v, f"{name}: reliable 플래그가 없다"
