"""테스트 공통 설정.

이 저장소는 데이터(data/)와 모델 아티팩트(models/)가 .gitignore 대상이다. 그래서 테스트를
두 층으로 나눈다.

  순수 로직 테스트  — 데이터·아티팩트 없이 돈다. CI에서 항상 실행된다.
  아티팩트 테스트    — 학습된 .joblib이나 원본 CSV가 필요하다. 없으면 skip한다.

아티팩트가 없을 때 조용히 통과하는 게 아니라 'skipped'로 보여야 한다 — 통과로 보이면
CI가 아무것도 검증하지 않는데 초록불이 뜬다.
"""
from __future__ import annotations

import os

import pytest

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def _has(name: str) -> bool:
    return os.path.exists(os.path.join(MODELS_DIR, f"{name}.joblib"))


needs_artifacts = pytest.mark.skipif(
    not (_has("converter_solar") and _has("classifier_solar_calibrated_sigmoid")),
    reason="학습된 아티팩트가 없습니다 — `python -m app.training.train_classifier` 후 실행하세요",
)

needs_data = pytest.mark.skipif(
    not os.path.isdir(os.environ.get("DATA_DIR", os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "raw"))),
    reason="원본 데이터가 없습니다 — DATA_DIR을 지정하세요",
)
