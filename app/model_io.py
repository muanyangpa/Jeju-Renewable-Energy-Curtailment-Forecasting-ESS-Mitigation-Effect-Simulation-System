"""
모델 아티팩트 저장/로드 — 재현성을 위해 메타데이터(학습 구간·피처·라이브러리 버전·학습 시각)를 함께 저장한다.

joblib로 저장한 scikit-learn 모델은 저장할 때와 로드할 때 scikit-learn 버전이 다르면
로드가 실패하거나 결과가 조용히 달라질 수 있다. 로드 시 버전이 다르면 경고를 출력한다.
"""
from __future__ import annotations

import os
import platform
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def save_artifact(name: str, model, features: list[str], **meta) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    artifact = {
        "model": model,
        "features": list(features),
        "meta": {
            "name": name,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
            **meta,
        },
    }
    joblib.dump(artifact, path)
    return path


def load_artifact(name: str) -> dict:
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 없음 — 먼저 `python -m app.training.train_converter` 등으로 학습을 실행하세요"
        )
    artifact = joblib.load(path)
    meta = artifact.get("meta")
    if meta is None:
        warnings.warn(f"{name}: 메타데이터 없는 구버전 아티팩트 — 재학습을 권장합니다", stacklevel=2)
    elif meta.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"{name}: 학습 시 scikit-learn {meta.get('sklearn_version')} / 현재 {sklearn.__version__} — "
            f"버전을 맞추거나 재학습하세요", stacklevel=2,
        )
    return artifact
