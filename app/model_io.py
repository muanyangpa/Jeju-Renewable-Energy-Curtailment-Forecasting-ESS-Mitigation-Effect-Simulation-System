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


def feature_ranges(train: pd.DataFrame, features: list[str]) -> dict[str, list[float]]:
    """학습 구간 각 피처의 [min, max]. 서빙 입력이 이 밖으로 나가면 외삽이라는 신호다.

    [2026-09-26 정정] 처음에는 p1~p99를 썼다. '단일 이상치가 범위를 넓혀 드리프트를 놓친다'는
    이유였는데, 실측 입력으로 검증해 보니 그 선택이 감지기를 망가뜨렸다 — p1~p99는 **정의상
    학습 데이터의 2%가 범위 밖**이고, 피처 4개를 OR로 합치면 시간 기준 5.70%가 된다. 하루
    24시간 중 10%(2.4시간)를 넘는 날이 **학습 구간에서 21.2%**, 2023 테스트 구간에서 34.2%였다.
    모델이 AUC 0.986을 낸 구간에서 3분의 1이 '외삽'이라고 경고하는 감지기는 잡음이다.

    min~max로 바꾸면 학습 구간이 정의상 0.00%이고 2023년은 시간 기준 0.37%, 하루 기준 1.9%다.
    드리프트 감지기는 **학습 데이터에서 절대 울리지 않아야** 한다 — 그것이 기준선이다.
    이상치가 범위를 넓히는 문제는 남지만, 가상의 미탐을 걱정해 실재하는 오탐을 만든 것이
    더 나쁜 거래였다.

    시각 피처(sin/cos)는 순환값이라 범위를 벗어날 수 없어 _drift_note가 검사에서 건너뛴다.
    """
    return {f: [round(float(train[f].min()), 6), round(float(train[f].max()), 6)]
            for f in features}


def converter_predict_mwh(artifact: dict, X) -> np.ndarray:
    """컨버터 예측을 항상 MWh로 돌려준다.

    태양광 컨버터는 타깃이 '이용률'이라 그대로 쓰면 0~1 값이 나온다. 이걸 MWh로
    환산하지 않고 분류기에 넣으면 capacity_factor가 수백 배 작아져 예측이 무작위가 된다
    (실제로 이 실수로 태양광 서비스 경로 AUC가 0.99 -> 0.59로 무너진 적이 있다).
    서빙(predict.py)과 평가 스크립트가 같은 경로를 타도록 여기 한 곳에 모아 둔다.

    곱하는 상수는 분류기가 나눌 때 쓰는 값과 같으므로, 분류기 입력 기준으로는 정확히
    상쇄된다 — 즉 상수의 절대값이 분류 성능에 영향을 주지 않는다.
    """
    raw = np.clip(artifact["model"].predict(X[artifact["features"]]), 0, None)
    meta = artifact.get("meta") or {}
    if meta.get("target") == "capacity_factor":
        proxy = meta.get("capacity_proxy_mwh")
        if not proxy:
            raise RuntimeError(
                f"{meta.get('name')}: 이용률 타깃인데 capacity_proxy_mwh가 없습니다 — "
                f"`python -m app.training.train_converter`로 재학습하세요"
            )
        raw = raw * float(proxy)
    return raw


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
