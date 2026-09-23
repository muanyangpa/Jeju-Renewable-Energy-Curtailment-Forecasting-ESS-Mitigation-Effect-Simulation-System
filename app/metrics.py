"""
분류 평가 지표 공통 모듈 — 학습·블록CV·파이프라인 평가 스크립트가 모두 이 함수를 쓴다.

상위5%포착률(top5_capture)은 계획서 05장 정의 그대로 유지하되, 해석을 돕는 지표를 함께 보고한다.
  - top5_ceiling: 상위5%를 전부 맞혀도 도달 가능한 최대 포착률 = min(1, k / 양성 수)
      양성 비율이 5%보다 높으면 이 값이 1보다 작아진다. 측정값이 이 값에 붙어 있으면
      모델 간 비교가 불가능하다(수정 전 코드에서 실제로 이 문제가 있었음).
  - top5_precision: 상위5%로 뽑은 시간 중 실제 출력제어 비율
  - pr_auc: 불균형 데이터에서 AUC보다 민감한 순위 지표(average precision)
  - brier: 확률값 자체의 정확도(낮을수록 좋음). ESS 결정에 확률을 직접 쓰므로 함께 본다.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

TOP_FRACTION = 0.05


def _k(n: int) -> int:
    return max(1, int(math.ceil(n * TOP_FRACTION)))


def top5_capture(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """05장 정의: 상위 5% 안에 들어간 실제 양성 / 전체 양성."""
    y_true = np.asarray(y_true)
    if len(y_true) == 0 or y_true.sum() == 0:
        return float("nan")
    top_idx = np.argsort(-np.asarray(y_score), kind="stable")[: _k(len(y_true))]
    return float(y_true[top_idx].sum() / y_true.sum())


def top5_ceiling(y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(y_true) == 0 or y_true.sum() == 0:
        return float("nan")
    return float(min(1.0, _k(len(y_true)) / y_true.sum()))


def top5_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return float("nan")
    top_idx = np.argsort(-np.asarray(y_score), kind="stable")[: _k(len(y_true))]
    return float(y_true[top_idx].mean())


def classification_report(y_true, y_score) -> dict:
    """모든 지표를 한 번에 계산해 dict로 반환 (소수 4자리 반올림)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    has_both = 0 < y_true.sum() < len(y_true)
    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "pos_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "auc": float(roc_auc_score(y_true, y_score)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_both else float("nan"),
        "brier": float(brier_score_loss(y_true, y_score)) if len(y_true) else float("nan"),
        "top5_capture": top5_capture(y_true, y_score),
        "top5_ceiling": top5_ceiling(y_true),
        "top5_precision": top5_precision(y_true, y_score),
    }
    return {k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v) for k, v in out.items()}


def saturation_warning(report: dict, tol: float = 0.01) -> str | None:
    """측정값이 상한에 붙어 있으면 경고 문구를 반환 (모델 비교가 무의미하다는 신호)."""
    cap, ceil = report.get("top5_capture"), report.get("top5_ceiling")
    if isinstance(cap, float) and isinstance(ceil, float) and ceil < 1.0 and cap >= ceil - tol:
        return (f"top5_capture({cap})가 이론상 최대값({ceil})에 도달 — 이 지표로는 모델 간 차이를 판별할 수 없음. "
                f"pr_auc·top5_precision으로 비교할 것")
    return None
