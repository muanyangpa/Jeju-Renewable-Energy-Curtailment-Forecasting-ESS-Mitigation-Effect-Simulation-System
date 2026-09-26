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


# ---------------------------------------------------------------------------
# 지표의 불확실성 — 일(day) 블록 부트스트랩
# ---------------------------------------------------------------------------

def _top5_capture(y: np.ndarray, p: np.ndarray) -> float:
    k = max(1, int(round(len(y) * 0.05)))
    idx = np.argsort(-p)[:k]
    return float(y[idx].sum() / y.sum()) if y.sum() else float("nan")


_BOOT_METRICS = {
    "auc": lambda y, p: float(roc_auc_score(y, p)),
    "pr_auc": lambda y, p: float(average_precision_score(y, p)),
    "top5_capture": _top5_capture,
}


def day_block_ci(y, p, days, metrics=("auc", "pr_auc", "top5_capture"),
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """날짜를 블록으로 리샘플한 부트스트랩 95% 신뢰구간.

    [왜 날짜 블록인가]
    출력제어는 날짜 단위로 뭉쳐 있다 — 2023년 풍력은 563시간이 117일에 분포하고 하루 평균
    4.8시간이다. 시간 단위로 리샘플하면 같은 날의 시간들을 독립으로 취급해 유효 표본수를
    부풀리고, 신뢰구간이 실제보다 1.6~2.1배 좁아진다. 지표를 소수점 4자리로 보고하려면
    그만큼의 정밀도가 있어야 하는데, 없다.

    y, p: 1차원 배열. days: 같은 길이의 날짜 배열(리샘플 단위).
    반환: {지표: {"point", "lo", "hi", "halfwidth"}}
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    days = np.asarray(days)
    uniq = np.unique(days)
    groups = {d: np.flatnonzero(days == d) for d in uniq}
    rng = np.random.default_rng(seed)

    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[d] for d in pick])
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue  # 양성이 없으면 AUC·PR-AUC가 정의되지 않는다
        pp = p[idx]
        for m in metrics:
            draws[m].append(_BOOT_METRICS[m](yy, pp))

    out = {}
    for m in metrics:
        v = np.asarray(draws[m])
        lo, hi = np.percentile(v, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        out[m] = {"point": round(_BOOT_METRICS[m](y, p), 4),
                  "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                  "halfwidth": round(float(hi - lo) / 2, 4)}
    return out


def paired_day_block_ci(y, p_a, p_b, days, metric: str = "pr_auc",
                        n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """두 모델의 지표 차이(A − B)에 대한 신뢰구간. 같은 날짜 리샘플을 공유해 짝지어 비교한다.

    짝지어야 하는 이유: 두 모델이 같은 데이터를 보므로 오차가 상관돼 있다. 독립 구간을
    겹쳐보면 실제로는 유의한 차이를 '겹친다'고 오판한다.
    """
    y = np.asarray(y).astype(int)
    a, b = np.asarray(p_a, dtype=float), np.asarray(p_b, dtype=float)
    days = np.asarray(days)
    uniq = np.unique(days)
    groups = {d: np.flatnonzero(days == d) for d in uniq}
    rng = np.random.default_rng(seed)
    fn = _BOOT_METRICS[metric]

    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[d] for d in pick])
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        diffs.append(fn(yy, a[idx]) - fn(yy, b[idx]))
    v = np.asarray(diffs)
    lo, hi = np.percentile(v, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    verdict = "A" if lo > 0 else "B" if hi < 0 else "tie"
    return {"metric": metric, "diff_point": round(fn(y, a) - fn(y, b), 4),
            "diff_median": round(float(np.median(v)), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4), "winner": verdict}
