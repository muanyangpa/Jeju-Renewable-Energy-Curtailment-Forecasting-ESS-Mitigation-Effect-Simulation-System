"""서빙 모델 진단 — 신뢰도 곡선과 순열 중요도.

[왜 필요한가]
이 저장소는 확률을 sigmoid로 보정하고 Brier가 좋아졌다는 것까지 확인했다. 그런데 Brier는
한 개의 요약값이라 '어느 확률 구간에서 얼마나 어긋나는지'를 숨긴다. 신뢰도 곡선(reliability
curve)을 보면 보정 후에도 중간 구간이 몇 배씩 어긋나는 것이 드러난다 — 그것이 운영 임계값이
작동하지 않는 근본 이유다.

순열 중요도는 '어느 피처가 실제로 쓰이는지'를 본다. 이걸 한 번만 봤다면 풍력 모델에서
태양광이 빠진 것을 훨씬 일찍 찾았을 것이다(실제로 capacity_factor·penetration의 중요도가
음수로 나와 죽은 피처임이 드러났고, 그래서 crossp 경로에서 제외했다).

실행: python -m app.training.diagnose_models
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from app.model_io import MODELS_DIR, load_artifact
from app.serving_config import artifact_name
from app.training.train_classifier import TEST_END, TEST_START, build_dataset

# 신뢰도 곡선 구간. 확률이 0 근처에 몰려 있으므로 등간격이 아니라 로그 느낌으로 자른다.
BINS = (0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.01)

PATHS = (("solar", False, False), ("solar", True, False),
         ("wind", False, False), ("wind", True, False), ("wind", True, True))


def _served(et: str, ud: bool, uc: bool):
    cross = "converter" if uc else None
    df, _ = build_dataset(et, ud, cross)
    te = df[(df.dt >= TEST_START) & (df.dt < TEST_END)]
    art = load_artifact(artifact_name(et, ud, uc))
    y = te["is_curtailed"].to_numpy().astype(int)
    p = art["model"].predict_proba(te[art["features"]])[:, 1]
    return art, te, y, p


def reliability(y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        pred, obs = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": int(m.sum()),
                     "mean_predicted": round(pred, 4), "observed": round(obs, 4),
                     "ratio_obs_over_pred": round(obs / pred, 2) if pred > 0 else None,
                     "n_pos": int(y[m].sum())})
    return rows


def main() -> None:
    rel_rows, imp_rows = [], []
    for et, ud, uc in PATHS:
        art, te, y, p = _served(et, ud, uc)
        name = art["meta"]["name"]
        fam = art["meta"].get("model_family", "rf")

        print("=" * 84)
        print(f"[{name}]  모델군 {fam} · 피처 {len(art['features'])}개 · "
              f"양성 {y.sum()}시간")
        print("  신뢰도 곡선 — 예측 확률 구간별 실제 발생률")
        print(f"  {'구간':16} {'n':>6} {'양성':>5} {'평균예측':>9} {'실제':>8} {'실제/예측':>9}")
        for r in reliability(y, p):
            flag = ""
            if r["ratio_obs_over_pred"] is not None:
                if r["ratio_obs_over_pred"] >= 2:
                    flag = "  <- 과소예측"
                elif r["ratio_obs_over_pred"] <= 0.5:
                    flag = "  <- 과대예측"
            print(f"  {r['bin']:16} {r['n']:6d} {r['n_pos']:5d} {r['mean_predicted']:9.4f} "
                  f"{r['observed']:8.4f} {str(r['ratio_obs_over_pred']):>9}{flag}")
            rel_rows.append({"model": name, "family": fam, **r})

        # 순열 중요도. PR-AUC 기준 — 이 제품이 상위 구간 품질을 보므로 정확도보다 적절하다.
        r = permutation_importance(art["model"], te[art["features"]], y, n_repeats=10,
                                   random_state=0, scoring="average_precision", n_jobs=-1)
        print("  순열 중요도 (PR-AUC 감소분, 10회)")
        for i in np.argsort(-r.importances_mean):
            f = art["features"][i]
            mean, std = float(r.importances_mean[i]), float(r.importances_std[i])
            dead = "  <- 죽은 피처(음수)" if mean < 0 else ""
            print(f"    {f:20} {mean:+.4f} ± {std:.4f}{dead}")
            imp_rows.append({"model": name, "family": fam, "feature": f,
                             "importance_mean": round(mean, 4), "importance_std": round(std, 4)})
        print()

    pd.DataFrame(rel_rows).to_csv(os.path.join(MODELS_DIR, "reliability_curves.csv"), index=False)
    pd.DataFrame(imp_rows).to_csv(os.path.join(MODELS_DIR, "permutation_importance.csv"), index=False)
    print(f"저장: {MODELS_DIR}/reliability_curves.csv, permutation_importance.csv")

    dead = pd.DataFrame(imp_rows)
    dead = dead[dead.importance_mean < 0]
    if len(dead):
        print("\n⚠ 중요도가 음수인 피처 — 제외를 검토할 것 (serving_config.FEATURE_DROP):")
        print(dead[["model", "feature", "importance_mean"]].to_string(index=False))

    bad = pd.DataFrame(rel_rows)
    bad = bad[(bad.n >= 50) & ((bad.ratio_obs_over_pred >= 2) | (bad.ratio_obs_over_pred <= 0.5))]
    if len(bad):
        print("\n⚠ 실제/예측이 2배 이상 어긋나는 구간(n>=50) — 확률의 '크기'를 쓰는 표시는 금지:")
        print(bad[["model", "bin", "n", "mean_predicted", "observed",
                   "ratio_obs_over_pred"]].to_string(index=False))


if __name__ == "__main__":
    main()
