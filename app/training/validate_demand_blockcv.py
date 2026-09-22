"""
J장 방식 leave-block-out 6블록 교차검증 재현 — 풍력 수요피처 "정식 채택" 재확인.

배경: README/09장에 기록된 알려진 한계 — 단일 2023년 테스트 분할에서는 수요피처 포함 분류기의
top5포착률(0.233)이 미포함(0.236)보다 오히려 낮게 나와, J장이 6블록 교차검증으로 확정한
"정식 채택"(0.377->0.493) 결과와 모순되는 것처럼 보였다. 이 스크립트는 J장과 동일한 방법론
(6개월 단위 6블록, leave-block-out, 가중평균 집계)을 프로덕션 코드/데이터로 재현해 어느 쪽이
맞는지(혹은 둘 다 맞고 조건에 따라 달라지는지) 확인한다.

블록 경계(J장과 동일): 21.01-07, 21.07-22.01, 22.01-07, 22.07-23.01, 23.01-07, 23.07-24.01
각 블록을 테스트로, 나머지 5블록을 학습으로 사용 (leave-block-out).
집계는 블록별 curtailed(양성) 시간 수로 가중평균한다 — J장과 동일하게 표본이 적은 블록의
잡음이 전체 결론을 왜곡하지 않도록 하기 위함.

실행: python -m app.training.validate_demand_blockcv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.data_prep import add_time_features, load_curtailment, load_demand_actual, load_generation_actual

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

BLOCKS = [
    ("2021-01-01", "2021-07-01"),
    ("2021-07-01", "2022-01-01"),
    ("2022-01-01", "2022-07-01"),
    ("2022-07-01", "2023-01-01"),
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
]

BASE_FEATURES = ["generation_mwh", "hour_sin", "hour_cos", "month_sin", "month_cos"]
DEMAND_FEATURES = BASE_FEATURES + ["demand_mw"]


def _top5_capture(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """상위 5%로 예측된 시간 중 실제 양성(출력제어 발생)을 얼마나 포착했는지."""
    n = len(y_true)
    if n == 0 or y_true.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(n * 0.05)))
    top_idx = np.argsort(-y_score)[:k]
    return float(y_true[top_idx].sum()) / float(y_true.sum())


def build_dataset() -> pd.DataFrame:
    gen = load_generation_actual("wind")
    demand = load_demand_actual()
    curt = load_curtailment("wind")
    df = gen.merge(demand, on="dt", how="inner").merge(
        curt[["dt", "is_curtailed"]], on="dt", how="inner"
    )
    df = add_time_features(df)
    df["is_curtailed"] = df["is_curtailed"].astype(int)
    return df.dropna(subset=BASE_FEATURES + ["demand_mw"]).sort_values("dt").reset_index(drop=True)


def run_block_cv(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start, end in BLOCKS:
        test_mask = (df["dt"] >= start) & (df["dt"] < end)
        test = df[test_mask]
        train = df[~test_mask]
        if len(test) == 0 or test["is_curtailed"].sum() == 0:
            print(f"[block {start}~{end}] 표본/양성 없음 — 건너뜀 (n={len(test)})")
            continue

        result = {"block": f"{start}~{end}", "n_test": len(test), "n_pos": int(test["is_curtailed"].sum())}
        for label, features in (("no_demand", BASE_FEATURES), ("with_demand", DEMAND_FEATURES)):
            model = RandomForestClassifier(
                n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
            )
            model.fit(train[features], train["is_curtailed"])
            proba = model.predict_proba(test[features])[:, 1]
            top5 = _top5_capture(test["is_curtailed"].to_numpy(), proba)
            auc = None
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(test["is_curtailed"], proba)
            except ValueError:
                pass
            result[f"top5_{label}"] = round(top5, 4)
            result[f"auc_{label}"] = round(float(auc), 4) if auc is not None else None
        rows.append(result)
        print(f"[block {result['block']}] n={result['n_test']} pos={result['n_pos']} "
              f"top5(no_demand)={result['top5_no_demand']} top5(with_demand)={result['top5_with_demand']}")
    return pd.DataFrame(rows)


def weighted_summary(results: pd.DataFrame) -> dict:
    w = results["n_pos"].to_numpy()
    summary = {}
    for label in ("no_demand", "with_demand"):
        vals = results[f"top5_{label}"].to_numpy()
        summary[f"top5_{label}_weighted"] = round(float(np.average(vals, weights=w)), 4)
        auc_vals = results[f"auc_{label}"].to_numpy()
        summary[f"auc_{label}_weighted"] = round(float(np.average(auc_vals, weights=w)), 4)
    return summary


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"전체 데이터셋: {len(dataset)}행, 양성(출력제어)={int(dataset['is_curtailed'].sum())}건")
    results = run_block_cv(dataset)
    print("\n=== 블록별 결과 ===")
    print(results.to_string(index=False))
    summary = weighted_summary(results)
    print("\n=== 가중평균 (블록별 양성 건수 가중) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(
        f"\ntop5 개선: {summary['top5_no_demand_weighted']} -> {summary['top5_with_demand_weighted']} "
        f"({'개선' if summary['top5_with_demand_weighted'] > summary['top5_no_demand_weighted'] else '악화/동일'})"
    )
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "demand_blockcv_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")
