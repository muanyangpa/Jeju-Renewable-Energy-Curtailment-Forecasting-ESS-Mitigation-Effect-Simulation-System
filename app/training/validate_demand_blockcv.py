"""
J장 방식 leave-block-out 6블록 교차검증 재현 — 풍력 수요피처 "정식 채택" 재확인.

블록 경계(J장과 동일): 21.01-07, 21.07-22.01, 22.01-07, 22.07-23.01, 23.01-07, 23.07-24.01
각 블록을 테스트로, 나머지 5블록을 학습으로 사용 (leave-block-out).
집계는 블록별 양성 시간 수로 가중평균한다.

[수정 사항]
- 이전 결과(모든 블록에서 '같거나 개선')는 데이터가 출력제어 발생일만으로 구성돼 top5 포착률이
  블록마다 이론상 최대값에 막혀 있던 상태에서 나온 것이라 판단 근거가 되지 못했다.
  이제 전체 달력(build_labeled_hourly)으로 재구성하고, 상한/PR-AUC/정밀도를 함께 보고한다.
- 채택 판단은 상한에 막히지 않는 pr_auc 기준으로 요약한다.

실행: python -m app.training.validate_demand_blockcv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from app.data_prep import add_normalized_features, load_generation_actual, add_time_features, build_labeled_hourly, load_demand_actual
from app.metrics import classification_report, saturation_warning
from app.model_io import MODELS_DIR
from app.training.train_classifier import BASE_FEATURES, make_model

BLOCKS = [
    ("2021-01-01", "2021-07-01"),
    ("2021-07-01", "2022-01-01"),
    ("2022-01-01", "2022-07-01"),
    ("2022-07-01", "2023-01-01"),
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
]
# 프로덕션 분류기와 같은 구성으로 비교한다 — 수요는 자유 피처(demand_mw)로도,
# 침투율(penetration = 발전량/수요)의 분모로도 들어간다.
DEMAND_FEATURES = BASE_FEATURES + ["penetration", "demand_mw"]
METRIC_KEYS = ["auc", "pr_auc", "brier", "top5_capture", "top5_precision"]


def build_dataset() -> pd.DataFrame:
    df = add_time_features(build_labeled_hourly("wind"))
    df = add_normalized_features(df, load_generation_actual("wind"), load_demand_actual())
    return df.dropna(subset=DEMAND_FEATURES).sort_values("dt").reset_index(drop=True)


def run_block_cv(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start, end in BLOCKS:
        test_mask = (df["dt"] >= start) & (df["dt"] < end)
        test, train = df[test_mask], df[~test_mask]
        if len(test) == 0 or test["is_curtailed"].sum() == 0:
            print(f"[block {start}~{end}] 표본/양성 없음 — 건너뜀 (n={len(test)})")
            continue

        row = {"block": f"{start}~{end}"}
        for label, features in (("no_demand", BASE_FEATURES), ("with_demand", DEMAND_FEATURES)):
            model = make_model()
            model.fit(train[features], train["is_curtailed"])
            rep = classification_report(test["is_curtailed"].to_numpy(), model.predict_proba(test[features])[:, 1])
            row.update({"n_test": rep["n"], "n_pos": rep["n_pos"], "top5_ceiling": rep["top5_ceiling"]})
            for k in METRIC_KEYS:
                row[f"{k}_{label}"] = rep[k]
            warn = saturation_warning(rep)
            if warn:
                print(f"  ⚠ [{row['block']} {label}] {warn}")
        rows.append(row)
        print(f"[block {row['block']}] n={row['n_test']} pos={row['n_pos']} "
              f"PR-AUC {row['pr_auc_no_demand']} -> {row['pr_auc_with_demand']} | "
              f"top5 {row['top5_capture_no_demand']} -> {row['top5_capture_with_demand']} (최대 {row['top5_ceiling']})")
    return pd.DataFrame(rows)


def weighted_summary(results: pd.DataFrame) -> dict:
    w = results["n_pos"].to_numpy()
    summary = {}
    for k in METRIC_KEYS:
        for label in ("no_demand", "with_demand"):
            vals = results[f"{k}_{label}"].to_numpy(dtype=float)
            ok = ~np.isnan(vals)
            summary[f"{k}_{label}_weighted"] = round(float(np.average(vals[ok], weights=w[ok])), 4) if ok.any() else None
    better = (results["pr_auc_with_demand"] > results["pr_auc_no_demand"]).sum()
    worse = (results["pr_auc_with_demand"] < results["pr_auc_no_demand"]).sum()
    summary["pr_auc_blocks_improved"] = int(better)
    summary["pr_auc_blocks_worsened"] = int(worse)
    return summary


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"전체 데이터셋: {len(dataset)}행, 양성(출력제어)={int(dataset['is_curtailed'].sum())}건 "
          f"(양성비율 {dataset['is_curtailed'].mean():.3%})")
    results = run_block_cv(dataset)
    print("\n=== 블록별 결과 ===")
    print(results.to_string(index=False))
    summary = weighted_summary(results)
    print("\n=== 가중평균 (블록별 양성 건수 가중) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    a, b = summary["pr_auc_no_demand_weighted"], summary["pr_auc_with_demand_weighted"]
    print(f"\nPR-AUC: {a} -> {b} ({'개선' if b > a else '악화/동일'}), "
          f"개선 블록 {summary['pr_auc_blocks_improved']}개 / 악화 블록 {summary['pr_auc_blocks_worsened']}개")
    results.to_csv(os.path.join(MODELS_DIR, "demand_blockcv_results.csv"), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(MODELS_DIR, "demand_blockcv_summary.csv"), index=False)
    print(f"\n저장: {MODELS_DIR}/demand_blockcv_results.csv, demand_blockcv_summary.csv")
