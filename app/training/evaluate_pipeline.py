"""
서비스 경로(날씨 -> 컨버터 -> 분류기) 그대로 2023년을 평가하는 스크립트. [신규]

왜 필요한가:
  분류기는 '실측 발전량'으로 학습·평가됐지만, 실제 /predict는 '컨버터가 예측한 발전량'을 넣는다.
  따라서 classifier_metrics.csv의 수치는 서비스 성능이 아니라 상한선이다.
  이 스크립트는 같은 분류기에 (A) 실측 발전량, (B) 컨버터 예측 발전량을 각각 넣어 비교한다 —
  계획서 05장의 '실측 입력 vs 예보/재구성 입력' 비교를 프로덕션 코드로 재현하는 것.

주의(해석 범위):
  - 날씨 입력은 ASOS '실측' 관측치다. 실제 서비스는 '예보'를 넣으므로 (B)도 여전히 낙관적인 추정이다.
  - 풍력 수요 모델의 수요 입력도 실측 수요다(하루전 수요예측 아카이브는 이 저장소에 없음).
  - 컨버터·분류기 모두 2023년을 학습에 쓰지 않았으므로 리키지는 없다.

선행: train_converter, train_classifier 실행
실행: python -m app.training.evaluate_pipeline
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from app.data_prep import (OTHER_SOURCE, add_cross_source_penetration,
                           add_normalized_features, add_time_features, build_labeled_hourly,
                           load_asos, load_asos_multi, load_demand_actual, load_generation_actual)
from app.metrics import classification_report, saturation_warning
from app.model_io import MODELS_DIR, converter_predict_mwh, load_artifact

TEST_START, TEST_END = "2023-01-01", "2024-01-01"


def _weather(energy_type: str) -> pd.DataFrame:
    w = load_asos_multi(("184", "185", "188")) if energy_type == "wind" else load_asos("184")
    return add_time_features(w)


def evaluate(energy_type: str, use_demand: bool, cross: str | None = None) -> list[dict]:
    """cross: None | "actual"(_cross 아티팩트) | "converter"(_crossp 아티팩트)."""
    use_cross = cross is not None
    converter = load_artifact(f"converter_{energy_type}")
    # /predict가 서빙하는 것은 보정 모델이므로 서비스 경로 평가도 보정 모델로 한다.
    clf_name = (f"classifier_{energy_type}" + ("_demand" if use_demand else "")
                + {None: "", "actual": "_cross", "converter": "_crossp"}[cross]
                + "_calibrated_sigmoid")
    classifier = load_artifact(clf_name)

    labeled = build_labeled_hourly(energy_type)
    labeled = add_normalized_features(labeled, load_generation_actual(energy_type), load_demand_actual())
    if use_cross:
        labeled = add_cross_source_penetration(labeled, energy_type)
    labeled = labeled[(labeled["dt"] >= TEST_START) & (labeled["dt"] < TEST_END)]
    weather = _weather(energy_type)
    df = labeled.merge(weather, on="dt", how="inner")
    df = df.dropna(subset=converter["features"] + ["generation_mwh"]).reset_index(drop=True)

    df["generation_pred"] = converter_predict_mwh(converter, df)
    if use_cross:
        # 서비스 경로는 타 발전원도 컨버터 예측을 쓴다 — 실측 태양광을 넣으면 낙관적으로 오염된다.
        other_conv = load_artifact(f"converter_{OTHER_SOURCE[energy_type]}")
        missing = [c for c in other_conv["features"] if c not in df.columns]
        if missing:
            raise RuntimeError(f"{clf_name} 서비스 경로 평가에 필요한 기상 컬럼이 없습니다: {missing}")
        df = df.dropna(subset=other_conv["features"]).reset_index(drop=True)
        df["other_generation_pred"] = converter_predict_mwh(other_conv, df)

    rows = []
    for input_name, gen_col in (("actual_generation", "generation_mwh"), ("converter_generation", "generation_pred")):
        # 파생 피처(capacity_factor·penetration)도 해당 발전량 기준으로 다시 계산해야 한다
        x = df.assign(generation_mwh=df[gen_col])
        x["capacity_factor"] = x["generation_mwh"] / x["capacity_proxy_mwh"]
        if "demand_mw" in x.columns:
            x["penetration"] = x["generation_mwh"] / x["demand_mw"]
            if use_cross:
                other_col = ("other_generation_pred" if gen_col == "generation_pred"
                             else "other_generation_mwh")
                x["total_penetration"] = (x["generation_mwh"] + x[other_col]) / x["demand_mw"]
        proba = classifier["model"].predict_proba(x[classifier["features"]])[:, 1]
        rep = classification_report(df["is_curtailed"].to_numpy(), proba)
        # Sum(p)를 함께 남긴다. expected_curtailment_mwh 총합이 이 값에 비례하므로,
        # 순위 지표가 좋아도 Sum(p)가 실제 제어 시간 수에서 멀어지면 총합 추정이 무너진다.
        rows.append({"model": clf_name, "input": input_name,
                     "sum_proba": round(float(proba.sum()), 1), **rep})
        warn = saturation_warning(rep)
        if warn:
            print(f"  ⚠ [{clf_name} / {input_name}] {warn}")

    a, b = rows
    for k in ("pr_auc", "top5_capture"):
        if a[k] and isinstance(a[k], float):
            b[f"{k}_rel_change_pct"] = round((b[k] - a[k]) / a[k] * 100, 1)
    print(f"[{clf_name}] n={a['n']} pos={a['n_pos']} | "
          f"PR-AUC 실측입력 {a['pr_auc']} -> 컨버터입력 {b['pr_auc']} ({b.get('pr_auc_rel_change_pct')}%) | "
          f"top5 {a['top5_capture']} -> {b['top5_capture']} ({b.get('top5_capture_rel_change_pct')}%) | "
          f"AUC {a['auc']} -> {b['auc']}")
    return rows


if __name__ == "__main__":
    results = (evaluate("solar", False) + evaluate("solar", True)
               + evaluate("wind", False) + evaluate("wind", True)
               # 계통 전체 침투율 모델 — 풍력 서빙 경로에서 태양광 기상값이 오면 이 모델이 쓰인다.
               # 여기서는 태양광 발전량도 컨버터 예측값이므로 오차가 두 번 얹힌다.
               + evaluate("wind", True, cross="actual")
               + evaluate("wind", True, cross="converter"))
    out = os.path.join(MODELS_DIR, "pipeline_eval_metrics.csv")
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n저장: {out}")
