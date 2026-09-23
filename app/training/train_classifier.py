"""
출력제어 확률 예측 분류모델(RandomForestClassifier) 학습.

계획서 04장 01번 기능과 동일한 스펙:
RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=6)
입력: 발전량 · 시_sin/cos · 월_sin/cos (+ 풍력은 수요 실측을 추가 피처로 정식 채택, 05장 근거)
태양광은 수요 피처를 정식 채택하지 않는다 — 검증 가능한 표본 부족(05장).

[수정 사항]
- 학습/평가 데이터를 '출력제어 발생일만'이 아니라 전체 달력(build_labeled_hourly)으로 구성
- 지표에 top5_ceiling / top5_precision / pr_auc / brier 추가, 상한 도달 시 경고
- 모델 아티팩트에 메타데이터(학습 구간 등) 저장

실행: python -m app.training.train_classifier
"""
from __future__ import annotations

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.data_prep import CURTAILMENT_COVERAGE, add_time_features, build_labeled_hourly, load_demand_actual
from app.metrics import classification_report, saturation_warning
from app.model_io import MODELS_DIR, save_artifact

BASE_FEATURES = ["generation_mwh", "hour_sin", "hour_cos", "month_sin", "month_cos"]
TEST_START, TEST_END = "2023-01-01", "2024-01-01"


def build_dataset(energy_type: str, use_demand: bool) -> tuple[pd.DataFrame, list[str]]:
    df = add_time_features(build_labeled_hourly(energy_type))
    features = list(BASE_FEATURES)
    if use_demand:
        df = df.merge(load_demand_actual(), on="dt", how="inner")
        features.append("demand_mw")
    df = df.dropna(subset=features).reset_index(drop=True)
    return df, features


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(class_weight="balanced", n_estimators=200, max_depth=6,
                                  random_state=42, n_jobs=-1)


def train_one(energy_type: str, use_demand: bool) -> dict:
    df, features = build_dataset(energy_type, use_demand)

    # 리키지-프리 시간분할: 2023년을 테스트로 사용 (07장 ESS 계산과 같은 기준 연도)
    train = df[df["dt"] < TEST_START]
    test = df[(df["dt"] >= TEST_START) & (df["dt"] < TEST_END)]

    model = make_model()
    model.fit(train[features], train["is_curtailed"])
    proba = model.predict_proba(test[features])[:, 1]
    report = classification_report(test["is_curtailed"].to_numpy(), proba)

    suffix = "_demand" if use_demand else ""
    name = f"classifier_{energy_type}{suffix}"
    out_path = save_artifact(
        name, model, features,
        train_period=[str(train["dt"].min()), str(train["dt"].max())],
        test_period=[TEST_START, TEST_END],
        label_coverage=list(CURTAILMENT_COVERAGE[energy_type]),
        input_generation="actual",  # 학습 입력은 실측 발전량 — 서비스 경로 성능은 evaluate_pipeline 참고
        test_report=report,
    )

    metrics = {"energy_type": energy_type, "use_demand": use_demand,
               "n_train": len(train), "n_pos_train": int(train["is_curtailed"].sum()), **report}
    print(f"[{name}] AUC={report['auc']} PR-AUC={report['pr_auc']} "
          f"top5={report['top5_capture']}(최대 {report['top5_ceiling']}) prec@5%={report['top5_precision']} "
          f"(test n={report['n']}, pos={report['n_pos']}, 양성비율={report['pos_rate']}) -> {out_path}")
    warn = saturation_warning(report)
    if warn:
        print("  ⚠ " + warn)
    return metrics


if __name__ == "__main__":
    results = [
        train_one("solar", use_demand=False),
        train_one("wind", use_demand=False),
        train_one("wind", use_demand=True),  # 정식 채택 모델 (05장)
    ]
    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "classifier_metrics.csv"), index=False)
