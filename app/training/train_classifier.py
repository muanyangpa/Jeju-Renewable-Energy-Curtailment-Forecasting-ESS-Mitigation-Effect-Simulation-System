"""
출력제어 확률 예측 분류모델(RandomForestClassifier) 학습 + isotonic 확률 보정.

계획서 04장 01번 기능과 동일한 기저 스펙:
RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=6)
입력: 발전량 · 시_sin/cos · 월_sin/cos (+ 풍력은 수요 실측을 추가 피처로 정식 채택, 05장 근거)
태양광은 수요 피처를 정식 채택하지 않는다 — 검증 가능한 표본 부족(05장).

[확률 보정 — 2026-09-23 추가]
RandomForest의 predict_proba는 트리 투표 비율이라 확률로 해석하면 편향돼 있다(class_weight
='balanced'까지 쓰면 더 부풀려진다). 제어량 기댓값(확률 x 조건부 제어량)과 ESS 판단에 확률을
'값'으로 쓰기 때문에 순위(AUC)뿐 아니라 크기가 맞아야 한다. 그래서 세 모델 모두 isotonic으로
보정하고, /predict는 보정 모델만 로드한다.

  학습 구간   : 각 발전원 라벨 시작 ~ 2022-07-01   (기저 RF 학습)
  보정 구간   : 2022-07-01 ~ 2023-01-01            (isotonic 매핑 학습 — 기저 학습에 미사용)
  테스트 구간 : 2023-01-01 ~ 2024-01-01            (양쪽 모두에 미사용)

보정 구간은 기저 모델 학습에도 2023 테스트에도 쓰지 않는다. 대신 학습 구간이 6개월 줄어들어
기저 성능이 약간 내려간다 — 그 대가로 확률의 크기가 의미를 갖는다. 보정 전/후 지표를 모두
저장해(classifier_metrics.csv의 calibrated 컬럼) README 성능표에서 나란히 비교한다.

⚠ 보정으로 확률의 '크기'가 크게 바뀐다. 0.5 같은 고정 임계값을 그대로 쓰면 안 된다
  (README '운영 임계값' 참고).

실행: python -m app.training.train_classifier
"""
from __future__ import annotations

import os

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator

from app.data_prep import CURTAILMENT_COVERAGE, add_time_features, build_labeled_hourly, load_demand_actual
from app.metrics import classification_report, saturation_warning
from app.model_io import MODELS_DIR, save_artifact

BASE_FEATURES = ["generation_mwh", "hour_sin", "hour_cos", "month_sin", "month_cos"]

TRAIN_END = "2022-07-01"
CALIB_START, CALIB_END = "2022-07-01", "2023-01-01"
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


def calibrate(base: RandomForestClassifier, calib: pd.DataFrame, features: list[str]) -> CalibratedClassifierCV:
    """기저 모델은 그대로 두고(FrozenEstimator) 보정 구간으로 isotonic 매핑만 학습."""
    model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    model.fit(calib[features], calib["is_curtailed"])
    return model


def train_one(energy_type: str, use_demand: bool) -> list[dict]:
    df, features = build_dataset(energy_type, use_demand)

    train = df[df["dt"] < TRAIN_END]
    calib = df[(df["dt"] >= CALIB_START) & (df["dt"] < CALIB_END)]
    test = df[(df["dt"] >= TEST_START) & (df["dt"] < TEST_END)]

    base = make_model()
    base.fit(train[features], train["is_curtailed"])
    calibrated = calibrate(base, calib, features)

    suffix = "_demand" if use_demand else ""
    name = f"classifier_{energy_type}{suffix}"
    y_test = test["is_curtailed"].to_numpy()

    rows = []
    for is_cal, model, artifact_name in ((False, base, name), (True, calibrated, f"{name}_calibrated")):
        report = classification_report(y_test, model.predict_proba(test[features])[:, 1])
        save_artifact(
            artifact_name, model, features,
            calibrated=is_cal,
            calibration_method="isotonic (FrozenEstimator, 보정구간 전용)" if is_cal else None,
            calibration_period=[CALIB_START, CALIB_END] if is_cal else None,
            n_calib=len(calib) if is_cal else None,
            n_pos_calib=int(calib["is_curtailed"].sum()) if is_cal else None,
            train_period=[str(train["dt"].min()), str(train["dt"].max())],
            test_period=[TEST_START, TEST_END],
            label_coverage=list(CURTAILMENT_COVERAGE[energy_type]),
            input_generation="actual",  # 서비스 경로 성능은 evaluate_pipeline 참고
            test_report=report,
            served_by_predict=is_cal,
        )
        rows.append({"energy_type": energy_type, "use_demand": use_demand, "calibrated": is_cal,
                     "artifact": artifact_name, "n_train": len(train),
                     "n_pos_train": int(train["is_curtailed"].sum()),
                     "n_calib": len(calib), "n_pos_calib": int(calib["is_curtailed"].sum()),
                     **report})

    un, cal = rows
    print(f"[{name}] 학습 n={un['n_train']}(양성 {un['n_pos_train']}) "
          f"보정 n={cal['n_calib']}(양성 {cal['n_pos_calib']}) 테스트 n={un['n']}(양성 {un['n_pos']})")
    print(f"  보정 전: AUC={un['auc']} PR-AUC={un['pr_auc']} Brier={un['brier']} "
          f"top5={un['top5_capture']}(최대 {un['top5_ceiling']}) prec@5%={un['top5_precision']}")
    print(f"  보정 후: AUC={cal['auc']} PR-AUC={cal['pr_auc']} Brier={cal['brier']} "
          f"top5={cal['top5_capture']}(최대 {cal['top5_ceiling']}) prec@5%={cal['top5_precision']}"
          f"   <- /predict가 사용 ({name}_calibrated)")
    if cal["n_pos_calib"] < 50:
        print(f"  ⚠ 보정 구간 양성이 {cal['n_pos_calib']}건뿐 — isotonic 매핑이 불안정할 수 있다")
    warn = saturation_warning(cal)
    if warn:
        print("  ⚠ " + warn)
    return rows


if __name__ == "__main__":
    results = (train_one("solar", use_demand=False)
               + train_one("wind", use_demand=False)
               + train_one("wind", use_demand=True))  # 정식 채택 모델 (05장)
    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "classifier_metrics.csv"), index=False)
