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

# /predict가 서빙하는 보정 방식. sigmoid는 단조 변환이라 순위 지표(AUC·PR-AUC·top5)를
# 보정 전과 동일하게 보존하면서 Brier를 개선한다. isotonic은 2단계 총합 오차만 더 낫다
# (-14.1% vs -19.6%) — 경보·순위 품질을 우선해 sigmoid를 채택했다(README 비교표 참고).
SERVED_METHOD = "sigmoid"

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


def calibrate(base: RandomForestClassifier, calib: pd.DataFrame, features: list[str],
              method: str = "isotonic") -> CalibratedClassifierCV:
    """기저 모델은 그대로 두고(FrozenEstimator) 보정 구간으로 매핑만 학습.

    method="isotonic": 단조 계단함수. 자유도가 높아 데이터가 적으면 과적합하기 쉽다.
    method="sigmoid" : Platt scaling. 파라미터 2개뿐이라 표본이 적을 때 안정적이지만
                       확률 왜곡이 시그모이드 형태라는 가정이 맞아야 한다.
    """
    model = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    model.fit(calib[features], calib["is_curtailed"])
    return model


def train_one(energy_type: str, use_demand: bool) -> list[dict]:
    df, features = build_dataset(energy_type, use_demand)

    train = df[df["dt"] < TRAIN_END]
    calib = df[(df["dt"] >= CALIB_START) & (df["dt"] < CALIB_END)]
    test = df[(df["dt"] >= TEST_START) & (df["dt"] < TEST_END)]

    base = make_model()
    base.fit(train[features], train["is_curtailed"])

    suffix = "_demand" if use_demand else ""
    name = f"classifier_{energy_type}{suffix}"
    y_test = test["is_curtailed"].to_numpy()

    # 아티팩트명에 보정 방식을 명시한다 — model_used만 보고 어떤 확률인지 알 수 있어야 한다.
    variants = [(None, base, name)]
    for method in ("isotonic", "sigmoid"):
        variants.append((method, calibrate(base, calib, features, method),
                         f"{name}_calibrated_{method}"))

    rows = []
    for method, model, artifact_name in variants:
        is_cal = method is not None
        report = classification_report(y_test, model.predict_proba(test[features])[:, 1])
        save_artifact(
            artifact_name, model, features,
            calibrated=is_cal,
            calibration_method=f"{method} (FrozenEstimator, 보정구간 전용)" if is_cal else None,
            calibration_period=[CALIB_START, CALIB_END] if is_cal else None,
            n_calib=len(calib) if is_cal else None,
            n_pos_calib=int(calib["is_curtailed"].sum()) if is_cal else None,
            train_period=[str(train["dt"].min()), str(train["dt"].max())],
            test_period=[TEST_START, TEST_END],
            label_coverage=list(CURTAILMENT_COVERAGE[energy_type]),
            input_generation="actual",  # 서비스 경로 성능은 evaluate_pipeline 참고
            test_report=report,
            served_by_predict=(method == SERVED_METHOD),
        )
        rows.append({"energy_type": energy_type, "use_demand": use_demand, "calibrated": is_cal,
                     "method": method or "none",
                     "artifact": artifact_name, "n_train": len(train),
                     "n_pos_train": int(train["is_curtailed"].sum()),
                     "n_calib": len(calib), "n_pos_calib": int(calib["is_curtailed"].sum()),
                     **report})

    un = rows[0]
    print(f"[{name}] 학습 n={un['n_train']}(양성 {un['n_pos_train']}) "
          f"보정 n={un['n_calib']}(양성 {un['n_pos_calib']}) 테스트 n={un['n']}(양성 {un['n_pos']})")
    for r in rows:
        tag = {"none": "보정 전  ", "isotonic": "isotonic", "sigmoid": "sigmoid "}[r["method"]]
        served = "   <- /predict가 사용" if r["method"] == SERVED_METHOD else ""
        print(f"  {tag}: AUC={r['auc']} PR-AUC={r['pr_auc']} Brier={r['brier']} "
              f"top5={r['top5_capture']}(최대 {r['top5_ceiling']}) prec@5%={r['top5_precision']}{served}")
    if un["n_pos_calib"] < 50:
        print(f"  ⚠ 보정 구간 양성이 {un['n_pos_calib']}건뿐 — 특히 isotonic이 불안정할 수 있다")
    warn = saturation_warning(next(r for r in rows if r["method"] == SERVED_METHOD))
    if warn:
        print("  ⚠ " + warn)
    return rows


if __name__ == "__main__":
    results = (train_one("solar", use_demand=False)
               + train_one("wind", use_demand=False)
               + train_one("wind", use_demand=True))  # 정식 채택 모델 (05장)
    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "classifier_metrics.csv"), index=False)
