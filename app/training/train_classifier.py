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

from app.data_prep import (CURTAILMENT_COVERAGE, OTHER_SOURCE, add_cross_source_penetration,
                           add_normalized_features, add_time_features, build_labeled_hourly,
                           latest_capacity_proxy, load_asos, load_asos_multi, load_demand_actual,
                           load_generation_actual)
from app.metrics import classification_report, saturation_warning
from app.training.train_converter import GEN_SOURCE
from app.model_io import MODELS_DIR, converter_predict_mwh, load_artifact, save_artifact

# [2026-09-25] 발전량 절대값(MWh) -> 정규화 피처로 교체.
# 제주 태양광 설비 증설로 MWh 분포가 해마다 위로 밀려, 학습 범위를 벗어난 입력에서
# RandomForest가 경계에 포화됐다(2023 테스트의 3.76%가 학습 범위 밖).
#   capacity_factor = 발전량 / 설비용량대리  -> 날씨 강도, 범위 안정
#   penetration     = 발전량 / 수요          -> 잉여 압력, 제어의 실제 구동 요인
# 실험 결과(models/normalization_experiment.csv): 태양광 PR-AUC 0.406 -> 0.703,
# top5 0.493 -> 0.826. 절대 MWh를 함께 넣으면 오히려 나빠져(0.629) 완전히 뺐다.
TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
BASE_FEATURES = ["capacity_factor"] + TIME_FEATURES

# /predict가 서빙하는 보정 방식. sigmoid는 단조 변환이라 순위 지표(AUC·PR-AUC·top5)를
# 보정 전과 동일하게 보존하면서 Brier를 개선한다. isotonic은 2단계 총합 오차만 더 낫다
# (-14.1% vs -19.6%) — 경보·순위 품질을 우선해 sigmoid를 채택했다(README 비교표 참고).
SERVED_METHOD = "sigmoid"

TRAIN_END = "2022-07-01"
CALIB_START, CALIB_END = "2022-07-01", "2023-01-01"
TEST_START, TEST_END = "2023-01-01", "2024-01-01"


def _other_gen_predicted(energy_type: str) -> pd.DataFrame:
    """타 발전원의 '컨버터 예측' 발전량 — 서빙 경로와 같은 입력으로 학습하기 위한 것.

    /predict는 타 발전원 발전량을 실측으로 알 수 없고 같은 요청의 기상값을 그 발전원의
    컨버터에 넣어 얻는다. 학습을 실측으로 하면 total_penetration의 수준이 서빙과 어긋난다.
    """
    other = OTHER_SOURCE[energy_type]
    conv = load_artifact(f"converter_{other}")
    w = add_time_features(load_asos("184") if other == "solar"
                          else load_asos_multi(("184", "185", "188")))
    w = w.dropna(subset=conv["features"]).copy()
    w["other_generation_mwh"] = converter_predict_mwh(conv, w)
    return w[["dt", "other_generation_mwh"]]


def build_dataset(energy_type: str, use_demand: bool,
                  cross_source: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """cross_source: None(미사용) | "actual"(타 발전원 실측) | "converter"(타 발전원 컨버터 예측)."""
    gen = load_generation_actual(energy_type)
    df = add_time_features(build_labeled_hourly(energy_type))
    df = add_normalized_features(df, gen, load_demand_actual())
    features = list(BASE_FEATURES)
    if use_demand:
        features.append("penetration")
        if energy_type == "wind":
            features.append("demand_mw")  # 풍력은 수요 자체도 정식 피처(05장)
    if cross_source is not None:
        other = _other_gen_predicted(energy_type) if cross_source == "converter" else None
        df = add_cross_source_penetration(df, energy_type, other)
        features.append("total_penetration")
    df = df.dropna(subset=features).reset_index(drop=True)
    return df, features


def feature_ranges(train: pd.DataFrame, features: list[str]) -> dict[str, list[float]]:
    """학습 구간 각 피처의 [p1, p99]. 서빙 입력이 이 밖으로 나가면 외삽이라는 신호다.

    최소·최대 대신 p1·p99를 쓰는 이유: 학습 데이터의 단일 이상치가 범위를 넓혀버리면
    드리프트를 놓친다. 시각 피처(sin/cos)는 순환값이라 항상 [-1,1]이므로 의미가 없지만
    제외하지 않고 그대로 둔다 — 제외 규칙을 두면 피처가 늘 때 빠뜨린다.
    """
    return {f: [round(float(train[f].quantile(0.01)), 6),
                round(float(train[f].quantile(0.99)), 6)] for f in features}


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


def train_one(energy_type: str, use_demand: bool, cross_source: str | None = None) -> list[dict]:
    df, features = build_dataset(energy_type, use_demand, cross_source)

    train = df[df["dt"] < TRAIN_END]
    calib = df[(df["dt"] >= CALIB_START) & (df["dt"] < CALIB_END)]
    test = df[(df["dt"] >= TEST_START) & (df["dt"] < TEST_END)]

    # 서빙 대리지표는 '컨버터가 내보내는 값과 같은 출처'로 계산해야 한다.
    # capacity_factor = 컨버터출력 / 대리지표 이므로, 둘의 모집단이 다르면 비율이 왜곡된다.
    proxy_now = latest_capacity_proxy(
        load_generation_actual(energy_type, source=GEN_SOURCE[energy_type]))

    base = make_model()
    base.fit(train[features], train["is_curtailed"])

    suffix = ("_demand" if use_demand else "") + {None: "", "actual": "_cross",
                                                   "converter": "_crossp"}[cross_source]
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
            # 서빙 시 capacity_factor의 분모로 쓰는 상수. /predict는 미래 시각을 받아
            # 그 시점의 대리지표를 계산할 수 없으므로 학습 시점의 최신값을 고정해 쓴다.
            # 재학습할 때마다 갱신되며, 설비가 크게 늘면 재학습이 필요하다는 신호이기도 하다.
            capacity_proxy_mwh=proxy_now,
            input_generation="actual",  # 서비스 경로 성능은 evaluate_pipeline 참고
            cross_source=cross_source,  # 타 발전원 발전량의 출처 (None/actual/converter)
            # 서빙에서 드리프트를 감지하기 위한 학습 분포 범위. RandomForest는 외삽을 못 하므로
            # 입력이 이 범위를 벗어나면 경계에 포화되고 순위가 무너진다 — 그런데 응답만 보면
            # 알 수 없다. /predict가 이 범위와 비교해 note로 알린다.
            train_feature_ranges=feature_ranges(train, features),
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
               + train_one("solar", use_demand=True)   # 침투율용 — 수요를 분모로만 사용
               + train_one("wind", use_demand=False)
               + train_one("wind", use_demand=True)    # 구 정식 모델 (05장)
               # [2026-09-26] 계통 전체 침투율 추가 — 출력제어는 재생E 합계 기준으로
               # 결정되므로 타 발전원을 봐야 한다(data_prep.add_cross_source_penetration 참고).
               # 2023 테스트에서 PR-AUC 0.717 -> 0.805, top5 0.547 -> 0.586, Brier도 개선.
               # 서빙은 태양광 기상값이 함께 오면 이 모델로 자동 전환한다.
               + train_one("wind", use_demand=True, cross_source="actual")
               # 서빙은 타 발전원도 컨버터 예측이라 학습 입력도 그렇게 맞춘 변형을 함께 만든다.
               # 실측으로 학습하면 컨버터 입력에서 확률의 크기가 눌린다(Sum(p) 718 -> 398).
               + train_one("wind", use_demand=True, cross_source="converter")
               + train_one("solar", use_demand=True, cross_source="actual"))
    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "classifier_metrics.csv"), index=False)
