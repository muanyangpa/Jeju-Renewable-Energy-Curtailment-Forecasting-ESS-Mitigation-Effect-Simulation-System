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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.data_prep import (CURTAILMENT_COVERAGE, OTHER_SOURCE, add_cross_source_penetration,
                           add_normalized_features, add_time_features, build_labeled_hourly,
                           latest_capacity_proxy, load_asos, load_asos_multi, load_demand_actual,
                           load_generation_actual)
from app.metrics import (classification_report, day_block_ci, paired_day_block_ci,
                         saturation_warning)
from app.serving_config import SERVED_FAMILY, artifact_name, features_for, path_key
from app.training.train_converter import GEN_SOURCE
from app.model_io import (MODELS_DIR, converter_predict_mwh, feature_ranges, load_artifact,
                          save_artifact)

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


def make_model() -> RandomForestClassifier:
    """RandomForest. 기존 이름을 유지한다(임계값 선정 스크립트 등이 참조)."""
    return RandomForestClassifier(class_weight="balanced", n_estimators=200, max_depth=6,
                                  random_state=42, n_jobs=-1)


def make_lr():
    """로지스틱 회귀 — 단조 파라메트릭 대조군.

    [왜 이 문제에 선형이 유리한가]
    순열 중요도에서 침투율 계열 피처 하나가 지배한다(풍력 total_penetration +0.4853,
    태양광 penetration +0.5538). 즉 '침투율이 높으면 제어가 난다'는 단조 관계가 본질이고,
    나머지는 계절성이다. RandomForest의 분할 구조는 여기서 이득이 없는 반면, 태양광은
    학습 양성이 101건뿐이라 용량 과잉으로 분산이 커진다. 선형 모델은 외삽도 되므로
    설비 증설 드리프트에도 덜 취약하다.

    StandardScaler는 필수다 — demand_mw(수백)와 penetration(0~0.4)의 스케일이 달라
    정규화 없이는 L2 penalty가 한쪽만 누른다.
    """
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000,
                                            random_state=42))


FAMILIES = {"rf": make_model, "lr": make_lr}


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
    """한 경로를 두 모델군(RandomForest·로지스틱 회귀) x 세 보정(없음·isotonic·sigmoid)으로 학습.

    모델군을 둘 다 학습해 저장하고, 어느 쪽이 나은지 일 블록 짝지은 부트스트랩으로 비교한다.
    실제 서빙은 serving_config.SERVED_FAMILY가 정한 쪽이다 — 비교 결과를 보고 사람이 정한다.
    """
    df, features_all = build_dataset(energy_type, use_demand, cross_source)
    use_cross = cross_source == "converter"
    key = path_key(energy_type, use_demand, use_cross)
    # 경로별 피처 제외 (순열 중요도가 음수인 것). '_cross'(실측 학습) 변형은 기록용이라 제외하지 않는다.
    features = features_for(key, features_all) if cross_source != "actual" else features_all

    train = df[df["dt"] < TRAIN_END]
    calib = df[(df["dt"] >= CALIB_START) & (df["dt"] < CALIB_END)]
    test = df[(df["dt"] >= TEST_START) & (df["dt"] < TEST_END)]
    y_test = test["is_curtailed"].to_numpy()
    days_test = test["dt"].dt.date.to_numpy()

    proxy_now = latest_capacity_proxy(
        load_generation_actual(energy_type, source=GEN_SOURCE[energy_type]))

    suffix = ("_demand" if use_demand else "") + {None: "", "actual": "_cross",
                                                  "converter": "_crossp"}[cross_source]
    stem = f"classifier_{energy_type}{suffix}"
    served_fam = SERVED_FAMILY.get(key, "rf")

    rows, served_proba = [], {}
    for fam, factory in FAMILIES.items():
        base = factory()
        base.fit(train[features], train["is_curtailed"])
        fam_tag = "" if fam == "rf" else f"_{fam}"
        variants = [(None, base, f"{stem}{fam_tag}")]
        for method in ("isotonic", "sigmoid"):
            variants.append((method, calibrate(base, calib, features, method),
                             f"{stem}{fam_tag}_calibrated_{method}"))

        for method, model, art in variants:
            is_cal = method is not None
            proba = model.predict_proba(test[features])[:, 1]
            report = classification_report(y_test, proba)
            if method == SERVED_METHOD:
                served_proba[fam] = proba
                # 신뢰구간은 서빙 방식에만 계산한다(부트스트랩 2000회 x 6개는 과하다).
                ci = day_block_ci(y_test, proba, days_test)
            else:
                ci = None
            save_artifact(
                art, model, features,
                model_family=fam,
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
                capacity_proxy_mwh=proxy_now,
                input_generation="actual",  # 서비스 경로 성능은 evaluate_pipeline 참고
                cross_source=cross_source,
                dropped_features=list(set(features_all) - set(features)) or None,
                # 서빙에서 드리프트를 감지하기 위한 학습 분포 범위(min~max).
                train_feature_ranges=feature_ranges(train, features),
                test_report=report,
                # 일 블록 부트스트랩 95% 신뢰구간. 시간 단위로 재면 1.6~2.1배 좁게 나온다 —
                # 출력제어가 날짜 단위로 뭉쳐 있기 때문이다(app/metrics.day_block_ci).
                test_report_ci=ci,
                served_by_predict=(method == SERVED_METHOD and fam == served_fam),
            )
            rows.append({"energy_type": energy_type, "use_demand": use_demand, "path": key,
                         "family": fam, "calibrated": is_cal, "method": method or "none",
                         "artifact": art, "n_features": len(features),
                         "n_train": len(train), "n_pos_train": int(train["is_curtailed"].sum()),
                         "n_calib": len(calib), "n_pos_calib": int(calib["is_curtailed"].sum()),
                         "n_pos_days_test": int(len(set(days_test[y_test == 1]))),
                         **report,
                         **({f"ci_{k}": f"[{v['lo']}, {v['hi']}]" for k, v in ci.items()}
                            if ci else {})})

    un = next(r for r in rows if r["family"] == "rf" and r["method"] == "none")
    print(f"[{stem}] 피처 {len(features)}개 학습 n={un['n_train']}(양성 {un['n_pos_train']}) "
          f"보정 n={un['n_calib']}(양성 {un['n_pos_calib']}) "
          f"테스트 n={un['n']}(양성 {un['n_pos']}시간 / {un['n_pos_days_test']}일)")
    for r in rows:
        if r["method"] != SERVED_METHOD:
            continue
        served = "   <- /predict가 사용" if r["family"] == served_fam else ""
        print(f"  {r['family']:3} sigmoid: AUC={r['auc']} {r.get('ci_auc','')}  "
              f"PR-AUC={r['pr_auc']} {r.get('ci_pr_auc','')}  top5={r['top5_capture']} "
              f"{r.get('ci_top5_capture','')}  Brier={r['brier']}{served}")

    # 모델군 짝지은 비교 — 같은 날짜 리샘플을 공유한다.
    cmp_rows = []
    for metric in ("auc", "pr_auc", "top5_capture"):
        d = paired_day_block_ci(y_test, served_proba["rf"], served_proba["lr"], days_test, metric)
        cmp_rows.append({"path": key, "n_features": len(features), **d})
        verdict = {"A": "RF 우세", "B": "LR 우세", "tie": "차이 없음"}[d["winner"]]
        print(f"    Δ{metric:12} (RF − LR) {d['diff_median']:+.4f} "
              f"[{d['lo']:+.4f}, {d['hi']:+.4f}]  {verdict}")

    if un["n_pos_calib"] < 50:
        print(f"  ⚠ 보정 구간 양성이 {un['n_pos_calib']}건뿐 — 특히 isotonic이 불안정할 수 있다")
    warn = saturation_warning(next(r for r in rows if r["method"] == SERVED_METHOD
                                   and r["family"] == served_fam))
    if warn:
        print("  ⚠ " + warn)
    return rows, cmp_rows


if __name__ == "__main__":
    specs = (("solar", False, None), ("solar", True, None),
             ("wind", False, None), ("wind", True, None),
             # [2026-09-26] 계통 전체 침투율 — 출력제어는 재생E 합계 기준으로 결정되므로
             # 타 발전원을 봐야 한다(data_prep.add_cross_source_penetration).
             ("wind", True, "converter"),
             # 실측 태양광으로 학습한 변형. 서빙 입력(컨버터 예측)과 어긋나 확률 크기가
             # 눌리므로 배포하지 않고 기록용으로만 남긴다.
             ("wind", True, "actual"),
             ("solar", True, "actual"))
    results, comparisons = [], []
    for spec in specs:
        r, c = train_one(*spec)
        results += r
        comparisons += c
    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "classifier_metrics.csv"), index=False)
    pd.DataFrame(comparisons).to_csv(
        os.path.join(MODELS_DIR, "model_family_comparison.csv"), index=False)
    print(f"\n저장: {MODELS_DIR}/classifier_metrics.csv, model_family_comparison.csv")
    print("서빙 모델군은 app/serving_config.py의 SERVED_FAMILY가 정한다 — "
          "위 비교 결과를 보고 갱신할 것.")
