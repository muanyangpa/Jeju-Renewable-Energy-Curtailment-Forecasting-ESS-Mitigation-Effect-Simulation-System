"""
풍력 제어량(MWh) 2단계 추정 — ①보정 분류 → ②조건부 회귀. 풍력 전용.

계획서 08장의 "제어량 회귀모델(풍력 전용)"을 2단계 구조로 재구성한다.
태양광은 이 모델을 만들 수 없다 — 전력거래소가 태양광 제어량(MWh)을 공식 산정하지 않기 때문
(계획서 07장, data.go.kr 메타데이터로 확인). 이 스크립트를 태양광에 억지로 돌리지 말 것.

[왜 2단계인가]
단일 회귀는 전체 달력(제어량 0인 시간이 약 94%)으로 학습하면 RandomForest가 예측을 0 쪽으로
강하게 끌어당겨(shrinkage) 2023 총합을 -81.7% 과소추정했다. 제어 '발생 여부'와 '발생했을 때의
크기'는 서로 다른 문제이므로 분리한다.

  ① stage1: 출력제어 발생 확률 P(curtail).
     train_classifier가 만든 classifier_wind_demand_calibrated(isotonic 보정)를 그대로 쓴다 —
     /predict가 서빙하는 확률과 같은 모델이어야 expected = probability x 조건부가 성립한다.
  ② stage2: 제어가 발생한 시간만으로 학습한 조건부 제어량 E[MWh | curtail].
     0인 시간을 아예 보지 않으므로 shrinkage가 없다.

  기댓값 = P(curtail) x E[MWh | curtail]  -> 총합 추정에 사용 (불편추정에 가깝다)
  임계값 방식 = P >= thr 이면 E[MWh | curtail], 아니면 0 -> 시간별 계열이 필요한 ESS 계산에 사용

[stage2 입력 발전량 두 가지]
서빙 경로(/predict)는 실측 발전량이 아니라 '컨버터가 예측한 발전량'을 넣는다. 학습 입력과
서빙 입력이 다르면 성능이 떨어지므로, 실측/컨버터 두 가지로 각각 학습해 2023 제어시간 MAE를
비교하고 서빙 조건과 일치하는 쪽을 배포한다.

선행: train_converter, train_classifier 실행
실행: python -m app.training.train_curtailment_regressor
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from app.data_prep import add_time_features, build_labeled_hourly, load_asos_multi, load_demand_actual
from app.metrics import classification_report
from app.model_io import MODELS_DIR, load_artifact, save_artifact
from app.training.train_classifier import build_dataset as _clf_dataset, calibrate, make_model
from app.services.ess_simulation import simulate_hourly_capped

FEATURES = ["generation_mwh", "hour_sin", "hour_cos", "month_sin", "month_cos", "demand_mw"]

TEST_START, TEST_END = "2023-01-01", "2024-01-01"
# 임계값은 2023(테스트)을 보지 않고 정해야 한다. 2022년을 임계값 선정용 검증구간으로 쓰되,
# 그 구간을 학습에 쓴 모델로 고르면 리키지이므로 2022 이전만으로 학습한 별도 모델을 쓴다.
THR_VAL_START, THR_VAL_END = "2022-01-01", "2023-01-01"

ESS_RATED_MW = 22.5  # 07장 기준 ESS 정격출력. 실측 제어량 기준 2023 흡수율 36.8%의 기준값.


# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """전체 달력 + 시간피처 + 수요 + 컨버터 예측 발전량(generation_pred)."""
    df = add_time_features(build_labeled_hourly("wind"))
    df = df.merge(load_demand_actual(), on="dt", how="inner")

    converter = load_artifact("converter_wind")
    weather = add_time_features(load_asos_multi(("184", "185", "188")))
    # 컨버터 피처는 날씨쪽 컬럼이므로 weather에서 계산한 뒤 dt로 붙인다.
    weather = weather.dropna(subset=converter["features"]).copy()
    weather["generation_pred"] = np.clip(converter["model"].predict(weather[converter["features"]]), 0, None)
    df = df.merge(weather[["dt", "generation_pred"]], on="dt", how="inner")

    return df.dropna(subset=FEATURES + ["curtailment_mwh", "generation_pred"]).sort_values("dt").reset_index(drop=True)


def _with_gen(df: pd.DataFrame, gen_col: str) -> pd.DataFrame:
    """generation_mwh 자리에 지정한 컬럼(실측 또는 컨버터 예측)을 넣은 피처 프레임."""
    out = df.copy()
    out["generation_mwh"] = df[gen_col]
    return out


# ---------------------------------------------------------------------------
# ① stage1: 보정된 발생확률 — train_classifier가 만든 classifier_wind_demand_calibrated를 그대로 쓴다
#    (/predict가 서빙하는 확률과 동일한 모델이어야 expected = probability x 조건부가 성립한다)
# ---------------------------------------------------------------------------

STAGE1_NAME = "classifier_wind_demand_calibrated"


def load_stage1() -> dict:
    return load_artifact(STAGE1_NAME)


# ---------------------------------------------------------------------------
# ② stage2: 조건부 제어량 (제어 발생 시간만)
# ---------------------------------------------------------------------------

def fit_stage2(train: pd.DataFrame, gen_col: str) -> RandomForestRegressor:
    curtailed = train[train["curtailment_mwh"] > 0]
    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    x = _with_gen(curtailed, gen_col)
    model.fit(x[FEATURES], x["curtailment_mwh"])
    return model


def _predict(model, df: pd.DataFrame, gen_col: str, proba: bool = False) -> np.ndarray:
    x = _with_gen(df, gen_col)
    if proba:
        return model.predict_proba(x[FEATURES])[:, 1]
    return np.clip(model.predict(x[FEATURES]), 0, None)


# ---------------------------------------------------------------------------
# 임계값 선정 (2022년, 2022 이전만으로 학습한 모델 사용 -> 리키지 없음)
# ---------------------------------------------------------------------------

def choose_threshold() -> dict:
    """임계값은 기저학습·보정·2023테스트 어디에도 쓰지 않은 구간에서 고른다.

    배포 모델(classifier_wind_demand_calibrated)은 2022-07-01까지 학습 + 2022 하반기로 보정했으므로,
    2022년 어디를 써도 그 모델에는 in-sample이다. 그래서 임계값 선정 전용으로 한 칸씩 앞당긴
    모델을 따로 만든다 — 기저 <2022-01-01, 보정 2022 상반기, 임계값 선정 2022 하반기.
    배포 모델과 같은 방식으로 보정된 확률이므로 임계값이 옮겨갈 수 있다.
    """
    df, features = _clf_dataset("wind", use_demand=True)
    base_tr = df[df["dt"] < "2022-01-01"]
    calib = df[(df["dt"] >= "2022-01-01") & (df["dt"] < "2022-07-01")]
    val = df[(df["dt"] >= "2022-07-01") & (df["dt"] < "2023-01-01")]

    base = make_model()
    base.fit(base_tr[features], base_tr["is_curtailed"])
    clf = calibrate(base, calib, features)
    p = clf.predict_proba(val[features])[:, 1]
    y = val["is_curtailed"].to_numpy().astype(bool)

    best = None
    for thr in np.arange(0.02, 0.96, 0.01):
        pred = p >= thr
        tp = int((pred & y).sum())
        if tp == 0:
            continue
        precision = tp / int(pred.sum())
        recall = tp / int(y.sum())
        f1 = 2 * precision * recall / (precision + recall)
        if best is None or f1 > best["f1"]:
            best = {"threshold": round(float(thr), 2), "f1": round(f1, 4),
                    "precision": round(precision, 4), "recall": round(recall, 4)}
    best.update({"selection_period": "2022-07-01~2023-01-01", "n_val": len(val),
                 "n_pos_val": int(y.sum()),
                 "selection_model": "기저<2022-01-01 + isotonic(2022-01-01~2022-07-01)"})
    return best


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> dict:
    df = build_dataset()
    train = df[df["dt"] < TEST_START]
    test = df[(df["dt"] >= TEST_START) & (df["dt"] < TEST_END)]
    y = test["curtailment_mwh"].to_numpy()
    curtailed = y > 0
    actual_total = float(y.sum())

    # --- stage1: /predict가 서빙하는 보정 분류기를 그대로 로드 ---
    stage1_art = load_stage1()
    stage1 = stage1_art["model"]

    # --- stage2 두 변형 비교: 학습 입력 x 평가 입력 ---
    stage2 = {v: fit_stage2(train, gen) for v, gen in (("actual", "generation_mwh"), ("converter", "generation_pred"))}
    mae_rows = []
    for trained_on, model in stage2.items():
        for eval_on, gen in (("actual", "generation_mwh"), ("converter", "generation_pred")):
            pred = _predict(model, test, gen)
            mae_rows.append({
                "trained_on": trained_on, "eval_on": eval_on,
                "mae_on_curtailed_hours_mwh": round(float(mean_absolute_error(y[curtailed], pred[curtailed])), 2),
                "n_curtailed_hours": int(curtailed.sum()),
            })
    mae_df = pd.DataFrame(mae_rows)

    # 서빙은 항상 컨버터 예측을 넣으므로 eval_on=converter 중 MAE가 낮은 쪽을 배포한다.
    serving = mae_df[mae_df["eval_on"] == "converter"].sort_values("mae_on_curtailed_hours_mwh").iloc[0]
    deployed = str(serving["trained_on"])

    # --- stage1 분류 성능 (서빙 확률의 출처이므로 README 성능표에 반영) ---
    y_bin = (y > 0).astype(int)
    stage1_rows = []
    for input_name, gen in (("actual_generation", "generation_mwh"), ("converter_generation", "generation_pred")):
        rep = classification_report(y_bin, _predict(stage1, test, gen, proba=True))
        stage1_rows.append({"model": STAGE1_NAME, "input": input_name, **rep})

    # --- (a) 기댓값(확률 x 조건부) 총합 오차 ---
    p_actual = _predict(stage1, test, "generation_mwh", proba=True)
    p_serving = _predict(stage1, test, "generation_pred", proba=True)
    cond = _predict(stage2[deployed], test, "generation_pred")
    cond_actual_in = _predict(stage2[deployed], test, "generation_mwh")

    expected_rows = []
    for input_name, p, c in (("actual_generation", p_actual, cond_actual_in),
                             ("converter_generation", p_serving, cond)):
        total = float((p * c).sum())
        expected_rows.append({
            "input": input_name,
            "actual_total_mwh_2023": round(actual_total, 1),
            "predicted_total_mwh_2023": round(total, 1),
            "total_error_pct": round((total - actual_total) / actual_total * 100, 1),
        })

    # --- (b) 임계값 방식 -> ESS hourly_capped 흡수율 ---
    thr_info = choose_threshold()
    thr = thr_info["threshold"]
    ess_rows = []
    baseline = simulate_hourly_capped(y.tolist(), ESS_RATED_MW)
    ess_rows.append({"series": "actual_curtailment", "threshold": None,
                     "total_curtailment_mwh": round(baseline.total_curtailment_mwh, 1),
                     "total_absorbed_mwh": round(baseline.total_absorbed_mwh, 1),
                     "absorption_rate": round(baseline.absorption_rate, 4)})
    for series_name, p, c in (("threshold_converter_input", p_serving, cond),
                              ("expected_value_converter_input", p_serving, cond)):
        hourly = (np.where(p >= thr, c, 0.0) if series_name.startswith("threshold") else p * c).tolist()
        r = simulate_hourly_capped(hourly, ESS_RATED_MW)
        ess_rows.append({"series": series_name, "threshold": thr if series_name.startswith("threshold") else None,
                         "total_curtailment_mwh": round(r.total_curtailment_mwh, 1),
                         "total_absorbed_mwh": round(r.total_absorbed_mwh, 1),
                         "absorption_rate": round(r.absorption_rate, 4),
                         "absorption_rate_abs_error_pp": round((r.absorption_rate - baseline.absorption_rate) * 100, 2)})

    # --- 아티팩트 저장 (stage1은 train_classifier가 저장한 보정 분류기를 재사용) ---
    p2 = save_artifact("curtailment_stage2_wind", stage2[deployed], FEATURES,
                       stage="2_conditional_regressor", trained_on_curtailed_hours_only=True,
                       stage1_artifact=STAGE1_NAME,
                       stage1_calibration_period=stage1_art["meta"].get("calibration_period"),
                       train_period=[str(train["dt"].min()), str(train["dt"].max())],
                       test_period=[TEST_START, TEST_END], input_generation=deployed,
                       ess_threshold=thr, threshold_selection=thr_info, mae_matrix=mae_rows)

    pd.DataFrame(stage1_rows).to_csv(os.path.join(MODELS_DIR, "curtailment_stage1_metrics.csv"), index=False)
    pd.DataFrame(mae_rows).to_csv(os.path.join(MODELS_DIR, "curtailment_stage2_mae.csv"), index=False)
    pd.DataFrame(expected_rows).to_csv(os.path.join(MODELS_DIR, "curtailment_regressor_metrics.csv"), index=False)
    pd.DataFrame(ess_rows).to_csv(os.path.join(MODELS_DIR, "curtailment_ess_eval.csv"), index=False)

    # --- 출력 ---
    print(f"\n[stage1={STAGE1_NAME}] 2023 분류 성능 (/predict의 curtailment_probability 출처)")
    for r in stage1_rows:
        print(f"  {r['input']:>22}: AUC={r['auc']} PR-AUC={r['pr_auc']} Brier={r['brier']} "
              f"top5={r['top5_capture']}(최대 {r['top5_ceiling']}) prec@5%={r['top5_precision']}")

    print("\n[stage2] 제어시간 MAE (학습입력 x 평가입력, 2023 제어시간 %d시간)" % int(curtailed.sum()))
    print(mae_df.to_string(index=False))
    print(f"  -> 배포: trained_on={deployed} (서빙 입력=컨버터 기준 MAE 최소) -> {p2}")
    print(f"  -> stage1: {STAGE1_NAME} (train_classifier가 저장, 보정구간 {stage1_art['meta'].get('calibration_period')})")

    print("\n[(a) 기댓값 방식] 2023 총합 (실측 %.0fMWh)" % actual_total)
    for r in expected_rows:
        print(f"  {r['input']:>22}: {r['predicted_total_mwh_2023']:>9.1f}MWh ({r['total_error_pct']:+.1f}%)")

    print(f"\n[(b) 임계값 방식 -> ESS {ESS_RATED_MW}MW hourly_capped]")
    print(f"  임계값 {thr} (2022년 검증구간에서 F1 최대, F1={thr_info['f1']} "
          f"정밀도={thr_info['precision']} 재현율={thr_info['recall']})\n"
          f"     선정구간 {thr_info['selection_period']}, 선정모델 {thr_info['selection_model']}")
    for r in ess_rows:
        extra = f" (실측 대비 {r['absorption_rate_abs_error_pp']:+.2f}%p)" if "absorption_rate_abs_error_pp" in r else " ← 기준"
        print(f"  {r['series']:>30}: 제어량합 {r['total_curtailment_mwh']:>8.1f}MWh "
              f"흡수 {r['total_absorbed_mwh']:>7.1f}MWh 흡수율 {r['absorption_rate']:.1%}{extra}")

    return {"mae": mae_rows, "expected": expected_rows, "ess": ess_rows, "threshold": thr_info}


if __name__ == "__main__":
    main()
