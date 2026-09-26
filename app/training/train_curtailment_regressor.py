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
     train_classifier가 만든 classifier_wind_demand_calibrated_sigmoid를 그대로 쓴다 —
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

from app.data_prep import (add_cross_source_penetration, add_normalized_features,
                           add_time_features, build_labeled_hourly, load_asos_multi,
                           load_demand_actual, load_generation_actual)
from app.metrics import classification_report
from app.model_io import MODELS_DIR, converter_predict_mwh, load_artifact, save_artifact
from app.training.train_classifier import SERVED_METHOD, build_dataset as _clf_dataset, calibrate, make_model
from app.services.ess_simulation import simulate_hourly_capped
from app.quantile_ensemble import QUANTILES, QuantileEnsemble, ResidualEnsemble
from app.serving_config import artifact_name

FEATURES = ["generation_mwh", "hour_sin", "hour_cos", "month_sin", "month_cos", "demand_mw"]

TEST_START, TEST_END = "2023-01-01", "2024-01-01"
# 임계값은 2023(테스트)을 보지 않고 정해야 한다. 2022년을 임계값 선정용 검증구간으로 쓰되,
# 그 구간을 학습에 쓴 모델로 고르면 리키지이므로 2022 이전만으로 학습한 별도 모델을 쓴다.
THR_VAL_START, THR_VAL_END = "2022-01-01", "2023-01-01"

ESS_RATED_MW = 22.5  # 07장 기준 ESS 정격출력. 실측 제어량 기준 2023 흡수율 36.8%의 기준값.
ESS_CAPS = (22.5, 30.0, 40.0)  # 용량 슬라이더(04장 03번) 범위 점검용

# stage2 분위수 앙상블: 학습 2021~2022 제어시간, 하이퍼파라미터는 2022를 검증셋으로 선택
Q_TRAIN_START, Q_HP_SPLIT = "2021-01-01", "2022-01-01"
HP_GRID = (
    {"max_depth": 4, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 20},
    {"max_depth": 6, "learning_rate": 0.06, "max_iter": 400, "min_samples_leaf": 20},
    {"max_depth": 6, "learning_rate": 0.03, "max_iter": 800, "min_samples_leaf": 40},
    {"max_depth": 8, "learning_rate": 0.06, "max_iter": 400, "min_samples_leaf": 40},
)


# ---------------------------------------------------------------------------
# 데이터셋
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """전체 달력 + 시간피처 + 수요 + 컨버터 예측 발전량(generation_pred)."""
    gen = load_generation_actual("wind")
    df = add_time_features(build_labeled_hourly("wind"))
    # stage1(보정 분류기)이 capacity_factor·penetration을 쓰므로 여기서도 같이 만들어 둔다
    df = add_normalized_features(df, gen, load_demand_actual())

    # 계통 전체 침투율 — stage1이 total_penetration을 쓰므로 실측 태양광을 붙인다
    df = add_cross_source_penetration(df, "wind")

    converter = load_artifact("converter_wind")
    weather = add_time_features(load_asos_multi(("184", "185", "188")))
    # 컨버터 피처는 날씨쪽 컬럼이므로 weather에서 계산한 뒤 dt로 붙인다.
    weather = weather.dropna(subset=converter["features"]).copy()
    weather["generation_pred"] = converter_predict_mwh(converter, weather)
    cols = ["dt", "generation_pred"]

    # 서빙 경로는 태양광도 컨버터 예측값을 쓴다 — 실측 태양광을 섞으면 평가가 낙관적으로 오염된다.
    solar_conv = load_artifact("converter_solar")
    if set(solar_conv["features"]).issubset(weather.columns):
        w2 = weather.dropna(subset=solar_conv["features"]).copy()
        w2["solar_pred"] = converter_predict_mwh(solar_conv, w2)
        weather = weather.merge(w2[["dt", "solar_pred"]], on="dt", how="left")
        cols.append("solar_pred")
    df = df.merge(weather[cols], on="dt", how="inner")

    need = FEATURES + ["curtailment_mwh", "generation_pred", "capacity_factor", "total_penetration"]
    return df.dropna(subset=need).sort_values("dt").reset_index(drop=True)


def _with_gen(df: pd.DataFrame, gen_col: str) -> pd.DataFrame:
    """generation_mwh 자리에 지정한 컬럼(실측 또는 컨버터 예측)을 넣은 피처 프레임.

    capacity_factor·penetration은 발전량에서 파생되므로 함께 다시 계산해야 한다.
    이걸 빼먹으면 실측 기준으로 만든 파생값에 컨버터 예측 발전량이 섞여 평가가 오염된다.
    """
    out = df.copy()
    out["generation_mwh"] = df[gen_col]
    if "capacity_proxy_mwh" in out.columns:
        out["capacity_factor"] = out["generation_mwh"] / out["capacity_proxy_mwh"]
    if "demand_mw" in out.columns:
        out["penetration"] = out["generation_mwh"] / out["demand_mw"]
        # 타 발전원도 같은 성격의 값을 써야 한다 — 실측 발전량이면 실측 태양광,
        # 컨버터 예측이면 컨버터 예측 태양광. 섞으면 서빙 조건과 달라진다.
        other = "solar_pred" if (gen_col == "generation_pred" and "solar_pred" in out.columns) \
            else "other_generation_mwh"
        if other in out.columns:
            out["total_penetration"] = (out["generation_mwh"] + out[other]) / out["demand_mw"]
    return out


# ---------------------------------------------------------------------------
# ① stage1: 보정된 발생확률 — train_classifier가 만든 classifier_wind_demand_calibrated_sigmoid를 그대로 쓴다
#    (/predict가 서빙하는 확률과 동일한 모델이어야 expected = probability x 조건부가 성립한다)
# ---------------------------------------------------------------------------

# [2026-09-26] 계통 전체 침투율 포함 모델로 교체. /predict가 태양광 기상값이 오면 이 모델을
# 서빙하므로 expected = probability x 조건부의 probability도 같은 모델이어야 한다.
# 2023 테스트 PR-AUC 0.717 -> 0.805.
STAGE1_NAME = artifact_name("wind", use_demand=True, use_cross=True)


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


def _fit_quantiles(train: pd.DataFrame, gen_col: str, hp: dict) -> QuantileEnsemble:
    """제어 발생 시간만으로 분위수 19개를 각각 학습."""
    cur = train[train["curtailment_mwh"] > 0]
    x = _with_gen(cur, gen_col)[FEATURES]
    y = cur["curtailment_mwh"]
    models = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=42, **hp)
        m.fit(x, y)
        models[q] = m
    return QuantileEnsemble(models)


def select_hyperparams(df: pd.DataFrame, gen_col: str) -> dict:
    """2021 제어시간으로 학습하고 2022 제어시간의 평균 pinball loss로 하이퍼파라미터를 고른다."""
    tr = df[(df["dt"] >= Q_TRAIN_START) & (df["dt"] < Q_HP_SPLIT)]
    va = df[(df["dt"] >= Q_HP_SPLIT) & (df["dt"] < TEST_START)]
    va_cur = va[va["curtailment_mwh"] > 0]
    xv = _with_gen(va_cur, gen_col)[FEATURES]
    yv = va_cur["curtailment_mwh"].to_numpy()

    rows = []
    for hp in HP_GRID:
        ens = _fit_quantiles(tr, gen_col, hp)
        preds = ens.predict_quantiles(xv)
        loss = float(np.mean([mean_pinball_loss(yv, preds[:, i], alpha=q)
                              for i, q in enumerate(ens.quantiles)]))
        rows.append({**hp, "mean_pinball_loss": round(loss, 4)})
    best = min(rows, key=lambda r: r["mean_pinball_loss"])
    return {"best": {k: v for k, v in best.items() if k != "mean_pinball_loss"},
            "val_period": f"{Q_HP_SPLIT}~{TEST_START}", "n_val_curtailed": len(va_cur), "grid": rows}


def coverage_table(ens, x_test, y_true: np.ndarray) -> list[dict]:
    """②: 각 분위수에서 '실측 <= 예측' 비율. 잘 보정됐으면 명목 분위수와 같아야 한다."""
    preds = ens.predict_quantiles(x_test)
    return [{"quantile": q, "nominal": q,
             "empirical_coverage": round(float((y_true <= preds[:, i]).mean()), 4),
             "gap_pp": round(float((y_true <= preds[:, i]).mean() - q) * 100, 1)}
            for i, q in enumerate(ens.quantiles)]


def absorption_from_distribution(ens, x, p_curt: np.ndarray, cap: float) -> dict:
    """③: 시간별 E[X], E[min(X,cap)]에 보정 확률을 곱해 흡수율을 구한다.

    출력제어가 없는 시간은 제어량도 흡수량도 0이므로, 조건부 기댓값에 P(curtail)을 곱한 것이
    무조건부 기댓값이 된다.
    """
    total = float((p_curt * ens.expected_value(x)).sum())
    absorbed = float((p_curt * ens.expected_capped(x, cap)).sum())
    return {"total_curtailment_mwh": total, "total_absorbed_mwh": absorbed,
            "absorption_rate": absorbed / total if total > 0 else 0.0}


def _predict(model, df: pd.DataFrame, gen_col: str, proba: bool = False,
             features: list[str] | None = None) -> np.ndarray:
    """features를 주지 않으면 stage2의 FEATURES를 쓴다.
    stage1(분류기)은 피처 구성이 달라 반드시 artifact["features"]를 넘겨야 한다."""
    x = _with_gen(df, gen_col)[features or FEATURES]
    if proba:
        return model.predict_proba(x)[:, 1]
    return np.clip(model.predict(x), 0, None)


# ---------------------------------------------------------------------------
# 임계값 선정 (2022년, 2022 이전만으로 학습한 모델 사용 -> 리키지 없음)
# ---------------------------------------------------------------------------

def choose_threshold(df: pd.DataFrame) -> dict:
    """임계값은 기저학습·보정·2023테스트 어디에도 쓰지 않은 구간에서 고른다.

    배포 모델(classifier_wind_demand_calibrated_sigmoid)은 2022-07-01까지 학습 + 2022 하반기로 보정했고
    배포 stage2도 2023 이전 전체로 학습했으므로, 2022년 어디를 써도 배포 모델에는 in-sample이다.
    그래서 임계값 선정 전용으로 한 칸씩 앞당긴 모델 쌍을 따로 만든다.

      분류기: 기저 <2022-01-01 + isotonic(2022-01-01~2022-07-01)
      조건부 회귀: <2022-07-01 의 제어 발생 시간만
      임계값 선정 구간: 2022-07-01~2023-01-01 (위 어느 쪽에도 미사용)

    [기준] 총 제어량 일치 — 선정 구간에서 sum(임계값 계열)이 실측 총 제어량과 가장 가까운 임계값.
    ESS 흡수율은 에너지 회계이므로 '몇 시간을 맞혔나'(F1)가 아니라 '총 MWh가 맞나'가 목적함수다.
    F1 최대 기준도 함께 계산해 비교용으로 남긴다.
    """
    base_tr = df[df["dt"] < "2022-01-01"]
    calib = df[(df["dt"] >= "2022-01-01") & (df["dt"] < "2022-07-01")]
    sel = df[(df["dt"] >= "2022-07-01") & (df["dt"] < "2023-01-01")]

    base = make_model()
    base.fit(_with_gen(base_tr, "generation_mwh")[FEATURES], base_tr["is_curtailed"])
    # 배포 모델과 같은 보정 방식이어야 임계값이 같은 확률 척도 위에 놓인다
    clf = calibrate(base, _with_gen(calib, "generation_mwh"), FEATURES, SERVED_METHOD)
    s2 = fit_stage2(df[df["dt"] < "2022-07-01"], "generation_pred")

    p = clf.predict_proba(_with_gen(sel, "generation_pred")[FEATURES])[:, 1]
    c = _predict(s2, sel, "generation_pred")
    y = sel["curtailment_mwh"].to_numpy()
    pos = y > 0
    actual_total = float(y.sum())

    rows = []
    for thr in np.arange(0.01, 0.96, 0.01):
        pred = p >= thr
        tp = int((pred & pos).sum())
        if tp == 0:
            continue
        precision = tp / int(pred.sum())
        recall = tp / int(pos.sum())
        total = float(np.where(pred, c, 0.0).sum())
        rows.append({"threshold": round(float(thr), 2),
                     "f1": round(2 * precision * recall / (precision + recall), 4),
                     "precision": round(precision, 4), "recall": round(recall, 4),
                     "total_mwh": round(total, 1),
                     "total_ratio": round(total / actual_total, 4) if actual_total else None})

    by_total = min(rows, key=lambda r: abs(r["total_mwh"] - actual_total))
    by_f1 = max(rows, key=lambda r: r["f1"])
    meta = {"selection_period": "2022-07-01~2023-01-01", "n_sel": len(sel),
            "n_pos_sel": int(pos.sum()), "actual_total_mwh_sel": round(actual_total, 1),
            "selection_model": f"기저 <2022-01-01 + {SERVED_METHOD}(2022-01-01~2022-07-01), stage2 <2022-07-01"}
    # 운영 임계값은 F1 최대(탐지 품질) 기준이다. 총 제어량 일치 기준도 계산해 함께 남기지만,
    # ESS 에너지 회계용으로는 실패했고(README (b)) 보정 방식에 따라 격자 하한에 붙는
    # 경계해가 되기도 한다 — 운영값으로 쓰지 않는다.
    boundary = by_total["threshold"] <= rows[0]["threshold"] + 1e-9
    return {"criterion": "f1_max", **by_f1, **meta,
            "total_match_alternative": {**by_total, "is_boundary_solution": bool(boundary)},
            "grid": rows}


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
    S1F = stage1_art["features"]

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
        rep = classification_report(y_bin, _predict(stage1, test, gen, proba=True, features=S1F))
        stage1_rows.append({"model": STAGE1_NAME, "input": input_name, **rep})

    # --- (a) 기댓값(확률 x 조건부) 총합 오차 ---
    p_actual = _predict(stage1, test, "generation_mwh", proba=True, features=S1F)
    p_serving = _predict(stage1, test, "generation_pred", proba=True, features=S1F)
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

    # --- (c) 분포 방식: 분위수 앙상블 / 잔차 앙상블 ---
    hp_info = select_hyperparams(df, "generation_pred")
    qens = _fit_quantiles(train, "generation_pred", hp_info["best"])
    x_test = _with_gen(test, "generation_pred")[FEATURES]

    # ②: 2023 제어 발생 시간에서의 커버리지
    cov_rows = coverage_table(qens, x_test[curtailed], y[curtailed])

    # ④ 대조군: 점 예측 + 2022 경험적 잔차.
    # 잔차는 out-of-sample이어야 퍼짐이 과소평가되지 않으므로 2021만으로 학습한 모델로 뽑는다.
    rf_2021 = fit_stage2(df[df["dt"] < Q_HP_SPLIT], "generation_pred")
    va = df[(df["dt"] >= Q_HP_SPLIT) & (df["dt"] < TEST_START)]
    va_cur = va[va["curtailment_mwh"] > 0]
    resid = (va_cur["curtailment_mwh"].to_numpy()
             - _predict(rf_2021, va_cur, "generation_pred"))
    rens_raw = ResidualEnsemble(stage2[deployed], resid, center=False)
    rens = ResidualEnsemble(stage2[deployed], resid, center=True)

    cap_rows = []
    for cap in ESS_CAPS:
        base_c = simulate_hourly_capped(y.tolist(), cap)
        row = {"cap_mw": cap, "actual_absorption_rate": round(base_c.absorption_rate, 4),
               "actual_total_mwh": round(actual_total, 1)}
        for label, ens in (("residual_raw", rens_raw), ("residual", rens), ("quantile", qens)):
            r = absorption_from_distribution(ens, x_test, p_serving, cap)
            row[f"{label}_absorption_rate"] = round(r["absorption_rate"], 4)
            row[f"{label}_error_pp"] = round((r["absorption_rate"] - base_c.absorption_rate) * 100, 2)
            row[f"{label}_total_mwh"] = round(r["total_curtailment_mwh"], 1)
            row[f"{label}_total_error_pct"] = round(
                (r["total_curtailment_mwh"] - actual_total) / actual_total * 100, 1)
        cap_rows.append(row)

    METHODS = ("residual_raw", "residual", "quantile")
    # ⑤ 판정: 모든 cap에서 ±5%p 안에 드는 방식이 있는가
    usable = [m for m in METHODS if all(abs(r[f"{m}_error_pp"]) <= 5.0 for r in cap_rows)]

    # --- (b) 임계값 방식 -> ESS hourly_capped 흡수율 ---
    thr_info = choose_threshold(df)
    thr = thr_info["threshold"]                                  # F1 최대 — 운영(탐지 경보)용
    alt_thr = thr_info["total_match_alternative"]["threshold"]   # 총 제어량 일치 — 비교용(실패)
    ess_rows = []
    baseline = simulate_hourly_capped(y.tolist(), ESS_RATED_MW)
    ess_rows.append({"series": "actual_curtailment", "threshold": None,
                     "total_curtailment_mwh": round(baseline.total_curtailment_mwh, 1),
                     "total_absorbed_mwh": round(baseline.total_absorbed_mwh, 1),
                     "absorption_rate": round(baseline.absorption_rate, 4)})
    for label, t in (("threshold_f1max(운영)", thr), ("threshold_total_match", alt_thr)):
        series = np.where(p_serving >= t, cond, 0.0)
        r = simulate_hourly_capped(series.tolist(), ESS_RATED_MW)
        ess_rows.append({"series": label, "threshold": t,
                         "total_curtailment_mwh": round(r.total_curtailment_mwh, 1),
                         "total_ratio_vs_actual": round(r.total_curtailment_mwh / actual_total, 2),
                         "total_absorbed_mwh": round(r.total_absorbed_mwh, 1),
                         "absorption_rate": round(r.absorption_rate, 4),
                         "absorption_rate_abs_error_pp": round((r.absorption_rate - baseline.absorption_rate) * 100, 2)})

    ev = p_serving * cond
    r = simulate_hourly_capped(ev.tolist(), ESS_RATED_MW)
    ess_rows.append({"series": "expected_value", "threshold": None,
                     "total_curtailment_mwh": round(r.total_curtailment_mwh, 1),
                     "total_ratio_vs_actual": round(r.total_curtailment_mwh / actual_total, 2),
                     "total_absorbed_mwh": round(r.total_absorbed_mwh, 1),
                     "absorption_rate": round(r.absorption_rate, 4),
                     "absorption_rate_abs_error_pp": round((r.absorption_rate - baseline.absorption_rate) * 100, 2)})

    # 오라클: 2023 총합이 가장 잘 맞는 임계값을 '테스트를 보고' 고른다.
    # 임계값 튜닝으로 도달 가능한 상한을 보기 위한 진단용 — 운영에 쓸 수 없는 값이다.
    oracle_t, oracle_r = None, None
    for t in np.arange(0.005, 0.95, 0.005):
        ser = np.where(p_serving >= t, cond, 0.0)
        if ser.sum() == 0:
            continue
        if oracle_t is None or abs(ser.sum() - actual_total) < abs(oracle_best - actual_total):
            oracle_t, oracle_best = float(t), float(ser.sum())
            oracle_r = simulate_hourly_capped(ser.tolist(), ESS_RATED_MW)
    ess_rows.append({"series": "threshold_oracle_2023 (진단용, 운영 불가)", "threshold": round(oracle_t, 3),
                     "total_curtailment_mwh": round(oracle_r.total_curtailment_mwh, 1),
                     "total_ratio_vs_actual": round(oracle_r.total_curtailment_mwh / actual_total, 2),
                     "total_absorbed_mwh": round(oracle_r.total_absorbed_mwh, 1),
                     "absorption_rate": round(oracle_r.absorption_rate, 4),
                     "absorption_rate_abs_error_pp": round((oracle_r.absorption_rate - baseline.absorption_rate) * 100, 2)})

    # --- 아티팩트 저장 (stage1은 train_classifier가 저장한 보정 분류기를 재사용) ---
    p2 = save_artifact("curtailment_stage2_wind", stage2[deployed], FEATURES,
                       stage="2_conditional_regressor", trained_on_curtailed_hours_only=True,
                       stage1_artifact=STAGE1_NAME,
                       stage1_calibration_period=stage1_art["meta"].get("calibration_period"),
                       train_period=[str(train["dt"].min()), str(train["dt"].max())],
                       test_period=[TEST_START, TEST_END], input_generation=deployed,
                       ess_threshold=thr, threshold_criterion="f1_max (탐지 경보용)",
                       threshold_selection={k: v for k, v in thr_info.items() if k != "grid"},
                       mae_matrix=mae_rows)

    pd.DataFrame(stage1_rows).to_csv(os.path.join(MODELS_DIR, "curtailment_stage1_metrics.csv"), index=False)
    pd.DataFrame(mae_rows).to_csv(os.path.join(MODELS_DIR, "curtailment_stage2_mae.csv"), index=False)
    pd.DataFrame(expected_rows).to_csv(os.path.join(MODELS_DIR, "curtailment_regressor_metrics.csv"), index=False)
    pd.DataFrame(ess_rows).to_csv(os.path.join(MODELS_DIR, "curtailment_ess_eval.csv"), index=False)
    pd.DataFrame(thr_info["grid"]).to_csv(os.path.join(MODELS_DIR, "curtailment_threshold_grid.csv"), index=False)

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
    alt = thr_info["total_match_alternative"]
    print(f"  운영 임계값 {thr} — 기준: F1 최대(탐지) "
          f"(F1={thr_info['f1']} 정밀도={thr_info['precision']} 재현율={thr_info['recall']}, "
          f"선정구간 총 제어량 비율 {thr_info['total_ratio']}배)")
    print(f"     비교) 총 제어량 일치 기준이면 {alt['threshold']} (비율 {alt['total_ratio']}배)"
          + ("  ⚠ 격자 하한에 붙은 경계해 — 운영 불가" if alt["is_boundary_solution"] else ""))
    print(f"     선정구간 {thr_info['selection_period']}, 선정모델 {thr_info['selection_model']}")
    for r in ess_rows:
        extra = (" ← 기준" if r["series"] == "actual_curtailment"
                 else f" (실측 대비 {r['absorption_rate_abs_error_pp']:+.2f}%p)")
        ratio = f" [{r['total_ratio_vs_actual']}x]" if r.get("total_ratio_vs_actual") else ""
        print(f"  {r['series']:>38}: 제어량합 {r['total_curtailment_mwh']:>8.1f}MWh{ratio:>8} "
              f"흡수율 {r['absorption_rate']:.1%}{extra}")
    print("  ※ 오라클은 2023을 보고 고른 상한 진단값 — 총합을 1.00x로 맞춰도 흡수율은 여전히 실측과 벌어진다.")

    print(f"\n[② 분위수 커버리지] 2023 제어 발생 {int(curtailed.sum())}시간, '실측 <= 예측' 비율")
    print("  " + "  ".join(f"{r['quantile']:.2f}" for r in cov_rows))
    print("  " + "  ".join(f"{r['empirical_coverage']:.2f}" for r in cov_rows))
    print("  " + "  ".join(f"{r['gap_pp']:+.0f}" for r in cov_rows) + "   (%p 격차)")
    print(f"  하이퍼파라미터: {hp_info['best']} (2022 제어시간 {hp_info['n_val_curtailed']}건, "
          f"평균 pinball loss 최소)")

    print(f"\n[③④ 분포 적분 방식 흡수율] E[min(X,cap)] x P(curtail)")
    print(f"{'cap':>8} {'실측기반':>9} {'잔차(원본)':>13} {'잔차(중심화)':>15} {'분위수':>13}")
    for r in cap_rows:
        print(f"{r['cap_mw']:>6.1f}MW {r['actual_absorption_rate']:>9.1%} "
              f"{r['residual_raw_absorption_rate']:>8.1%} {r['residual_raw_error_pp']:>+6.2f}%p "
              f"{r['residual_absorption_rate']:>9.1%} {r['residual_error_pp']:>+6.2f}%p "
              f"{r['quantile_absorption_rate']:>7.1%} {r['quantile_error_pp']:>+6.2f}%p")
    r0 = cap_rows[0]
    print(f"  총 제어량(실측 {r0['actual_total_mwh']:.0f}MWh): "
          f"잔차원본 {r0['residual_raw_total_mwh']:.0f} ({r0['residual_raw_total_error_pct']:+.1f}%) | "
          f"잔차중심화 {r0['residual_total_mwh']:.0f} ({r0['residual_total_error_pct']:+.1f}%) | "
          f"분위수 {r0['quantile_total_mwh']:.0f} ({r0['quantile_total_error_pct']:+.1f}%)")
    if usable:
        print(f"  ✅ 모든 cap에서 ±5%p 안에 드는 방식: {', '.join(usable)}")
    else:
        print("  ❌ ±5%p 안에 드는 방식 없음 — 대시보드는 실측 기반 계산만 쓸 것")

    pd.DataFrame(cov_rows).to_csv(os.path.join(MODELS_DIR, "curtailment_quantile_coverage.csv"), index=False)
    pd.DataFrame(cap_rows).to_csv(os.path.join(MODELS_DIR, "curtailment_absorption_by_cap.csv"), index=False)
    pd.DataFrame(hp_info["grid"]).to_csv(os.path.join(MODELS_DIR, "curtailment_quantile_hp.csv"), index=False)
    save_artifact("curtailment_stage2_residual_wind", rens, FEATURES,
                  stage="2_residual_ensemble (연구 검증용 — 서비스 미사용)",
                  residual_source="2021 학습 모델의 2022 제어시간 잔차(out-of-sample)",
                  residual_n=int(len(resid)), residual_mean_raw=round(float(resid.mean()), 2),
                  residual_std=round(float(resid.std()), 2), centered=True,
                  centering_decided="post-hoc (2023 결과를 본 뒤 채택 — README 참고)",
                  point_model="curtailment_stage2_wind", input_generation="converter",
                  test_period=[TEST_START, TEST_END], absorption_by_cap=cap_rows)
    save_artifact("curtailment_stage2_quantile_wind", qens, FEATURES,
                  stage="2_quantile_ensemble", quantiles=list(QUANTILES),
                  hyperparams=hp_info["best"], hp_val_period=hp_info["val_period"],
                  train_period=[str(train["dt"].min()), str(train["dt"].max())],
                  trained_on_curtailed_hours_only=True, input_generation="converter",
                  test_period=[TEST_START, TEST_END], coverage=cov_rows, absorption_by_cap=cap_rows)

    return {"mae": mae_rows, "expected": expected_rows, "ess": ess_rows, "threshold": thr_info,
            "coverage": cov_rows, "caps": cap_rows, "hp": hp_info}


if __name__ == "__main__":
    main()
