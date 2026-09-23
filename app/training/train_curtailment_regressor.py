"""
풍력 제어량(MWh) 회귀모델 학습 — 풍력 전용.

계획서 08장 로드맵에 명시된 "제어량 회귀모델(RandomForestRegressor, 풍력 전용)".
태양광은 이 모델을 만들 수 없다 — 전력거래소가 태양광 제어량(MWh)을 공식적으로 산정하지 않기
때문이다(계획서 07장, data.go.kr 메타데이터로 확인됨). 이 스크립트를 태양광에 억지로 돌리지 말 것.

타깃(curtailment_mwh)은 대부분의 시간에 0이고, 제어가 발생한 시간에만 양수인 분포다.

[수정 사항]
- 이전에는 출력제어 발생일만으로 학습해, 제어가 없는 평범한 날에도 양수 제어량을 예측하는
  경향이 있었다. 전체 달력(build_labeled_hourly)으로 재구성 — 제어 없는 시간은 제어량 0.
- 평가 지표를 '전체 합계 오차'뿐 아니라 '제어 발생 시간 합계/오차'와 '비제어 시간에 잘못 예측한
  총량(false_mwh_on_clear_hours)'으로 나눠 보고한다.

실행: python -m app.training.train_curtailment_regressor
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from app.data_prep import add_time_features, build_labeled_hourly, load_demand_actual
from app.model_io import MODELS_DIR, save_artifact

FEATURES = ["generation_mwh", "hour_sin", "hour_cos", "month_sin", "month_cos", "demand_mw"]


def main() -> dict:
    df = add_time_features(build_labeled_hourly("wind"))
    df = df.merge(load_demand_actual(), on="dt", how="inner").dropna(subset=FEATURES + ["curtailment_mwh"])

    train = df[df["dt"] < "2023-01-01"]
    test = df[df["dt"] >= "2023-01-01"]

    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["curtailment_mwh"])

    pred = np.clip(model.predict(test[FEATURES]), 0, None)
    y = test["curtailment_mwh"].to_numpy()
    curtailed = y > 0
    actual_total, pred_total = float(y.sum()), float(pred.sum())

    metrics = {
        "n_train": len(train), "n_test": len(test), "n_curtailed_hours_test": int(curtailed.sum()),
        "actual_total_mwh_2023": round(actual_total, 1),
        "predicted_total_mwh_2023": round(pred_total, 1),
        "total_error_pct": round((pred_total - actual_total) / actual_total * 100, 1) if actual_total else None,
        "predicted_on_curtailed_hours_mwh": round(float(pred[curtailed].sum()), 1),
        "false_mwh_on_clear_hours": round(float(pred[~curtailed].sum()), 1),
        "mae_on_curtailed_hours_mwh": round(float(mean_absolute_error(y[curtailed], pred[curtailed])), 2)
        if curtailed.any() else None,
    }
    out_path = save_artifact(
        "curtailment_regressor_wind", model, FEATURES,
        train_period=[str(train["dt"].min()), str(train["dt"].max())],
        test_period=["2023-01-01", "2024-01-01"], input_generation="actual", test_metrics=metrics,
    )
    print(f"[curtailment_regressor:wind] 2023 실측 총합={metrics['actual_total_mwh_2023']}MWh, "
          f"예측 총합={metrics['predicted_total_mwh_2023']}MWh (오차 {metrics['total_error_pct']}%)\n"
          f"  └ 제어시간 예측합={metrics['predicted_on_curtailed_hours_mwh']}MWh, "
          f"비제어시간 오예측합={metrics['false_mwh_on_clear_hours']}MWh, "
          f"제어시간 MAE={metrics['mae_on_curtailed_hours_mwh']}MWh -> {out_path}")
    pd.DataFrame([metrics]).to_csv(os.path.join(MODELS_DIR, "curtailment_regressor_metrics.csv"), index=False)
    return metrics


if __name__ == "__main__":
    main()
