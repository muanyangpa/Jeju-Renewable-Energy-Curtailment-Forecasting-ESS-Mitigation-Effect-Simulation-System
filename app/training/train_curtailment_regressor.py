"""
풍력 제어량(MWh) 회귀모델 학습 — 풍력 전용.

계획서 08장 로드맵에 명시된 "제어량 회귀모델(RandomForestRegressor, 풍력 전용)".
태양광은 이 모델을 만들 수 없다 — 전력거래소가 태양광 제어량(MWh)을 공식적으로 산정하지 않기
때문이다(계획서 07장, data.go.kr 메타데이터로 확인됨). 이 스크립트를 태양광에 억지로 돌리지 말 것.

타깃(curtailment_mwh)은 대부분의 시간에 0이고, 제어가 발생한 시간에만 양수인 분포다.
회귀 자체는 전 시간대를 대상으로 하되(0을 포함), 평가는 07장과 동일하게 2023년 전체 합계·
제어시간 평균오차를 함께 본다.

실행: python -m app.training.train_curtailment_regressor
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from app.data_prep import add_time_features, load_curtailment, load_demand_actual, load_generation_actual

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
FEATURES = ["generation_mwh", "hour_sin", "hour_cos", "month_sin", "month_cos", "demand_mw"]


def main() -> dict:
    gen = load_generation_actual("wind")
    gen = add_time_features(gen)
    curt = load_curtailment("wind")[["dt", "curtailment_mwh"]]
    demand = load_demand_actual()

    df = gen.merge(curt, on="dt", how="inner").merge(demand, on="dt", how="inner").dropna()

    train = df[df["dt"] < "2023-01-01"]
    test = df[df["dt"] >= "2023-01-01"]

    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["curtailment_mwh"])

    pred = np.clip(model.predict(test[FEATURES]), 0, None)
    actual_total = test["curtailment_mwh"].sum()
    pred_total = pred.sum()

    curtailed_mask = test["curtailment_mwh"] > 0
    mae_on_curtailed_hours = mean_absolute_error(test.loc[curtailed_mask, "curtailment_mwh"], pred[curtailed_mask.to_numpy()])

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "curtailment_regressor_wind.joblib")
    joblib.dump({"model": model, "features": FEATURES}, out_path)

    metrics = {
        "n_train": len(train), "n_test": len(test),
        "actual_total_mwh_2023": round(float(actual_total), 1),
        "predicted_total_mwh_2023": round(float(pred_total), 1),
        "total_error_pct": round(float((pred_total - actual_total) / actual_total * 100), 1),
        "mae_on_curtailed_hours_mwh": round(float(mae_on_curtailed_hours), 2),
    }
    print(f"[curtailment_regressor:wind] 2023 실측 총합={metrics['actual_total_mwh_2023']}MWh, "
          f"예측 총합={metrics['predicted_total_mwh_2023']}MWh "
          f"(오차 {metrics['total_error_pct']}%), 제어시간 평균오차={metrics['mae_on_curtailed_hours_mwh']}MWh "
          f"-> {out_path}")
    pd.DataFrame([metrics]).to_csv(os.path.join(MODELS_DIR, "curtailment_regressor_metrics.csv"), index=False)
    return metrics


if __name__ == "__main__":
    main()
