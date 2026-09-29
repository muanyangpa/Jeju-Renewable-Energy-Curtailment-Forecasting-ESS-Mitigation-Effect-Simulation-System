"""
날씨->발전량 변환모델(RandomForestRegressor) 학습.

계획서 02장/05장에 정의된 ②단계 모델. H장에서 확인된 교훈(시간 피처 없으면 상관계수 0.54,
추가하면 0.90 이상)을 그대로 반영해 hour_sin/cos, month_sin/cos를 입력에 포함한다.

[수정 사항] data_prep의 24시 정렬 버그 수정이 반영되므로 재학습 필요. 아티팩트에 메타데이터 저장.

실행: python -m app.training.train_converter
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from app.data_prep import (add_time_features, capacity_proxy, latest_capacity_proxy, load_asos,
                           load_asos_multi, load_generation_actual)
from app.model_io import MODELS_DIR, feature_ranges, save_artifact

SOLAR_FEATURES = ["solar_rad", "temp", "cloud", "hour_sin", "hour_cos", "month_sin", "month_cos"]
WIND_FEATURES = ["wind_speed", "hour_sin", "hour_cos", "month_sin", "month_cos"]

# [2026-09-25] 발전원별로 타깃을 다르게 쓴다 — 원인이 다르므로 처방도 다르다.
#   태양광: 설비 증설로 타깃이 해마다 커져 RandomForest가 학습 최대(269MWh)를 못 넘었다
#           (2023 예측 최대 251MWh, 실측 336MWh). 이용률을 타깃으로 두고 예측 후 되곱하면
#           외삽이 가능해진다. NMAE 29.63% -> 25.44%(고정 대리지표 기준).
#   풍력  : 증설이 거의 없어 2023 실측이 학습 최대를 넘는 시간이 0.03%뿐이다.
#           정규화해도 42.04% -> 42.06%로 변화가 없어 현행 유지. 풍력 오차의 원인은
#           외삽이 아니라 관측지점과 풍력단지의 위치 불일치이고, 3지점 평균으로 이미 대응했다.
NORMALIZED_TARGET = {"solar": True, "wind": False}

# 테스트 구간도 발전원별로 다르다 — 쓸 수 있는 데이터 범위가 다르기 때문이다.
#   태양광: 2026-01까지 확보 -> 최근 1년(2025)을 테스트로 사용
#   풍력  : 제도 전환으로 2024-06부터 집계가 끊겨 2023년 테스트를 유지
TEST_SPLIT = {"solar": ("2025-01-01", "2026-01-01"), "wind": ("2023-01-01", "2024-01-01")}

# 발전원별 데이터 출처 — 두 출처는 모집단이 다르므로 섞지 않는다(data_prep 주석 참고).
#   태양광: 신규(전력시장 거래량)만. 2024 학습 / 2025 테스트. 혼합보다 정확했다(22.74% vs 25.09%)
#   풍력  : 기존 + 신규(2024-05까지). 신규 단독으로는 표본이 부족하다.
GEN_SOURCE = {"solar": "market", "wind": "all"}


def _nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.mean(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return mean_absolute_error(y_true, y_pred) / denom * 100


def train_one(energy_type: str) -> dict:
    if energy_type == "wind":
        # I장 재검증 결과: 3지점(제주184·고산185·성산188) 평균이 단일지점보다 위치오차를 상쇄해
        # 더 정확하다(corr 0.656->0.763, NMAE 15.58%->12.03%). 풍력은 3지점 평균을 사용.
        weather = load_asos_multi(("184", "185", "188"))
        station = "184+185+188(평균)"
    else:
        weather = load_asos("184")  # 태양광은 일사량 기준 지점 하나로 충분(H장 검증)
        station = "184"
    weather = add_time_features(weather)
    gen = load_generation_actual(energy_type, source=GEN_SOURCE[energy_type])

    df = weather.merge(gen, on="dt", how="inner")
    features = SOLAR_FEATURES if energy_type == "solar" else WIND_FEATURES
    normalized = NORMALIZED_TARGET[energy_type]

    if normalized:
        # 타깃 정규화용 대리지표는 '시점별 인과적' 값을 쓴다(과거만 사용).
        df["capacity_proxy_mwh"] = df["dt"].map(capacity_proxy(gen))
        df = df.dropna(subset=["capacity_proxy_mwh"])
        df["target"] = df["generation_mwh"] / df["capacity_proxy_mwh"]
    else:
        df["target"] = df["generation_mwh"]
    df = df.dropna(subset=features + ["target", "generation_mwh"])

    # 리키지-프리 시간분할: 테스트 구간 이전만 학습에 사용
    t0, t1 = TEST_SPLIT[energy_type]
    train = df[df["dt"] < t0]
    test = df[(df["dt"] >= t0) & (df["dt"] < t1)]

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train[features], train["target"])

    # 서빙용 고정 대리지표. 분류기와 같은 상수를 써야 MWh<->이용률 왕복이 상쇄된다.
    serve_proxy = latest_capacity_proxy(gen) if normalized else None
    # 평가는 '학습 시점에 알 수 있었던' 값으로 해야 정직하다(전체 이력의 마지막 값을 쓰면
    # 2023 테스트에 미래 정보가 들어간다).
    eval_proxy = float(train["capacity_proxy_mwh"].iloc[-1]) if normalized else None

    raw = np.clip(model.predict(test[features]), 0, None)
    pred = raw * eval_proxy if normalized else raw
    corr = np.corrcoef(test["generation_mwh"], pred)[0, 1]
    nmae = _nmae(test["generation_mwh"].to_numpy(), pred)

    out_path = save_artifact(
        f"converter_{energy_type}", model, features,
        station=station,
        target="capacity_factor" if normalized else "generation_mwh",
        capacity_proxy_mwh=serve_proxy,      # 서빙에서 곱하는 상수 (정규화 모델만)
        eval_capacity_proxy_mwh=eval_proxy,  # 아래 지표를 낼 때 쓴 값
        train_period=[str(train["dt"].min()), str(train["dt"].max())],
        test_period=[t0, t1],
        test_corr=round(float(corr), 4), test_nmae_pct=round(float(nmae), 2),
        # 서빙에서 '기상 입력'의 드리프트를 감지하기 위한 학습 분포 범위.
        # 분류기 쪽 범위(파생 피처)만 보면 컨버터 자신의 외삽 실패를 놓친다 — 컨버터가 학습
        # 범위 밖으로 나가지 못해 발전량을 눌러 출력하면 그 눌린 값은 분류기 범위 안에 들어와
        # 조용해진다(풍속 14m/s가 9m/s보다 조용했던 이유). 원본 기상값을 직접 봐야 한다.
        train_feature_ranges=feature_ranges(train, features),
    )

    metrics = {"energy_type": energy_type, "station": station, "test_period": f"{t0}~{t1}",
               "target": "capacity_factor" if normalized else "generation_mwh",
               "corr": round(float(corr), 4), "nmae_pct": round(float(nmae), 2),
               "n_train": len(train), "n_test": len(test)}
    extra = f" [이용률 타깃, 서빙 대리지표={serve_proxy}MWh]" if normalized else ""
    print(f"[converter:{energy_type}] corr={metrics['corr']} NMAE={metrics['nmae_pct']}% "
          f"(train={metrics['n_train']}, test={metrics['n_test']}){extra} -> {out_path}")
    return metrics


if __name__ == "__main__":
    results = [train_one("solar"), train_one("wind")]
    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "converter_metrics.csv"), index=False)
