"""
태양광 "조건부 잠재발전량 모델" — 07장에서 남겨둔 3번 과제.

배경(계획서 07장): 태양광은 전력거래소가 출력제어 '발생 시간'(2023년 282시간)만 공개하고
시간당 제어량(MWh)은 공개하지 않는다. 과거 시도한 두 방법(전체기간 월·시 중앙값 기준 / 상위 99분위
기준)은 모두 실패했다 — 태양광 설비가 2019~2023년 사이 정오 평균 80MWh→186MWh로 2배 이상
증설되면서, 과거 데이터를 통째로 섞은 기준값이 2023년의 실제 잠재발전량을 구조적으로 낮게 잡아
"제어 후 발전량"이 "과거 기준값"보다 오히려 높게 나오는 역전이 발생했기 때문이다.

이 스크립트는 07장이 제안한 해법을 실제로 시도한다:
"설비증설 추세를 반영한 조건부 잠재발전량 모델(예: 날씨→발전량 변환모델을 출력제어가 없었던 시간만으로
재학습해 제어 시간의 실제 기상 데이터에 적용)"

방법:
1. 출력제어가 없었던 시간만으로 날씨->발전량 회귀모델을 학습하되, 설비증설 추세를 모델이 인식할 수
   있도록 연속적인 시간추세 피처(trend_years = 관측시점의 연 단위 경과시간)를 입력에 포함한다.
2. 이 모델이 실제로 추세를 따라가는지 먼저 검증한다 — 비제어 시간을 2019~2022(학습)/2023(테스트)로
   리키지-프리 분할해 corr/NMAE를 확인. (여기서 잘 맞아야 3번 단계의 추정치를 신뢰할 수 있다.)
3. 검증되면 전체 비제어 시간(2019~2023)으로 최종 모델을 재학습하고, 2023년 282개 제어 시간의 실제
   기상데이터에 적용해 "그 시간에 제어가 없었다면 얼마나 발전했을까"(잠재발전량)를 추정한다.
4. 제어량(MWh) = max(0, 잠재발전량 - 실제(제어후)발전량) 으로 시간별 추정, 합산.
5. 음수(잠재발전량 < 실제발전량, 즉 여전히 역전되는 시간)가 몇 건인지도 정직하게 집계한다 —
   방법A/B와 같은 실패 패턴이 남아있는지 확인하기 위함.

실행: python -m app.training.solar_potential_curtailment
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from app.data_prep import add_time_features, load_asos, load_curtailment, load_generation_actual

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

FEATURES = ["solar_rad", "temp", "cloud", "hour_sin", "hour_cos", "month_sin", "month_cos", "trend_years"]


def _nmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.mean(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return mean_absolute_error(y_true, y_pred) / denom * 100


def build_dataset() -> pd.DataFrame:
    weather = add_time_features(load_asos("184"))
    gen = load_generation_actual("solar")
    curt = load_curtailment("solar")  # is_curtailed만 신뢰 가능, curtailment_mwh는 NaN(공식 미산정)

    df = weather.merge(gen, on="dt", how="inner").merge(curt[["dt", "is_curtailed"]], on="dt", how="left")
    df["is_curtailed"] = df["is_curtailed"].fillna(False).astype(bool)

    epoch = df["dt"].min()
    df["trend_years"] = (df["dt"] - epoch).dt.total_seconds() / (365.25 * 24 * 3600)

    return df.dropna(subset=["solar_rad", "temp", "cloud", "generation_mwh"]).sort_values("dt").reset_index(drop=True)


def validate_trend_awareness(df: pd.DataFrame) -> dict:
    """비제어 시간만으로, 2019~2022 학습 -> 2023 테스트 리키지-프리 검증.

    설비증설 추세를 trend_years 피처가 제대로 따라가는지 먼저 확인 — 여기서 성능이 나빠야
    (즉 결과가 신뢰 안 되면) 3번 단계의 잠재발전량 추정치도 신뢰할 수 없다.
    """
    clean = df[~df["is_curtailed"]]
    train = clean[clean["dt"] < "2023-01-01"]
    test = clean[clean["dt"] >= "2023-01-01"]

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["generation_mwh"])
    pred = model.predict(test[FEATURES])

    corr = float(np.corrcoef(test["generation_mwh"], pred)[0, 1])
    nmae = _nmae(test["generation_mwh"].to_numpy(), pred)
    bias = float(np.mean(pred - test["generation_mwh"]))  # 양수면 과대추정 경향
    print(f"[검증: 비제어시간 2019-22 학습 -> 2023 테스트] corr={corr:.4f} NMAE={nmae:.2f}% "
          f"평균편향={bias:+.2f}MWh (n_train={len(train)}, n_test={len(test)})")
    return {"corr": round(corr, 4), "nmae_pct": round(nmae, 2), "bias_mwh": round(bias, 2)}


def estimate_curtailment(df: pd.DataFrame) -> pd.DataFrame:
    """전체 비제어 시간으로 최종 모델 재학습 -> 제어 시간에 적용해 제어량 추정."""
    clean = df[~df["is_curtailed"]]
    curtailed = df[df["is_curtailed"]].copy()

    model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(clean[FEATURES], clean["generation_mwh"])

    curtailed["potential_generation_mwh"] = model.predict(curtailed[FEATURES])
    curtailed["estimated_curtailment_mwh"] = (
        curtailed["potential_generation_mwh"] - curtailed["generation_mwh"]
    ).clip(lower=0)
    curtailed["reversed"] = curtailed["potential_generation_mwh"] < curtailed["generation_mwh"]
    return curtailed[["dt", "generation_mwh", "potential_generation_mwh", "estimated_curtailment_mwh", "reversed"]]


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"전체 데이터셋: {len(dataset)}행 (2023년 제어시간={int(dataset[dataset['dt'] >= '2023-01-01']['is_curtailed'].sum())}건)")

    val = validate_trend_awareness(dataset)

    result = estimate_curtailment(dataset)  # 학습은 전체 비제어시간, 적용은 전체 제어시간
    result_2023 = result[result["dt"] >= "2023-01-01"]
    n_total = len(result_2023)
    n_reversed = int(result_2023["reversed"].sum())
    total_mwh = float(result_2023["estimated_curtailment_mwh"].sum())

    print(f"\n=== 2023년 태양광 제어시간 {n_total}건 잠재발전량 기반 추정 ===")
    print(f"역전(잠재<실제, 추정 0으로 처리)된 시간: {n_reversed}건 ({n_reversed / n_total * 100:.1f}%)")
    print(f"추정 총 제어량: {total_mwh:.1f} MWh (시간당 평균 {total_mwh / n_total:.2f} MWh)")
    print(f"\n(참고 — 기존 방법A 중앙값기준: 약 43MWh, 방법B 99분위기준: 약 4,736MWh, 역전 36%)")

    out_path = os.path.join(MODELS_DIR, "solar_potential_curtailment_2023.csv")
    result_2023.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")
