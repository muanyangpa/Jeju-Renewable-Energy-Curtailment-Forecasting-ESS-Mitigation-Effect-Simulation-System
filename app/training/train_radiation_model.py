"""일사량 추정 모델 — 기상청 단기예보로 태양광 경로를 돌리기 위한 마지막 조각.

[왜 필요한가]
단기예보 14개 항목에 일사량이 없다(app/solar_radiation.py 주석 참고). 태양광 컨버터는
`solar_rad`를 필수로 쓰므로, 예보만으로 돌리려면 일사량을 예보 항목에서 추정해야 한다.

    일사량 ≈ 청천일사량(물리, 결정론적) × 감쇠계수(하늘상태·습도·기온으로 학습)

학습·평가는 ASOS 실측으로 한다. 단, **입력을 예보 수준으로 떨어뜨려서** 학습한다 —
전운량(0~10)을 SKY 코드(1/3/4)로 양자화하고 예보에 있는 항목만 쓴다. 그러지 않으면 실측
전운량으로 학습한 모델을 SKY만 오는 예보에 넣게 되어 입력 분포가 어긋난다.

[이 모델이 측정하는 것과 못 하는 것]
  측정한다   — 예보 항목만 쓸 때의 '정보 손실'(전운량 -> SKY 양자화, 일사량 -> 추정)
  못 한다    — 예보 자체의 오차. ASOS 실측 SKY/기온/습도를 넣으므로 '완벽한 예보' 가정이다.
             실제 운영은 여기에 예보 오차가 더 얹힌다. 즉 이 결과도 **상한**이다.

두 가지 타깃을 비교한다.
  직접   : 일사량(MJ/m²)을 바로 회귀
  청천지수: kt = 일사량/청천일사량을 회귀하고 청천일사량을 되곱 — 야간에 0이 보장되고
           계절·시각 의존을 물리가 흡수하므로 표본 대비 유리할 수 있다

실행: python -m app.training.train_radiation_model
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from app.data_prep import add_time_features, load_asos
from app.model_io import MODELS_DIR, feature_ranges, save_artifact
from app.solar_radiation import add_solar_geometry, sky_from_cloud

STATION = "184"                       # 태양광 컨버터가 쓰는 지점(H장 검증)
TEST_START, TEST_END = "2025-01-01", "2026-01-01"   # 태양광 컨버터 테스트 구간과 정렬

# 예보에 실제로 있는 항목만 쓴다. cloud는 SKY로 양자화해서 넣는다.
FEATURES = ["clear_sky_mj", "sky", "temp", "humidity",
            "hour_sin", "hour_cos", "month_sin", "month_cos"]
# 청천일사량이 이보다 작으면 야간·박명으로 보고 0을 반환한다(비율 모델의 분모 폭발 방지).
NIGHT_MJ = 0.02


def build_dataset() -> pd.DataFrame:
    df = add_time_features(load_asos(STATION))
    df = add_solar_geometry(df, STATION)
    df["sky"] = sky_from_cloud(df["cloud"])
    df = df.dropna(subset=FEATURES + ["solar_rad"]).reset_index(drop=True)
    return df


def _model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)


def main() -> pd.DataFrame:
    df = build_dataset()
    tr = df[df.dt < TEST_START]
    te = df[(df.dt >= TEST_START) & (df.dt < TEST_END)]
    day_tr, day_te = tr[tr.clear_sky_mj > NIGHT_MJ], te[te.clear_sky_mj > NIGHT_MJ]
    print(f"학습 {len(tr)}시간(주간 {len(day_tr)}) / 테스트 {len(te)}시간(주간 {len(day_te)})")
    print(f"실측 일사량: 학습 평균 {tr.solar_rad.mean():.3f} / 테스트 평균 {te.solar_rad.mean():.3f} MJ/m²\n")

    y_te = te.solar_rad.to_numpy()
    preds: dict[str, np.ndarray] = {}
    models: dict[str, object] = {}

    # ① 직접 회귀 — 주간만 학습하고 야간은 0으로 둔다(야간 표본이 절반을 넘어 손실을 지배한다)
    m = _model(); m.fit(day_tr[FEATURES], day_tr.solar_rad)
    p = np.zeros(len(te))
    mask = (te.clear_sky_mj > NIGHT_MJ).to_numpy()
    p[mask] = np.clip(m.predict(te.loc[mask, FEATURES]), 0, None)
    preds["직접 (MJ/m² 회귀)"] = p; models["direct"] = m

    # ② 청천지수 회귀 — kt를 학습하고 청천일사량을 되곱한다
    kt_tr = (day_tr.solar_rad / day_tr.clear_sky_mj).clip(0, 1.2)
    mk = _model(); mk.fit(day_tr[FEATURES], kt_tr)
    p = np.zeros(len(te))
    p[mask] = np.clip(mk.predict(te.loc[mask, FEATURES]), 0, 1.2) * te.loc[mask, "clear_sky_mj"]
    preds["청천지수 (kt 회귀 x 청천)"] = p; models["kt"] = mk

    # ③ 기준선 — 하늘상태별 평균 청천지수만 쓰는 물리+상수 모델(학습 거의 없음)
    kt_by_sky = kt_tr.groupby(day_tr.sky).mean()
    p = np.zeros(len(te))
    p[mask] = te.loc[mask, "sky"].map(kt_by_sky).to_numpy() * te.loc[mask, "clear_sky_mj"]
    preds["기준선 (SKY별 평균 kt)"] = np.nan_to_num(p); models["baseline"] = kt_by_sky.to_dict()
    print("SKY별 평균 청천지수(학습 구간): "
          + " · ".join(f"{int(k)}={v:.3f}" for k, v in kt_by_sky.items()))

    rows = []
    print(f"\n{'모델':26} {'MAE':>8} {'주간 MAE':>9} {'R²':>8} {'주간 NMAE':>10}")
    for name, p in preds.items():
        mae = mean_absolute_error(y_te, p)
        dmae = mean_absolute_error(y_te[mask], p[mask])
        nmae = 100 * dmae / y_te[mask].mean()
        rows.append({"model": name, "mae": round(float(mae), 4),
                     "day_mae": round(float(dmae), 4), "r2": round(float(r2_score(y_te, p)), 4),
                     "day_nmae_pct": round(float(nmae), 2)})
        print(f"{name:26} {mae:8.4f} {dmae:9.4f} {r2_score(y_te, p):+8.4f} {nmae:9.2f}%")

    best = min(rows, key=lambda r: r["day_mae"])
    print(f"\n채택: {best['model']} (주간 MAE {best['day_mae']})")
    key = "kt" if best["model"].startswith("청천지수") else "direct"
    path = save_artifact(
        "radiation_estimator", models[key], FEATURES,
        station=STATION, target="clear_sky_index" if key == "kt" else "solar_rad_mj",
        night_threshold_mj=NIGHT_MJ,
        train_period=[str(tr.dt.min()), str(tr.dt.max())],
        test_period=[TEST_START, TEST_END],
        test_day_mae=best["day_mae"], test_day_nmae_pct=best["day_nmae_pct"],
        train_feature_ranges=feature_ranges(day_tr, FEATURES),
        # 예보 항목만 쓴다는 것이 이 모델의 전제다 — cloud가 아니라 sky를 받는다.
        input_note="기상청 단기예보 항목만 사용(SKY·TMP·REH) + 청천일사량(계산값)",
        candidates=rows,
    )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(MODELS_DIR, "radiation_model_comparison.csv"), index=False)
    print(f"저장: {path}\n      {MODELS_DIR}/radiation_model_comparison.csv")
    return out


if __name__ == "__main__":
    main()
