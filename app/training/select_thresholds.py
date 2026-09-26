"""경로별 운영 임계값 재선정.

[왜 필요한가]
0.03은 `classifier_wind_demand`의 확률 척도에서 고른 값인데 /predict의 다섯 경로에 공통으로
쓰고 있었다. 실측 입력으로 확인한 2023년 발동률(확률 >= 0.03인 시간의 비율)이 경로마다
3.6%~31.8%로 약 9배 벌어진다 — 풍력 단독 모델은 실제 제어율 6.43%의 4.9배를 경보로 띄운다.
확률 척도는 모델의 피처 구성과 보정 매핑이 함께 만드는 것이므로, 모델이 바뀌면 임계값도
다시 골라야 한다. 이 문서는 '보정 방식을 바꾸면' 다시 고르라고만 적어 두었다.

[리키지] 배포 모델은 기저 <2022-07-01 + 보정 2022-07~2023-01이다. 그 두 구간 어디서 골라도
배포 모델에는 in-sample이고, 2023은 테스트 구간이라 쓸 수 없다. 그래서 한 칸 앞당긴
중첩 분할로 '선정 전용' 모델을 따로 만든다 — train_curtailment_regressor.choose_threshold와
같은 방식이되, 거기서는 stage2 피처로 대리 모델을 만들었던 것을 각 경로의 **실제 피처
구성**으로 바꾼다. 확률 척도는 피처 구성에 딸린 것이므로 대리 피처로 고른 임계값은
그 척도 위에 있지 않다.

  기저 학습    : < 2022-01-01
  보정         : 2022-01-01 ~ 2022-07-01   (배포 모델과 같은 sigmoid)
  임계값 선정   : 2022-07-01 ~ 2023-01-01
  검증(참고용)  : 2023-01-01 ~ 2024-01-01  — 고른 임계값의 발동률을 실제 제어율과 비교

[기준] F1 최대 = 탐지 품질. 총 제어량 일치 기준은 이 저장소에서 실패했다(README (b)).

실행: python -m app.training.select_thresholds
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from app.model_io import MODELS_DIR, load_artifact
from app.training.train_classifier import (SERVED_METHOD, build_dataset, calibrate, make_model)

BASE_END = "2022-01-01"
CALIB_START, CALIB_END = "2022-01-01", "2022-07-01"
SEL_START, SEL_END = "2022-07-01", "2023-01-01"
VAL_START, VAL_END = "2023-01-01", "2024-01-01"

GRID = np.round(np.arange(0.01, 0.96, 0.01), 2)

# 선정 구간 양성이 이보다 적으면 고른 임계값을 신뢰할 수 없다고 표시한다.
# train_classifier가 보정 구간 양성 50건 미만에 경고를 내는 것과 같은 기준이다 —
# 태양광은 2022 하반기 양성이 20건뿐이고, 실제로 F1 격자가 0.43~0.46에서 평평하다
# (임계값이 데이터가 아니라 격자 해상도로 결정된다는 뜻).
MIN_POS_FOR_THRESHOLD = 50

# /predict의 모델 자동 전환 경로 전부 — (발전원, 수요, 교차피처, 서빙 아티팩트명)
PATHS = (
    ("wind", True, "converter", "classifier_wind_demand_crossp_calibrated_sigmoid"),
    ("wind", True, None, "classifier_wind_demand_calibrated_sigmoid"),
    ("wind", False, None, "classifier_wind_calibrated_sigmoid"),
    ("solar", True, None, "classifier_solar_demand_calibrated_sigmoid"),
    ("solar", False, None, "classifier_solar_calibrated_sigmoid"),
)


def _f1_grid(p: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    pos = y.astype(bool)
    for thr in GRID:
        pred = p >= thr
        tp = int((pred & pos).sum())
        if tp == 0:
            continue
        precision = tp / int(pred.sum())
        recall = tp / int(pos.sum())
        rows.append({"threshold": float(thr),
                     "f1": round(2 * precision * recall / (precision + recall), 4),
                     "precision": round(precision, 4), "recall": round(recall, 4),
                     "fire_rate_pct": round(float(pred.mean()) * 100, 2)})
    return rows


def select_one(energy_type: str, use_demand: bool, cross: str | None, artifact: str) -> dict:
    df, features = build_dataset(energy_type, use_demand, cross)
    base_tr = df[df.dt < BASE_END]
    calib = df[(df.dt >= CALIB_START) & (df.dt < CALIB_END)]
    sel = df[(df.dt >= SEL_START) & (df.dt < SEL_END)]
    val = df[(df.dt >= VAL_START) & (df.dt < VAL_END)]

    base = make_model()
    base.fit(base_tr[features], base_tr["is_curtailed"])
    # 배포 모델과 같은 보정 방식이어야 임계값이 같은 확률 척도 위에 놓인다
    model = calibrate(base, calib, features, SERVED_METHOD)

    p_sel = model.predict_proba(sel[features])[:, 1]
    grid = _f1_grid(p_sel, sel["is_curtailed"].to_numpy())
    if not grid:
        raise RuntimeError(f"{artifact}: 선정 구간에 양성이 없어 임계값을 고를 수 없습니다")
    best = max(grid, key=lambda r: r["f1"])

    # 검증: 고른 임계값을 2023에 적용해 발동률을 실제 제어율과 비교한다.
    # 선정용 모델과 '배포' 모델 둘 다에서 본다 — 두 값이 벌어지면 임계값을 쓸 수 없다는 뜻이다.
    # 배포 모델은 기저 학습이 6개월 더 많고 보정 구간도 달라 확률 척도가 다르다
    # (2023 풍력 crossp 기준 중앙값 0.0095 vs 0.0166, 1.75배).
    # 2023을 '고르는 데'는 쓰지 않고 '진단하는 데'만 쓴다 — 이 저장소가 2023 수치를 보고하는
    # 방식과 같다.
    base_rate = float(val["is_curtailed"].mean()) * 100
    p_val = model.predict_proba(val[features])[:, 1]
    fire = float((p_val >= best["threshold"]).mean()) * 100
    dep = load_artifact(artifact)
    p_dep = dep["model"].predict_proba(val[dep["features"]])[:, 1]
    fire_dep = float((p_dep >= best["threshold"]).mean()) * 100

    n_pos = int(sel["is_curtailed"].sum())
    # 평탄 구간 길이 — F1이 최대와 같은 임계값의 개수. 길면 데이터가 임계값을 정하지 못한 것이다.
    plateau = sum(1 for r in grid if r["f1"] == best["f1"])
    # 신뢰 조건 세 가지를 모두 넘어야 한다. 배포 모델 발동률이 실제 제어율의 2배를 넘으면
    # 선정 척도와 배포 척도가 어긋난 것이므로 그 임계값을 경보에 쓸 수 없다.
    over_base = (fire_dep / base_rate) if base_rate else None
    reliable = bool(n_pos >= MIN_POS_FOR_THRESHOLD and plateau <= 5
                    and over_base is not None and over_base <= 2.0)
    return {"artifact": artifact, "energy_type": energy_type, "use_demand": use_demand,
            "cross_source": cross or "", "n_features": len(features),
            "threshold": best["threshold"],
            "reliable": reliable, "f1_plateau_width": plateau,
            "sel_f1": best["f1"], "sel_precision": best["precision"], "sel_recall": best["recall"],
            "sel_fire_rate_pct": best["fire_rate_pct"],
            "n_sel": len(sel), "n_pos_sel": int(sel["is_curtailed"].sum()),
            "val_fire_rate_pct": round(fire, 2), "val_base_rate_pct": round(base_rate, 2),
            "val_fire_over_base": round(fire / base_rate, 2) if base_rate else None,
            "deployed_fire_rate_pct": round(fire_dep, 2),
            "deployed_fire_over_base": round(over_base, 2) if over_base else None,
            "_grid": grid}


def main() -> pd.DataFrame:
    rows = [select_one(*p) for p in PATHS]
    grids = {r["artifact"]: r.pop("_grid") for r in rows}
    out = pd.DataFrame(rows)

    out.to_csv(os.path.join(MODELS_DIR, "operational_thresholds.csv"), index=False)
    with open(os.path.join(MODELS_DIR, "operational_thresholds.json"), "w") as f:
        json.dump({r["artifact"]: {
            "threshold": r["threshold"], "reliable": r["reliable"],
            "n_pos_sel": r["n_pos_sel"], "selection_period": f"{SEL_START}~{SEL_END}",
            "deployed_fire_rate_pct_2023": r["deployed_fire_rate_pct"],
            "base_rate_pct_2023": r["val_base_rate_pct"],
        } for r in rows}, f, indent=2)
    pd.concat([pd.DataFrame(g).assign(artifact=a) for a, g in grids.items()]).to_csv(
        os.path.join(MODELS_DIR, "operational_threshold_grids.csv"), index=False)

    print(f"선정 구간 {SEL_START}~{SEL_END} / 검증 2023\n")
    print(f"{'경로':46} {'임계값':>7} {'F1':>7} {'선정모델':>9} {'배포모델':>9} {'실제':>6} {'신뢰':>5}")
    for r in rows:
        print(f"{r['artifact']:46} {r['threshold']:7.2f} {r['sel_f1']:7.4f} "
              f"{r['val_fire_rate_pct']:8.1f}% {r['deployed_fire_rate_pct']:8.1f}% "
              f"{r['val_base_rate_pct']:5.2f}% {'O' if r['reliable'] else 'X':>5}")
    print("\n  선정모델/배포모델 발동률이 벌어지면 임계값을 쓸 수 없다 — 확률 척도가 다르기 때문이다.")

    old = out[out.threshold != 0.03]
    if len(old):
        print(f"\n0.03과 다른 임계값이 선정된 경로 {len(old)}개 — 공통 임계값을 쓸 수 없다는 뜻이다:")
        for _, r in old.iterrows():
            print(f"  {r.artifact}: {r.threshold}")

    bad = out[~out.reliable]
    if len(bad):
        print(f"\n⚠ 임계값을 신뢰할 수 없는 경로 {len(bad)}개 — 양성 {MIN_POS_FOR_THRESHOLD}건 미만 "
              f"또는 F1 평탄 구간이 넓다. 이 경로는 임계값 대신 등급(상위 5%)으로 표시할 것:")
        for _, r in bad.iterrows():
            print(f"  {r.artifact}: 양성 {r.n_pos_sel}건, F1 평탄폭 {r.f1_plateau_width}, "
                  f"배포 모델 발동률이 실제의 {r.deployed_fire_over_base}배")
    print(f"\n저장: {MODELS_DIR}/operational_thresholds.{{csv,json}}, operational_threshold_grids.csv")
    return out


if __name__ == "__main__":
    main()
