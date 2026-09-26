"""나이브 기준선 대비 모델 기여도 — "모델이 정말 필요한가"에 대한 답.

[왜 필요한가]
예측 문헌의 표준 관행이다. 재생에너지 예측 리뷰들은 persistence(직전값)와 climatology(기후값)
기준선을 항상 함께 보고하라고 요구한다 — 하루전 예측 논문은 통상 persistence·climatology·
reference 세 가지 결정론적 기준선과 비교한다. 이 저장소에는 그 표가 없었고, 그래서
"AUC 0.988이 대단한 것인가, 밤에 제어가 없다는 것을 맞힌 것인가"에 답할 근거가 없었다.

[기준선 5종]
  P  persistence      전일 같은 시각의 제어 여부. 문헌의 표준 기준선.
  C  climatology      학습 구간의 (시각 x 월)별 제어 발생률. 기상 정보를 전혀 쓰지 않는다.
  B1 이용률 단독       발전량/설비용량대리. 학습 없이 점수로 그대로 사용.
  B2 침투율 단독       발전량/수요. 출력제어의 물리적 구동 요인 그 자체.
  B3 합산 침투율 단독   (자기 발전량 + 타 발전원)/수요. 학습 없는 버전의 total_penetration.

[읽는 방법]
기준선은 확률이 아니라 원시 점수다. 그래서 순위 지표(AUC·PR-AUC·top5)만 비교한다.
Brier를 내기 위해 min-max로 [0,1]에 옮기지만 단조 변환이라 순위 지표는 바뀌지 않는다.

공정성: 각 모델은 자기가 실제로 쓸 수 있는 입력으로 만든 기준선과 비교해야 한다.
수요를 받지 않는 모델을 침투율 기준선과 비교하면 불공정하다(침투율은 수요가 필요하다).

실행: python -m app.training.evaluate_baselines
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from app.metrics import classification_report
from app.model_io import MODELS_DIR, load_artifact
from app.training.train_classifier import TEST_END, TEST_START, TRAIN_END, build_dataset

# (발전원, 수요 사용, 교차 피처) -> 서빙 아티팩트 이름과 '공정한' 기준선 목록
SERVED = (
    ("solar", False, None,        "classifier_solar_calibrated_sigmoid",
     ("persistence", "climatology", "이용률 단독")),
    ("solar", True,  None,        "classifier_solar_demand_calibrated_sigmoid",
     ("persistence", "climatology", "이용률 단독", "침투율 단독")),
    ("wind",  False, None,        "classifier_wind_calibrated_sigmoid",
     ("persistence", "climatology", "이용률 단독")),
    ("wind",  True,  None,        "classifier_wind_demand_calibrated_sigmoid",
     ("persistence", "climatology", "이용률 단독", "침투율 단독")),
    ("wind",  True,  "converter", "classifier_wind_demand_crossp_calibrated_sigmoid",
     ("persistence", "climatology", "이용률 단독", "침투율 단독", "합산 침투율 단독")),
)

RANK_KEYS = ("auc", "pr_auc", "top5_capture")


def _scores(train: pd.DataFrame, test: pd.DataFrame, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "persistence" in names:
        # 전일 같은 시각의 제어 여부. dt로 정확히 24시간 전을 찾는다(행 순서에 의존하지 않는다).
        prev = pd.concat([train, test])[["dt", "is_curtailed"]].copy()
        prev["dt"] = prev["dt"] + pd.Timedelta(hours=24)
        out["persistence"] = test[["dt"]].merge(prev, on="dt", how="left")["is_curtailed"] \
            .fillna(False).astype(float).to_numpy()
    if "climatology" in names:
        clim = train.groupby([train.dt.dt.hour, train.dt.dt.month])["is_curtailed"].mean()
        out["climatology"] = pd.MultiIndex.from_arrays(
            [test.dt.dt.hour, test.dt.dt.month]).map(clim).to_numpy(dtype=float)
    for key, col in (("이용률 단독", "capacity_factor"), ("침투율 단독", "penetration"),
                     ("합산 침투율 단독", "total_penetration")):
        if key in names and col in test.columns:
            out[key] = test[col].to_numpy(dtype=float)
    return out


def main() -> pd.DataFrame:
    rows = []
    for energy_type, use_demand, cross, art_name, baselines in SERVED:
        df, _ = build_dataset(energy_type, use_demand, cross)
        train = df[df.dt < TRAIN_END]
        test = df[(df.dt >= TEST_START) & (df.dt < TEST_END)]
        y = test["is_curtailed"].to_numpy()

        art = load_artifact(art_name)
        scored = {"모델(서빙)": art["model"].predict_proba(test[art["features"]])[:, 1]}
        scored.update(_scores(train, test, baselines))

        model_rep = None
        for label, s in scored.items():
            s = pd.Series(s).fillna(0.0).to_numpy(dtype=float)
            if s.min() < 0 or s.max() > 1:  # 단조 변환 — 순위 지표는 불변
                s = (s - s.min()) / ((s.max() - s.min()) or 1.0)
            rep = classification_report(y, s)
            row = {"model": art_name, "score": label, "n": len(y), "n_pos": int(y.sum()),
                   **{k: rep[k] for k in RANK_KEYS}, "brier": rep["brier"]}
            if label == "모델(서빙)":
                model_rep = rep
            else:
                for k in RANK_KEYS:
                    base = rep[k]
                    row[f"{k}_model_gain_pct"] = (
                        round((model_rep[k] - base) / base * 100, 1) if base else None)
            rows.append(row)

    out = pd.DataFrame(rows)
    path = os.path.join(MODELS_DIR, "baseline_comparison.csv")
    out.to_csv(path, index=False)

    for art_name, g in out.groupby("model", sort=False):
        print(f"\n[{art_name}] n={g.n.iloc[0]} 양성={g.n_pos.iloc[0]}")
        print(f"  {'점수':<18} {'AUC':>7} {'PR-AUC':>8} {'top5':>7}   모델 이득(PR-AUC / top5)")
        for _, r in g.iterrows():
            gain = ("  —" if r.score == "모델(서빙)" else
                    f"  {r.pr_auc_model_gain_pct:+.1f}% / {r.top5_capture_model_gain_pct:+.1f}%")
            print(f"  {r.score:<18} {r.auc:7.4f} {r.pr_auc:8.4f} {r.top5_capture:7.4f}{gain}")
    print(f"\n저장: {path}")
    return out


if __name__ == "__main__":
    main()
