"""의사결정 가치 — 순위 지표가 아니라 '무엇을 할 수 있나'로 성능을 표현한다.

[왜 필요한가]
이 저장소는 성능을 PR-AUC·top5_capture로 보고해왔다. 머신러닝 안에서는 맞는 지표지만,
계통 실무자에게 "PR-AUC 0.790"은 행동으로 번역되지 않는다. 같은 모델을 실제 결정에 놓고
재보면 훨씬 직접적인 문장이 나온다.

  결정: "연간 N시간만 ESS를 대기시킨다면, 어느 시간을 고를 것인가"
  측정: 그 선택이 실제 출력제어량(MWh)을 얼마나 덮는가

이 틀에서는 세 가지를 나란히 놓을 수 있다.
  완전예지  — 제어량을 미리 다 안다고 가정한 상한
  모델      — 우리 예측 확률로 고른 경우
  기후값    — ML 없이 '월×시각 평균 제어율' 표만 보고 고른 경우 (실무자의 현실적 대안)

기후값 대비 차이가 이 시스템이 만드는 가치이고, 완전예지 대비 비율이 남은 개선 여지다.

[한계]
- 제어량(MWh)이 집계된 구간이 필요하다. 제주 풍력은 2024-06 제도 전환으로 집계가 끊겼으므로
  2023년이 마지막 온전한 해다. 태양광은 제어량이 집계되지 않아(플래그만 존재) 시간 수로만 낸다.
- 완전예지는 실현 불가능한 상한이다. '달성률'을 성능으로 읽지 말 것.
- 입력은 ASOS 실측이다. 예보 입력에서는 더 낮아진다(README '예보 입력 정보 손실').

실행: python -m app.training.evaluate_decision_value
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from app.model_io import MODELS_DIR, load_artifact
from app.serving_config import artifact_name
from app.training.train_classifier import TEST_END, TEST_START, TRAIN_END, build_dataset

BUDGETS = (0.02, 0.05, 0.10, 0.20)


def _climatology(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """ML 없는 대안: 학습 구간의 (월, 시각)별 제어율 표. 실무자가 엑셀로 만들 수 있는 것."""
    t = tr.assign(mo=pd.DatetimeIndex(tr.dt).month, hr=pd.DatetimeIndex(tr.dt).hour)
    tbl = t.groupby(["mo", "hr"])["is_curtailed"].mean()
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(te.dt).month, pd.DatetimeIndex(te.dt).hour])
    return idx.map(tbl).to_numpy(dtype=float, na_value=float(tr["is_curtailed"].mean()))


def evaluate(energy_type: str, use_demand: bool = True,
             use_cross: bool | None = None) -> pd.DataFrame:
    use_cross = (energy_type == "wind") if use_cross is None else use_cross
    df, _ = build_dataset(energy_type, use_demand=use_demand,
                          cross_source="converter" if use_cross else None)
    art = load_artifact(artifact_name(energy_type, use_demand, use_cross))
    tr = df[df.dt < TRAIN_END]
    te = df[(df.dt >= TEST_START) & (df.dt < TEST_END)].copy()
    te["p"] = art["model"].predict_proba(te[art["features"]])[:, 1]
    te["clim"] = _climatology(tr, te)

    # 제어량이 있으면 MWh로, 없으면 '제어 시간 수'로 센다 — 태양광은 제어량이 집계되지 않는다
    has_mwh = te["curtailment_mwh"].fillna(0).sum() > 0
    te["w"] = te["curtailment_mwh"].fillna(0.0) if has_mwh else te["is_curtailed"].astype(float)
    unit = "MWh" if has_mwh else "제어시간"
    total = te["w"].sum()

    rows = []
    for b in BUDGETS:
        k = int(round(len(te) * b))
        model = te.nlargest(k, "p")["w"].sum()
        clim = te.nlargest(k, "clim")["w"].sum()
        oracle = te.nlargest(k, "w")["w"].sum()
        rows.append({
            "energy_type": energy_type, "unit": unit, "n_hours": len(te),
            "total": round(total, 1), "budget_pct": b * 100, "budget_hours": k,
            "model_pct": round(model / total * 100, 1),
            "climatology_pct": round(clim / total * 100, 1),
            "oracle_pct": round(oracle / total * 100, 1),
            "model_minus_clim_pp": round((model - clim) / total * 100, 1),
            "model_over_oracle_pct": round(model / oracle * 100, 1),
        })
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    out = pd.concat([evaluate("wind"), evaluate("solar")], ignore_index=True)
    for et, g in out.groupby("energy_type", sort=False):
        u = g["unit"].iloc[0]
        print(f"\n[{'풍력' if et == 'wind' else '태양광'}] 테스트 {TEST_START}~{TEST_END} "
              f"· {g['n_hours'].iloc[0]:,}시간 · 총 {g['total'].iloc[0]:,.0f} {u}")
        print(f"  {'주의 예산':>14} {'모델':>8} {'기후값':>8} {'완전예지':>8} {'모델−기후값':>11} {'달성률':>8}")
        for _, r in g.iterrows():
            print(f"  {r.budget_pct:4.0f}% ({r.budget_hours:>4}h) {r.model_pct:7.1f}% {r.climatology_pct:7.1f}%"
                  f" {r.oracle_pct:7.1f}% {r.model_minus_clim_pp:+10.1f}%p {r.model_over_oracle_pct:7.1f}%")
    path = os.path.join(MODELS_DIR, "decision_value.csv")
    out.to_csv(path, index=False)
    print(f"\n저장: {path}")
    return out


if __name__ == "__main__":
    main()
