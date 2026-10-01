"""밴드 바닥값 — '오늘은 위험 시간 없음'을 표시할 수 있게 하는 상수 하나.

[무엇을 고치는가]
대시보드는 하루 24시간 안에서 순위를 매겨 등급을 만든다. 순위는 정의상 척도를 지우므로
제어가 전혀 없는 날에도 1등은 반드시 존재하고, 그 1등이 '매우 높음'으로 표시된다.
2023년 테스트에서 무제어일 248일 **전부**에 경보가 떴다. 튜닝으로 못 고치는 구조적 성질이다.

[왜 바닥값인가 — 더 나은 방법을 재고 버렸다]
확률 분포의 이동 분위수를 기준으로 쓰면 더 좋다(MWh 커버 19.3% -> 76.8%). 그러나 서비스에
상태가 생기고, 기준 분포가 쌓일 때까지 약 83일이 필요하며, 학습 구간 확률은 표본 내라
과신되어 있어 기준으로 쓸 수 없다(학습 q95=0.978 vs 테스트 0.740).

바닥값은 **상수 하나**라 그 비용이 전부 사라진다. 무상태 유지, 콜드 스타트 없음, 새 실패
모드 없음. 그러면서 개선의 상당 부분을 얻는다(아래 표).

  방식                              발화   정밀도   MWh커버  오경보일
  현재: 일내 상위 1h                 4.2%   28.5%    19.3%    100%
  일내 상위 3h AND p>=바닥값          4.9%   65.0%    54.9%     21%
  (참고) 이동 분위수 q95             5.5%   79.1%    76.8%      7%

[바닥값을 어떻게 고르는가]
'조용해진 날 중 실제 제어일이 몇 일인가'를 기준으로 고른다. 정밀도를 올리려고 바닥을 높이면
제어일을 통째로 놓치기 시작하므로, **놓친 제어일이 허용치를 넘지 않는 가장 높은 값**을 쓴다.
기본 허용치는 2%다 — 제어일을 놓치는 비용(흡수 못 한 MWh)이 헛걸음 비용보다 크기 때문이다.

경로마다 확률 척도가 다르므로(같은 임계값에서 발화율 9배 차이) **경로별로 따로 고른다.**
선정 구간은 학습·보정 구간이고 테스트 구간은 보지 않는다.

실행: python -m app.training.select_band_floor
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from app.model_io import MODELS_DIR, load_artifact
from app.serving_config import PATH_KEYS, artifact_name
from app.training.train_classifier import TEST_START, TEST_END, TRAIN_END, build_dataset

GRID = np.concatenate([np.arange(0.005, 0.10, 0.005), np.arange(0.10, 0.85, 0.02)])
MAX_MISSED_DAY_RATE = 0.02      # 조용해진 날 중 실제 제어일 허용 비율


def _spec(key: str) -> tuple[str, bool, bool]:
    et = "solar" if key.startswith("solar") else "wind"
    return et, "demand" in key, "crossp" in key


def select(key: str) -> dict:
    et, use_demand, use_cross = _spec(key)
    df, _ = build_dataset(et, use_demand=use_demand,
                          cross_source="converter" if use_cross else None)
    art = load_artifact(artifact_name(et, use_demand, use_cross))
    df = df.copy()
    df["p"] = art["model"].predict_proba(df[art["features"]])[:, 1]
    df["day"] = pd.DatetimeIndex(df.dt).normalize()

    sel = df[df.dt < TEST_START]                     # 선정은 학습+보정 구간에서만
    g = sel.groupby("day").agg(pmax=("p", "max"), cur=("is_curtailed", "max"))
    n_cur = int(g["cur"].sum())

    best, best_quiet, rows = 0.0, 0, []
    for f in GRID:
        quiet = g[g["pmax"] < f]
        missed = int(quiet["cur"].sum())
        rate = missed / max(n_cur, 1)
        rows.append({"floor": round(float(f), 4), "quiet_days": len(quiet),
                     "missed_curtail_days": missed, "missed_rate": round(rate, 4)})
        if rate <= MAX_MISSED_DAY_RATE:
            best, best_quiet = float(f), len(quiet)
    # 바닥값이 '조용한 날'을 실질적으로 만들어내지 못하면 이 경로에서는 쓸 수 없다.
    # wind 단독 경로가 그렇다 — PR-AUC 0.391로 애초에 날짜 판별을 못 하므로, 놓침 2% 제약을
    # 지키는 바닥값이 사실상 0에 붙는다. 억지로 올리면 제어일을 통째로 놓친다.
    reliable = best_quiet >= 0.20 * len(g)
    return {"key": key, "floor": round(best, 4), "reliable": bool(reliable),
            "quiet_days": best_quiet, "n_sel_days": len(g),
            "n_curtail_days": n_cur, "grid": pd.DataFrame(rows)}


def main() -> pd.DataFrame:
    out = []
    for key in PATH_KEYS:
        r = select(key)
        et, ud, uc = _spec(key)
        df, _ = build_dataset(et, use_demand=ud, cross_source="converter" if uc else None)
        art = load_artifact(artifact_name(et, ud, uc))
        te = df[(df.dt >= TEST_START) & (df.dt < TEST_END)].copy()
        te["p"] = art["model"].predict_proba(te[art["features"]])[:, 1]
        te["day"] = pd.DatetimeIndex(te.dt).normalize()
        te["rk"] = te.groupby("day")["p"].rank(ascending=False, method="first")
        nc = te.groupby("day")["is_curtailed"].transform("max") == 0
        nd = te[nc]["day"].nunique()

        def m(mask):
            s = te[mask]
            fa = te[nc & mask]["day"].nunique()
            return (len(s) / len(te) * 100,
                    s["is_curtailed"].mean() * 100 if len(s) else 0.0,
                    s["is_curtailed"].sum() / max(te["is_curtailed"].sum(), 1) * 100,
                    fa / max(nd, 1) * 100)
        cur_f, cur_p, cur_r, cur_a = m(te.rk <= 1)
        new_f, new_p, new_r, new_a = m((te.rk <= 3) & (te.p >= r["floor"]))
        out.append({"path": key, "floor": r["floor"], "reliable": r["reliable"],
                    "quiet_days_sel": r["quiet_days"], "sel_days": r["n_sel_days"],
                    "sel_curtail_days": r["n_curtail_days"],
                    "cur_fire_pct": round(cur_f, 1), "cur_prec_pct": round(cur_p, 1),
                    "cur_recall_pct": round(cur_r, 1), "cur_falsealarm_day_pct": round(cur_a, 0),
                    "new_fire_pct": round(new_f, 1), "new_prec_pct": round(new_p, 1),
                    "new_recall_pct": round(new_r, 1), "new_falsealarm_day_pct": round(new_a, 0)})

    t = pd.DataFrame(out)
    print(f"\n{'경로':22} {'바닥값':>7} {'신뢰':>5} | {'현재 정밀/오경보일':>19} | {'바닥+상위3h':>19}")
    for _, r in t.iterrows():
        print(f"  {r.path:20} {r.floor:7.3f} {'O' if r.reliable else 'X':>5} |"
              f" {r.cur_prec_pct:8.1f}% {r.cur_falsealarm_day_pct:7.0f}% |"
              f" {r.new_prec_pct:8.1f}% {r.new_falsealarm_day_pct:7.0f}%")
    if (~t.reliable).any():
        bad = ", ".join(t[~t.reliable].path)
        print(f"  ⚠ 신뢰 불가 경로: {bad} — 조용한 날을 만들지 못한다. 대시보드에서 현재 방식 유지 + 경고 표시")
    path = os.path.join(MODELS_DIR, "band_floors.csv")
    t.to_csv(path, index=False)
    print(f"\n저장: {path}")
    print("  -> app/serving_config.py의 BAND_FLOOR에 반영할 것 (이 스크립트가 근거)")
    return t


if __name__ == "__main__":
    main()
