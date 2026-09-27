"""stage2(조건부 제어량) 진단 — 과소분산이 손실함수 문제인지 정보 문제인지 가른다.

[왜 필요한가]
기댓값 방식의 2023 총합이 -37.6%다. 확률 합(Sum p)은 518로 실제 563시간의 92%까지 정직해졌으니
(README '제어량 2단계 추정') 남은 오차는 전부 조건부 제어량에 있다. 조건부 평균 예측이
32.4MWh인데 실측은 46.5MWh이고 예측 표준편차는 실측의 0.38배다 — 중앙으로 수축했다.

오른쪽으로 긴 꼬리(왜도 1.13)를 제곱오차로 학습하면 수축이 생기는 것은 교과서적 현상이고,
처방도 교과서에 있다: 로그 링크, Gamma/Tweedie 손실. README도 '로그변환 타깃'을 시도하지 않은
아이디어로 적어뒀다. 그래서 실제로 붙여본다.

**결론을 먼저 적는다: 전부 현행보다 나쁘다.** 과소분산은 손실함수 선택의 문제가 아니라
정보의 문제다 — 피처에 제어량의 크기를 설명할 항이 없다. 조건부 회귀의 R2가 0.068이고
상수(평균) 예측 대비 MAE 개선이 18.6%뿐인 것이 그 증거다. 이것은 실패 기록이 아니라
"모델링으로는 더 갈 수 없고 계통 제약 데이터가 필요하다"는 주장의 근거다
(README '출력제어는 이렇게 결정된다 — 우리 모델에 없는 항').

실행: python -m app.training.diagnose_stage2
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.model_io import MODELS_DIR
from app.training.train_curtailment_regressor import (FEATURES, TEST_END, TEST_START, _with_gen,
                                                      build_dataset)

TRAIN_END = "2023-01-01"
GEN_COL = "generation_pred"  # 배포 stage2는 컨버터 입력으로 학습한다


def _rf():
    return RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)


def _hgb(**kw):
    return HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400,
                                         min_samples_leaf=20, random_state=42, **kw)


def main() -> pd.DataFrame:
    df = build_dataset()
    tr = df[df.dt < TRAIN_END]
    te = df[(df.dt >= TEST_START) & (df.dt < TEST_END)]
    trc, tec = tr[tr.curtailment_mwh > 0], te[te.curtailment_mwh > 0]
    xtr, xte = _with_gen(trc, GEN_COL), _with_gen(tec, GEN_COL)
    ytr, yte = trc.curtailment_mwh.to_numpy(), tec.curtailment_mwh.to_numpy()

    print(f"학습 제어시간 {len(ytr)} · 테스트 제어시간 {len(yte)} · 피처 {FEATURES}")
    print(f"타깃: 학습 평균 {ytr.mean():.1f} σ {ytr.std():.1f} p95 {np.percentile(ytr, 95):.0f} "
          f"최대 {ytr.max():.0f} 왜도 {pd.Series(ytr).skew():.2f}")
    print(f"      테스트 평균 {yte.mean():.1f} σ {yte.std():.1f} p95 {np.percentile(yte, 95):.0f}\n")

    preds: dict[str, np.ndarray] = {}
    m = _rf(); m.fit(xtr[FEATURES], ytr)
    preds["RF 제곱오차 (현행 배포)"] = np.clip(m.predict(xte[FEATURES]), 0, None)

    mlog = _rf(); mlog.fit(xtr[FEATURES], np.log1p(ytr))
    preds["RF 로그타깃 (expm1)"] = np.clip(np.expm1(mlog.predict(xte[FEATURES])), 0, None)
    # 로그 역변환은 편향을 만든다 — E[Y] = exp(mu + s^2/2). 잔차분산은 학습셋에서만 추정한다.
    s2 = float((np.log1p(ytr) - mlog.predict(xtr[FEATURES])).var())
    preds[f"RF 로그타깃 + 분산보정 (s²={s2:.3f})"] = np.clip(
        np.expm1(mlog.predict(xte[FEATURES]) + s2 / 2), 0, None)

    m = _hgb(); m.fit(xtr[FEATURES], ytr)
    preds["HGB 제곱오차"] = np.clip(m.predict(xte[FEATURES]), 0, None)
    m = _hgb(loss="gamma"); m.fit(xtr[FEATURES], np.clip(ytr, 1e-3, None))
    preds["HGB Gamma 손실"] = np.clip(m.predict(xte[FEATURES]), 0, None)
    m = make_pipeline(StandardScaler(), GammaRegressor(alpha=1.0, max_iter=2000))
    m.fit(xtr[FEATURES], np.clip(ytr, 1e-3, None))
    preds["Gamma GLM"] = np.clip(m.predict(xte[FEATURES]), 0, None)
    # 기준선 둘 — 학습 구간의 상수. 조건부 회귀가 이걸 얼마나 이기는지가 핵심 수치다.
    preds["상수 = 학습 평균"] = np.full(len(yte), float(ytr.mean()))
    preds["상수 = 학습 중앙값"] = np.full(len(yte), float(np.median(ytr)))

    rows = []
    const_mae = mean_absolute_error(yte, preds["상수 = 학습 평균"])
    print(f"{'모델':32} {'MAE':>7} {'R²':>8} {'평균':>7} {'σ':>7} {'σ비':>6} {'p95':>7} "
          f"{'상수대비':>8}")
    for name, p in preds.items():
        row = {"model": name, "mae": round(float(mean_absolute_error(yte, p)), 2),
               "r2": round(float(r2_score(yte, p)), 4), "mean": round(float(p.mean()), 1),
               "std": round(float(p.std()), 1),
               "std_ratio_vs_actual": round(float(p.std() / yte.std()), 2),
               "p95": round(float(np.percentile(p, 95)), 1),
               "mae_gain_vs_constant_pct": round(
                   100 * (1 - mean_absolute_error(yte, p) / const_mae), 1)}
        rows.append(row)
        print(f"{name:32} {row['mae']:7.2f} {row['r2']:+8.4f} {row['mean']:7.1f} "
              f"{row['std']:7.1f} {row['std_ratio_vs_actual']:6.2f} {row['p95']:7.1f} "
              f"{row['mae_gain_vs_constant_pct']:+7.1f}%")
    print(f"{'실측':32} {'':>7} {'':>8} {yte.mean():7.1f} {yte.std():7.1f} {1.0:6.2f} "
          f"{np.percentile(yte, 95):7.1f}")

    out = pd.DataFrame(rows)
    path = os.path.join(MODELS_DIR, "stage2_loss_comparison.csv")
    out.to_csv(path, index=False)

    cur = next(r for r in rows if r["model"].startswith("RF 제곱오차"))
    better = [r for r in rows if not r["model"].startswith("상수") and r["mae"] < cur["mae"]]
    print()
    if better:
        print("현행보다 MAE가 낮은 대안: " + ", ".join(r["model"] for r in better))
    else:
        print("✅ 현행(RF 제곱오차)이 모든 대안보다 MAE가 낮다 — 과소분산은 **손실함수 문제가 "
              "아니다.**")
    print(f"   조건부 회귀의 R² {cur['r2']}, 상수(평균) 대비 MAE 개선 "
          f"{cur['mae_gain_vs_constant_pct']}%, 예측 σ가 실측의 {cur['std_ratio_vs_actual']}배.")
    print("   즉 피처가 제어량의 '크기'를 거의 설명하지 못한다. 필요한 것은 다른 손실이 아니라")
    print("   계통 제약(필수운전 구성·연계선 조류·수용한계량) 데이터다 — 전부 비공개다.")
    print(f"\n저장: {path}")
    return out


if __name__ == "__main__":
    main()
