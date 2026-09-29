"""신제도(2024-06~) 구간에서 모델 성능을 대리 라벨로 측정한다. [신규]

왜 필요한가:
  제주는 2024-06-01 재생에너지 입찰제도 전환 이후 출력제어 실적을 공개하지 않는다. 그래서
  "2026년 예측 정확도를 측정할 방법이 없다"가 이 프로젝트의 최우선 한계였다. 재생에너지 입찰
  상한이 0원/kWh이므로 하루전 SMP <= 0은 '재생에너지가 낙찰되지 못한 시간' = 시장 기반
  출력제어의 대리 지표가 된다. 이 스크립트는 두 단계로 그 한계를 부분적으로 해소한다.

  1단계) 대리 라벨 자체를 검증한다 — 구 라벨과 겹치는 2024-03~05 구간에서 정밀도·재현율.
  2단계) 그 라벨로 신제도 구간의 모델 성능을 측정한다.

한계(반드시 함께 읽을 것):
  - SMP <= 0은 출력제어의 부분집합이다. 시장 가격은 연료비·정비·연계선에도 반응한다.
    따라서 결과를 '출력제어 예측 정확도'가 아니라 '시장 잉여 신호와의 일치도'로 표현해야 한다.
  - 전환 직후 2024-06~12는 시장 안정화 구간으로 SMP<=0이 거의 사라진다(6~9월 0.0~0.7%).
    평가는 2025년 이후를 기준으로 본다.
  - 풍력은 발전량 집계가 2024-06에 끊겨(GENERATION_VALID_END) 서비스 경로를 돌릴 수 없다.
    수요 데이터도 2024-07에서 끝나 침투율 포함 모델은 평가 대상이 아니다.
    따라서 현재 평가 가능한 것은 '수요 미포함 태양광 모델' 하나다.

선행: train_converter, train_classifier, scripts/fetch_smp
실행: python -m app.training.evaluate_new_regime
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.data_prep import (SMP_SURPLUS_THRESHOLD, add_time_features, load_asos, load_curtailment,
                           load_surplus_proxy_label)
from app.metrics import classification_report
from app.model_io import MODELS_DIR, converter_predict_mwh, load_artifact

REGIME_START = "2024-06-01"
WINDOWS = [("2024-06-01", "2025-01-01", "2024-06~12 (전환 직후)"),
           ("2025-01-01", "2026-01-01", "2025 전체"),
           ("2026-01-01", "2027-01-01", "2026 (~최신)"),
           ("2025-01-01", "2027-01-01", "2025 이후 (대표값)")]


def validate_proxy() -> list[dict]:
    """1단계 — 구 라벨과 겹치는 구간에서 대리 라벨의 정밀도·재현율."""
    smp = load_surplus_proxy_label()
    rows = []
    print("[1단계] 대리 라벨 검증 — 구 출력제어 라벨과 겹치는 구간")
    print(f"  {'발전원':<6} {'겹침':>6} {'실제제어':>7} {'라벨양성':>7} {'정밀도':>7} {'재현율':>7} {'F1':>6} {'−SMP AUC':>9}")
    for et, label in (("wind", "풍력"), ("solar", "태양광")):
        m = load_curtailment(et).merge(smp, on="dt", how="inner")
        if m.empty:
            continue
        y = m["is_curtailed"].astype(bool).to_numpy()
        z = m["is_surplus"].astype(bool).to_numpy()
        tp, fp, fn = int((y & z).sum()), int((~y & z).sum()), int((y & ~z).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        auc = roc_auc_score(y, -m["smp"]) if y.any() and (~y).any() else float("nan")
        rows.append({"energy_type": et, "n_overlap": len(m), "n_curtailed": int(y.sum()),
                     "n_proxy_pos": int(z.sum()), "precision": round(pr, 4),
                     "recall": round(rc, 4), "f1": round(f1, 4), "neg_smp_auc": round(auc, 4),
                     "period": f"{m['dt'].min():%Y-%m-%d}~{m['dt'].max():%Y-%m-%d}"})
        print(f"  {label:<6} {len(m):>6} {int(y.sum()):>7} {int(z.sum()):>7} "
              f"{pr:>7.3f} {rc:>7.3f} {f1:>6.3f} {auc:>9.4f}")
    print(f"  (대리 라벨 정의: 하루전 SMP <= {SMP_SURPLUS_THRESHOLD:g}원/kWh)")
    return rows


def evaluate_solar() -> list[dict]:
    """2단계 — 서비스 경로(ASOS -> 컨버터 -> 이용률 -> 분류기)로 신제도 구간 평가."""
    conv = load_artifact("converter_solar")
    clf = load_artifact("classifier_solar_calibrated_sigmoid")
    proxy = float((clf.get("meta") or {})["capacity_proxy_mwh"])

    w = add_time_features(load_asos("184")).dropna(subset=conv["features"])
    w["generation_mwh"] = converter_predict_mwh(conv, w)
    w["capacity_factor"] = w["generation_mwh"] / proxy
    df = w.merge(load_surplus_proxy_label(), on="dt", how="inner")

    rows = []
    print("\n[2단계] 태양광 수요미포함 모델의 신제도 성능 (서비스 경로)")
    print(f"  {'구간':<22} {'n':>6} {'양성':>5} {'양성율':>7} {'AUC':>7} {'PR-AUC':>7} {'top5':>7} {'prec@5%':>8}")
    for lo, hi, label in WINDOWS:
        d = df[(df["dt"] >= lo) & (df["dt"] < hi)]
        if len(d) < 500 or d["is_surplus"].sum() < 10:
            continue
        rep = classification_report(d["is_surplus"].to_numpy(),
                                    clf["model"].predict_proba(d[clf["features"]])[:, 1])
        rows.append({"window": label, "start": lo, "end": hi, **rep})
        print(f"  {label:<22} {rep['n']:>6} {rep['n_pos']:>5} {rep['pos_rate']:>7.2%} "
              f"{rep['auc']:>7.4f} {rep['pr_auc']:>7.4f} {rep['top5_capture']:>7.4f} "
              f"{rep['top5_precision']:>8.4f}")
    print("  비교) 구 제도 2023년 실제 라벨 기준 동일 모델: AUC 0.9809 PR-AUC 0.6386 top5 0.8085")
    print("  ※ 2024-06~12은 전환 직후 시장 안정화 구간이라 대표값에서 제외한다.")
    return rows


def monthly_surplus_rate() -> pd.DataFrame:
    smp = load_surplus_proxy_label()
    smp["ym"] = smp["dt"].dt.to_period("M")
    g = smp.groupby("ym").agg(n=("smp", "size"), surplus_rate=("is_surplus", "mean"),
                              n_negative=("smp", lambda s: int((s < 0).sum())))
    print("\n[참고] 월별 잉여 발생률 (하루전 SMP <= 0)")
    print("  " + "  ".join(f"{str(k)[2:]}:{v:.1%}" for k, v in g["surplus_rate"].items()))
    return g.reset_index().astype({"ym": str})


if __name__ == "__main__":
    pv = validate_proxy()
    ev = evaluate_solar()
    mg = monthly_surplus_rate()
    pd.DataFrame(pv).to_csv(os.path.join(MODELS_DIR, "new_regime_proxy_validation.csv"), index=False)
    pd.DataFrame(ev).to_csv(os.path.join(MODELS_DIR, "new_regime_eval.csv"), index=False)
    mg.to_csv(os.path.join(MODELS_DIR, "new_regime_monthly_surplus.csv"), index=False)
    print(f"\n저장: {MODELS_DIR}/new_regime_{{proxy_validation,eval,monthly_surplus}}.csv")
