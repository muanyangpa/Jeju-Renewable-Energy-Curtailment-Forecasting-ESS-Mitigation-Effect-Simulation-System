"""주 모델의 인과적 교차검증 + 하이퍼파라미터 탐색 기록.

[기존 블록CV와 무엇이 다른가]
validate_demand_blockcv.py는 leave-block-out이다 — 한 블록을 테스트로 두고 **나머지 전부**를
학습에 쓰므로 미래 블록으로 학습한다. 두 피처 구성을 동일 조건에서 비교하는 ablation에는
타당하지만, **성능 추정으로는 쓸 수 없다**(리키지). 이 스크립트는 rolling-origin(walk-forward)
방식으로, 각 폴드가 실제 재학습 상황과 같은 구조를 갖는다.

  폴드 b:  기저 학습 = b 이전의 보정구간 앞까지 / 보정 = b 직전 블록 / 테스트 = b

배포 모델(기저 <2022-07 + 보정 2022-07~2023-01 + 테스트 2023)과 같은 모양을 시점만 옮겨
반복하는 것이다. 단일 분할 하나로 성능을 주장하던 것을 여러 시점에서 확인한다.

[하이퍼파라미터 탐색]
지금까지 주 분류기의 하이퍼파라미터는 고정값이었고 탐색 기록이 없었다. 2023은 테스트 구간이라
쓸 수 없으므로 **2023 이전 폴드만으로** 고른다. 그 결과 풍력은 2개 폴드, 태양광은 0개다 —
태양광 라벨이 2021-10에 시작하고 블록별 양성이 3~20건뿐이기 때문이다. 없는 것은 없다고 적는다.

실행: python -m app.training.validate_blockcv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from app.metrics import classification_report, day_block_ci
from app.model_io import MODELS_DIR
from app.serving_config import SERVED_FAMILY, features_for, hyperparams, path_key
from app.training.train_classifier import FAMILIES, SERVED_METHOD, build_dataset, calibrate

BLOCKS = [("2021-01-01", "2021-07-01"), ("2021-07-01", "2022-01-01"),
          ("2022-01-01", "2022-07-01"), ("2022-07-01", "2023-01-01"),
          ("2023-01-01", "2023-07-01"), ("2023-07-01", "2024-01-01")]

# 폴드 성립 조건. 보정 매핑은 양성이 아주 적으면 불안정하고(README '태양광 보정 구간의 양성이
# 20건뿐'), 기저 학습도 양성이 없으면 의미가 없다. 테스트 블록도 양성이 있어야 지표가 정의된다.
MIN_POS_BASE, MIN_POS_CALIB, MIN_POS_TEST = 50, 15, 10

PATHS = (("solar", False, False), ("solar", True, False),
         ("wind", False, False), ("wind", True, False), ("wind", True, True))

# 탐색 격자. 현행값(RF max_depth=6/n_estimators=200, LR C=1.0)을 반드시 포함해 '현행이 왜
# 그 값인가'에 답할 수 있게 한다.
GRIDS = {
    "rf": ({"max_depth": 4, "n_estimators": 200}, {"max_depth": 6, "n_estimators": 200},
           {"max_depth": 8, "n_estimators": 200}, {"max_depth": 12, "n_estimators": 200},
           {"max_depth": 6, "n_estimators": 400}, {"max_depth": None, "n_estimators": 200}),
    "lr": ({"C": 0.01}, {"C": 0.1}, {"C": 1.0}, {"C": 10.0}),
}
CURRENT = {"rf": {"max_depth": 6, "n_estimators": 200}, "lr": {"C": 1.0}}


def _set_params(model, fam: str, params: dict):
    """모델군에 맞게 파라미터를 주입한다. lr은 Pipeline이라 스텝 이름을 붙여야 한다."""
    if fam == "lr":
        model.set_params(**{f"logisticregression__{k}": v for k, v in params.items()})
    else:
        model.set_params(**params)
    return model


def folds(df: pd.DataFrame):
    """(테스트 블록, 기저학습 프레임, 보정 프레임, 테스트 프레임) — 성립하는 폴드만 낸다."""
    for i in range(2, len(BLOCKS)):
        cs, ce = BLOCKS[i - 1]
        ts, te = BLOCKS[i]
        base = df[df.dt < cs]
        calib = df[(df.dt >= cs) & (df.dt < ce)]
        test = df[(df.dt >= ts) & (df.dt < te)]
        why = []
        if base["is_curtailed"].sum() < MIN_POS_BASE:
            why.append(f"기저 양성 {int(base['is_curtailed'].sum())}<{MIN_POS_BASE}")
        if calib["is_curtailed"].sum() < MIN_POS_CALIB:
            why.append(f"보정 양성 {int(calib['is_curtailed'].sum())}<{MIN_POS_CALIB}")
        if test["is_curtailed"].sum() < MIN_POS_TEST:
            why.append(f"테스트 양성 {int(test['is_curtailed'].sum())}<{MIN_POS_TEST}")
        yield f"{ts[:7]}~{te[:7]}", base, calib, test, why


def rolling_cv(et: str, ud: bool, uc: bool) -> tuple[list[dict], dict | None]:
    key = path_key(et, ud, uc)
    fam = SERVED_FAMILY[key]
    df, features_all = build_dataset(et, ud, "converter" if uc else None)
    # 서빙과 같은 피처 구성이어야 한다 — 죽은 피처를 포함하면 서빙 모델을 검증한 것이 아니다
    features = features_for(key, features_all)
    rows, pooled_y, pooled_p, pooled_days = [], [], [], []
    print(f"\n[{key}] 서빙 모델군 {fam} · 피처 {len(features)}개")
    for name, base, calib, test, why in folds(df):
        if why:
            print(f"  {name}  건너뜀 — {', '.join(why)}")
            rows.append({"path": key, "family": fam, "fold": name, "skipped": "; ".join(why)})
            continue
        # 서빙 하이퍼파라미터를 쓴다 — 기본값으로 재면 서빙 모델을 검증한 것이 아니다
        m = _set_params(FAMILIES[fam](), fam, {**CURRENT[fam], **hyperparams(key, fam)})
        m.fit(base[features], base["is_curtailed"])
        cal = calibrate(m, calib, features, SERVED_METHOD)
        y = test["is_curtailed"].to_numpy()
        p = cal.predict_proba(test[features])[:, 1]
        rep = classification_report(y, p)
        rows.append({"path": key, "family": fam, "fold": name, "skipped": "",
                     "n_base": len(base), "n_pos_base": int(base["is_curtailed"].sum()),
                     "n_pos_calib": int(calib["is_curtailed"].sum()), **rep})
        pooled_y.append(y); pooled_p.append(p)
        pooled_days.append(test["dt"].dt.date.to_numpy())
        print(f"  {name}  기저 양성 {int(base['is_curtailed'].sum()):3d} 테스트 양성 {rep['n_pos']:3d}  "
              f"AUC={rep['auc']:.4f} PR-AUC={rep['pr_auc']:.4f} top5={rep['top5_capture']:.4f}")

    if not pooled_y:
        print("  성립하는 폴드가 없다 — 라벨 구간이 짧고 양성이 편중돼 있다.")
        return rows, None
    y = np.concatenate(pooled_y); p = np.concatenate(pooled_p)
    days = np.concatenate(pooled_days)
    ci = day_block_ci(y, p, days)
    print(f"  풀링({len(pooled_y)}폴드, 양성 {y.sum()}시간): "
          + " · ".join(f"{k} {v['point']} [{v['lo']}, {v['hi']}]" for k, v in ci.items()))
    per_fold = [r for r in rows if not r.get("skipped")]
    spread = {f"{k}_fold_min": round(min(r[k] for r in per_fold), 4)
              for k in ("auc", "pr_auc", "top5_capture")}
    spread.update({f"{k}_fold_max": round(max(r[k] for r in per_fold), 4)
                   for k in ("auc", "pr_auc", "top5_capture")})
    print("  폴드 간 변동폭: " + " · ".join(
        f"{k} {spread[f'{k}_fold_min']}~{spread[f'{k}_fold_max']}"
        for k in ("auc", "pr_auc", "top5_capture")))
    return rows, {"path": key, "family": fam, "n_folds": len(pooled_y), "n_pos": int(y.sum()),
                  **{f"{k}_{s}": v[s] for k, v in ci.items() for s in ("point", "lo", "hi")},
                  **spread}


def tune(et: str, ud: bool, uc: bool) -> list[dict]:
    """2023을 보지 않고 하이퍼파라미터를 고른다 — 2023 이전 폴드만 쓴다."""
    key = path_key(et, ud, uc)
    fam = SERVED_FAMILY[key]
    df, features_all = build_dataset(et, ud, "converter" if uc else None)
    features = features_for(key, features_all)
    usable = [(n, b, c, t) for n, b, c, t, why in folds(df) if not why and n < "2023"]
    print(f"\n[{key}] {fam} 하이퍼파라미터 탐색 — 2023 이전 폴드 {len(usable)}개")
    if not usable:
        print("  ⚠ 2023 이전에 성립하는 폴드가 없다. 현행 하이퍼파라미터는 **탐색되지 않은 "
              "기본값**이며, 테스트 구간을 건드리지 않고 고를 방법이 현재 데이터로는 없다.")
        return [{"path": key, "family": fam, "params": "N/A", "n_folds": 0,
                 "mean_pr_auc": None, "note": "2023 이전 유효 폴드 없음 — 탐색 불가"}]
    rows = []
    for params in GRIDS[fam]:
        scores = []
        for _, base, calib, test in usable:
            m = _set_params(FAMILIES[fam](), fam, params)
            m.fit(base[features], base["is_curtailed"])
            cal = calibrate(m, calib, features, SERVED_METHOD)
            rep = classification_report(test["is_curtailed"].to_numpy(),
                                        cal.predict_proba(test[features])[:, 1])
            scores.append(rep["pr_auc"])
        row = {"path": key, "family": fam, "params": str(params), "n_folds": len(usable),
               "mean_pr_auc": round(float(np.mean(scores)), 4),
               "min_pr_auc": round(float(np.min(scores)), 4),
               "is_current": params == CURRENT[fam]}
        rows.append(row)
        print(f"  {str(params):44} 평균 PR-AUC {row['mean_pr_auc']:.4f} "
              f"(최소 {row['min_pr_auc']:.4f}){'   <- 현행' if row['is_current'] else ''}")
    best = max(rows, key=lambda r: r["mean_pr_auc"])
    cur = next(r for r in rows if r["is_current"])
    if best["params"] != cur["params"]:
        print(f"  -> 최적 {best['params']} (평균 {best['mean_pr_auc']}) vs "
              f"현행 {cur['params']} (평균 {cur['mean_pr_auc']}) — "
              f"차이 {best['mean_pr_auc'] - cur['mean_pr_auc']:+.4f}")
        print(f"     ⚠ 폴드가 {len(usable)}개뿐이라 이 차이는 근거가 약하다. 교체 전에 폴드를 "
              f"늘릴 수 있는지(라벨 확장) 먼저 볼 것.")
    else:
        print(f"  -> 현행값이 최적이다 (평균 PR-AUC {cur['mean_pr_auc']}).")
    return rows


def main() -> None:
    print("=" * 96)
    print("① rolling-origin 교차검증 (인과적) — 각 폴드가 실제 재학습 상황과 같은 구조")
    print("=" * 96)
    cv_rows, summ = [], []
    for p in PATHS:
        r, s = rolling_cv(*p)
        cv_rows += r
        if s:
            summ.append(s)
    pd.DataFrame(cv_rows).to_csv(os.path.join(MODELS_DIR, "blockcv_rolling_folds.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(MODELS_DIR, "blockcv_rolling_summary.csv"), index=False)

    print("\n" + "=" * 96)
    print("② 하이퍼파라미터 탐색 (2023 미사용)")
    print("=" * 96)
    tune_rows = []
    for p in PATHS:
        tune_rows += tune(*p)
    pd.DataFrame(tune_rows).to_csv(
        os.path.join(MODELS_DIR, "hyperparameter_search.csv"), index=False)
    print(f"\n저장: {MODELS_DIR}/blockcv_rolling_{{folds,summary}}.csv, hyperparameter_search.csv")


if __name__ == "__main__":
    main()
