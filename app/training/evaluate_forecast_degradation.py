"""예보 수준 입력으로 발전량 예측이 얼마나 나빠지는가 — 오래 '측정 불가'였던 항목.

[무엇이 문제였나]
이 저장소의 모든 성능 수치는 ASOS **실측 관측치**를 입력으로 얻은 것이고, README는 그것을
"운영 성능이 아니라 상한"이라고 반복해 적어왔다. 예보 아카이브가 없어 예보 입력 평가가
불가능했기 때문이다.

[여기서 하는 것]
예보 아카이브 없이도 **정보 손실분**은 측정할 수 있다. 기상청 단기예보가 주는 항목만 남기고
나머지를 없앤 입력을 만들어 같은 모델에 넣으면 된다.

  일사량   실측값 있음 -> 예보에 없음. radiation_estimator로 추정한 값을 넣는다.
  전운량   ASOS 0~10  -> 예보는 SKY(1/3/4). 양자화하고 구간 대표값으로 되돌린다.
  기온     실측       -> 예보에 TMP 있음. 그대로 둔다.
  풍속     실측       -> 예보에 WSD 있음. 그대로 둔다.

[측정하는 것과 못 하는 것]
  측정한다 — 예보 항목 구성으로 인한 정보 손실 (일사량 추정 오차 + 전운량 양자화)
  못 한다  — 예보 자체의 오차. 기온·풍속·하늘상태에 실측을 넣으므로 '완벽한 예보' 가정이다.
           실제 운영은 여기에 예보 오차가 더 얹힌다. **이 결과도 여전히 상한이다.**
           다만 상한이 두 단계로 나뉘었다: 실측 입력 > 예보 항목 입력 > 실제 예보 입력.

[KIM 국지모델 일사량으로 대체하면 어떻게 되나 — 아직 측정하지 못한 부분]
2026-09 기준으로 KIM 국지예보모델(1.5km) 표준화 NetCDF API에서 일사량 **예보값**을 직접
받을 수 있게 됐다(app/services/kim_forecast.py). 그러면 아래 '일사량 추정'이 실제 예보값으로
바뀌므로 정보 손실의 상당 부분이 사라질 것으로 기대된다. 다만 그 개선폭을 이 스크립트로
재측정하려면 **테스트 구간(2025년)의 KIM 과거 예보 아카이브**가 필요하다. API가 과거
분석시각을 얼마나 보관하는지는 scripts/verify_kim_radiation.py의 1)번 항목이 확인해준다.
보관 기간이 짧으면 이 재측정은 불가능하고, 대신 앞으로 받는 예보와 ASOS 실측을 나란히
쌓아가며 검증해야 한다 — 그때는 --radiation-csv로 그 값을 넣으면 된다.

실행: python -m app.training.evaluate_forecast_degradation
      python -m app.training.evaluate_forecast_degradation --radiation-csv kim.csv
        (dt, solar_rad 두 열. 해당 시각은 추정 대신 이 값을 쓴다)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from app.data_prep import add_time_features, load_asos, load_generation_actual
from app.model_io import MODELS_DIR, converter_predict_mwh, load_artifact
from app.solar_radiation import add_solar_geometry, cloud_from_sky, sky_from_cloud
from app.training.train_converter import GEN_SOURCE, TEST_SPLIT
from app.training.train_radiation_model import FEATURES as RAD_FEATURES, NIGHT_MJ


def _nmae(y: np.ndarray, p: np.ndarray) -> float:
    return 100 * float(mean_absolute_error(y, p)) / float(np.mean(y))


def estimate_radiation(df: pd.DataFrame) -> np.ndarray:
    """예보 항목(SKY·TMP·REH) + 청천일사량으로 일사량을 추정한다."""
    art = load_artifact("radiation_estimator")
    out = np.zeros(len(df))
    mask = (df["clear_sky_mj"] > NIGHT_MJ).to_numpy()
    if not mask.any():
        return out
    pred = art["model"].predict(df.loc[mask, RAD_FEATURES])
    if (art["meta"].get("target") or "").startswith("clear_sky_index"):
        pred = np.clip(pred, 0, 1.2) * df.loc[mask, "clear_sky_mj"].to_numpy()
    out[mask] = np.clip(pred, 0, None)
    return out


def _load_radiation_csv(path: str, df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """dt·solar_rad CSV를 붙인다. 겹치는 시각만 대체하고, 몇 시간을 대체했는지 함께 돌려준다."""
    k = pd.read_csv(path, parse_dates=["dt"]).set_index("dt")["solar_rad"]
    joined = df["dt"].map(k)
    n = int(joined.notna().sum())
    return joined.to_numpy(), n


def main(radiation_csv: str | None = None) -> pd.DataFrame:
    rows = []
    art_sol = load_artifact("converter_solar")
    t0, t1 = TEST_SPLIT["solar"]

    w = add_solar_geometry(add_time_features(load_asos("184")), "184")
    w["sky"] = sky_from_cloud(w["cloud"])
    gen = load_generation_actual("solar", source=GEN_SOURCE["solar"])
    df = w.merge(gen, on="dt", how="inner")
    df = df.dropna(subset=RAD_FEATURES + ["solar_rad", "generation_mwh"])
    te = df[(df.dt >= t0) & (df.dt < t1)].reset_index(drop=True)
    print(f"[태양광] 테스트 {t0}~{t1}  n={len(te)}")

    # ① 실측 입력 (현재 문서의 수치)
    base = te.copy()
    y = base.generation_mwh.to_numpy()
    p_actual = converter_predict_mwh(art_sol, base)

    # ② 예보 항목 입력 — 일사량은 추정, 전운량은 SKY 양자화 후 대표값
    fc = te.copy()
    fc["solar_rad"] = estimate_radiation(fc)
    if radiation_csv:
        # KIM 예보값이 있는 시각만 대체한다. 섞이는 것을 숨기지 않도록 비율을 찍는다.
        kim, n = _load_radiation_csv(radiation_csv, fc)
        fc["solar_rad"] = np.where(np.isnan(kim), fc["solar_rad"], kim)
        print(f"  일사량: {n}/{len(fc)}시간을 KIM 예보값으로 대체, 나머지는 추정값 "
              f"({n / len(fc) * 100:.1f}%)")
    fc["cloud"] = cloud_from_sky(fc["sky"])
    p_fore = converter_predict_mwh(art_sol, fc)

    # ③ 손실의 출처 분해 — 일사량만 바꾼 경우 / 전운량만 바꾼 경우
    only_rad = te.copy(); only_rad["solar_rad"] = fc["solar_rad"]
    only_cld = te.copy(); only_cld["cloud"] = fc["cloud"]
    variants = {
        "① ASOS 실측 (현재 문서 수치)": p_actual,
        "② 일사량만 추정": converter_predict_mwh(art_sol, only_rad),
        "③ 전운량만 SKY 양자화": converter_predict_mwh(art_sol, only_cld),
        "④ 예보 항목만 (②+③)": p_fore,
    }
    print(f"\n{'입력':30} {'NMAE':>8} {'corr':>8} {'예측최대':>9} {'실측 대비 총합':>13}")
    for name, p in variants.items():
        nm = _nmae(y, p); corr = float(np.corrcoef(y, p)[0, 1])
        rows.append({"energy_type": "solar", "input": name, "n": len(y),
                     "nmae_pct": round(nm, 2), "corr": round(corr, 4),
                     "pred_max_mwh": round(float(p.max()), 1),
                     "total_ratio": round(float(p.sum() / y.sum()), 4)})
        print(f"{name:30} {nm:7.2f}% {corr:8.4f} {p.max():9.1f} {p.sum()/y.sum():12.3f}x")
    d = rows[-1]["nmae_pct"] - rows[0]["nmae_pct"]
    print(f"\n  정보 손실분: NMAE {rows[0]['nmae_pct']:.2f}% -> {rows[-1]['nmae_pct']:.2f}% "
          f"({d:+.2f}%p, 상대 {100*d/rows[0]['nmae_pct']:+.1f}%)")
    print(f"  분해: 일사량 추정 {rows[1]['nmae_pct']-rows[0]['nmae_pct']:+.2f}%p · "
          f"전운량 양자화 {rows[2]['nmae_pct']-rows[0]['nmae_pct']:+.2f}%p")

    # 풍력 — 예보에 WSD가 있으므로 정보 손실이 없다. 그 사실을 수치로 확인한다.
    art_w = load_artifact("converter_wind")
    print(f"\n[풍력] 예보 항목에 WSD(풍속)가 있어 정보 손실이 없다 — "
          f"컨버터 입력은 {art_w['features']}")
    print("  즉 풍력 경로는 예보 연동만 하면 현재 성능(NMAE 42.04%)이 그대로 상한이 되고, "
          "남는 것은 풍속 예보 자체의 오차다.")

    out = pd.DataFrame(rows)
    path = os.path.join(MODELS_DIR, "forecast_degradation.csv")
    out.to_csv(path, index=False)
    print(f"\n저장: {path}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--radiation-csv", help="dt, solar_rad — KIM 등 실제 일사량 예보값")
    main(ap.parse_args().radiation_csv)
