"""롱 포맷 지역별 발전량 파일의 '거래시간' 규약을 겹치는 구간으로 실증 검증한다.

왜 필요한가:
  공공데이터 문서에 거래시간이 1~24시인지 0~23시인지 명시가 없다. 이 프로젝트는 과거에
  24시 정렬 버그로 D일 24시가 D일 00:00에 들어가 하루가 통째로 어긋난 적이 있다
  (README '2026-09-23 수정 이력'). 같은 실수를 반복하지 않도록, 기존 와이드 파일과
  겹치는 구간에서 시프트를 -2~+2시간 바꿔가며 상관계수와 오차를 비교한다.

판정:
  시프트 0에서 상관계수가 가장 높고 오차가 가장 작아야 한다. 다른 시프트가 이기면
  data_prep._hour_to_end_of_hour의 해석이 틀린 것이므로 고쳐야 한다.

실행: python -m scripts.verify_generation_overlap
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data_prep import _read_generation_long, _read_generation_wide


def verify(energy_type: str) -> None:
    wide = _read_generation_wide(energy_type).rename(columns={"generation_mwh": "old"})
    long = _read_generation_long(energy_type).rename(columns={"generation_mwh": "new"})
    label = {"solar": "태양광", "wind": "풍력"}[energy_type]

    if long.empty:
        print(f"[{label}] 롱 포맷 파일 없음 — 새 데이터셋을 data/raw에 넣은 뒤 다시 실행하세요.")
        return

    lo = max(wide["dt"].min(), long["dt"].min())
    hi = min(wide["dt"].max(), long["dt"].max())
    if lo >= hi:
        print(f"[{label}] 겹치는 구간이 없어 검증 불가 "
              f"(기존 ~{wide['dt'].max():%Y-%m}, 신규 {long['dt'].min():%Y-%m}~). "
              f"시간 규약을 확인할 방법이 없으니 며칠치라도 겹치는 파일을 받는 것을 권합니다.")
        return

    print(f"[{label}] 겹치는 구간 {lo:%Y-%m-%d} ~ {hi:%Y-%m-%d}")
    w = wide[(wide["dt"] >= lo) & (wide["dt"] <= hi)]
    best = None
    for shift in (-2, -1, 0, 1, 2):
        s = long.copy()
        s["dt"] = s["dt"] + pd.Timedelta(hours=shift)
        m = w.merge(s, on="dt", how="inner").dropna()
        if len(m) < 24:
            continue
        corr = float(np.corrcoef(m["old"], m["new"])[0, 1])
        mae = float(np.abs(m["old"] - m["new"]).mean())
        denom = float(np.abs(m["old"]).mean()) or 1.0
        mark = " <-- 현재 해석" if shift == 0 else ""
        print(f"   시프트 {shift:+d}시간: n={len(m):>6}  corr={corr:.4f}  "
              f"MAE={mae:8.2f}MWh ({mae / denom:.1%}){mark}")
        if best is None or corr > best[1]:
            best = (shift, corr)

    if best is None:
        print("   겹치는 행이 부족해 판정 불가")
    elif best[0] == 0:
        print(f"   ✅ 판정: 시프트 0이 최적(corr={best[1]:.4f}) — 현재 해석이 맞습니다.")
    else:
        print(f"   ❌ 판정: 시프트 {best[0]:+d}시간이 더 잘 맞습니다(corr={best[1]:.4f}). "
              f"data_prep._hour_to_end_of_hour를 수정하세요.")


if __name__ == "__main__":
    for et in ("solar", "wind"):
        verify(et)
        print()
