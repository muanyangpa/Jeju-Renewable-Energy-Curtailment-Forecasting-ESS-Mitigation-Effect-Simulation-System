"""학습 전 데이터 커버리지 점검. 재학습할 때마다 가장 먼저 돌릴 것.

파일명 규칙을 못 맞추면 로더가 조용히 옛 파일만 읽고 넘어간다(에러가 나지 않는다).
그래서 학습 전에 각 소스가 어디까지 들어왔는지 눈으로 확인해야 한다.

실행: python -m scripts.check_data
"""
from __future__ import annotations

import pandas as pd

from app.data_prep import (CURTAILMENT_COVERAGE, DATA_DIR, load_asos, load_curtailment,
                           load_demand_actual, load_generation_actual)

SOURCES = [
    ("발전량 태양광", lambda: load_generation_actual("solar")),
    ("발전량 풍력", lambda: load_generation_actual("wind")),
    ("출력제어 태양광", lambda: load_curtailment("solar")),
    ("출력제어 풍력", lambda: load_curtailment("wind")),
    ("전력수요", load_demand_actual),
    ("ASOS 184(제주)", lambda: load_asos("184")),
    ("ASOS 185(고산)", lambda: load_asos("185")),
    ("ASOS 188(성산)", lambda: load_asos("188")),
]


def main() -> None:
    print(f"DATA_DIR = {DATA_DIR}\n")
    rows = []
    for name, fn in SOURCES:
        try:
            d = fn()
            rows.append({"소스": name, "시작": f"{d['dt'].min():%Y-%m}",
                         "끝": f"{d['dt'].max():%Y-%m}", "행": len(d), "상태": "ok"})
        except Exception as e:
            rows.append({"소스": name, "시작": "-", "끝": "-", "행": 0,
                         "상태": f"실패: {type(e).__name__} {e}"})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    ok = df[df["상태"] == "ok"]
    if len(ok) < len(df):
        print("\n⚠ 읽지 못한 소스가 있습니다 — 파일명 규칙을 확인하세요.")
    elif ok["끝"].nunique() > 1:
        print(f"\n⚠ 소스별 종료 시점이 다릅니다({sorted(ok['끝'].unique())}). "
              f"가장 이른 시점이 학습 구간의 상한이 됩니다.")

    print("\n라벨 유효구간(CURTAILMENT_COVERAGE) — 제도 전환 경계를 넘기지 말 것:")
    for k, v in CURTAILMENT_COVERAGE.items():
        print(f"  {k}: {v[0]} ~ {v[1]}")
    print("  ※ 2024-06-01부터 제주는 재생에너지 입찰제도로 전환돼 라벨 정의가 다르다.")
    print("     그 경계를 넘겨 학습하면 서로 다른 제도의 데이터가 섞인다(README 참고).")


if __name__ == "__main__":
    main()
