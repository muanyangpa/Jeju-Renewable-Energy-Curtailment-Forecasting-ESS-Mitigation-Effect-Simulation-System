"""드리프트 경고 로직 — 가짜 아티팩트로 임계 동작을 고정한다. 데이터·모델 불필요.

실측 검증(app/training/verify_serving_real_inputs.py)이 잡은 버그를 회귀로 막는다:
범위를 p1~p99로 잡으면 학습 구간에서 21.2%의 날이 경고를 받았다. 여기서는 '10% 임계가
경계에서 정확히 어떻게 동작하는가'를 고정하고, 실측 발동률은 그 스크립트가 본다.
"""
from __future__ import annotations

import pandas as pd

from app.services.predict import _drift_note

RANGES = {"capacity_factor": [0.0, 1.0], "penetration": [0.0, 0.3],
          "hour_sin": [-1.0, 1.0], "month_cos": [-1.0, 1.0]}


def _clf(ranges=RANGES) -> dict:
    return {"meta": {"train_feature_ranges": ranges} if ranges else {}}


def _df(n_out: int, n: int = 24, col: str = "capacity_factor") -> pd.DataFrame:
    """n시간 중 n_out시간을 범위 밖(2.0)으로 만든 입력."""
    v = [2.0] * n_out + [0.5] * (n - n_out)
    return pd.DataFrame({col: v, "penetration": [0.1] * n,
                         "hour_sin": [0.0] * n, "month_cos": [0.0] * n})


def test_범위_안이면_경고하지_않는다():
    assert _drift_note(_df(0), [('분류기', _clf())]) is None


def test_임계_10퍼센트_미만이면_경고하지_않는다():
    assert _drift_note(_df(2), [('분류기', _clf())]) is None          # 2/24 = 8.3%


def test_임계를_넘으면_경고한다():
    note = _drift_note(_df(3), [('분류기', _clf())])                  # 3/24 = 12.5%
    assert note is not None
    assert "capacity_factor" in note and "학습범위" in note


def test_가장_심한_피처를_보고한다():
    df = _df(3)                                         # capacity_factor 12.5%
    df.loc[:11, "penetration"] = 9.9                    # penetration 50%
    note = _drift_note(df, [('분류기', _clf())])
    assert note is not None and "penetration" in note


def test_시각_피처는_검사하지_않는다():
    """순환값이라 범위를 벗어날 수 없다 — 검사에 넣으면 의미 없는 경고가 뜬다."""
    df = _df(0)
    df["hour_sin"] = 5.0                                # 전부 범위 밖
    df["month_cos"] = 5.0
    assert _drift_note(df, [('분류기', _clf())]) is None


def test_범위_메타데이터가_없으면_조용하다():
    """옛 아티팩트로도 서빙이 죽지 않아야 한다 — 경고만 못 낸다."""
    assert _drift_note(_df(24), [('분류기', _clf(ranges=None))]) is None


def test_아래쪽으로_벗어나도_경고한다():
    df = _df(0)
    df.loc[:5, "capacity_factor"] = -3.0                # 6/24 = 25%
    note = _drift_note(df, [('분류기', _clf())])
    assert note is not None and "capacity_factor" in note


def test_두_단계를_모두_보고한다():
    """컨버터 단계와 분류기 단계가 동시에 벗어나면 둘 다 문구에 들어가야 한다 —
    처방이 다르기 때문이다(컨버터 재학습 vs 분류기 재학습)."""
    df = _df(6)                                          # capacity_factor 25%
    df["wind_speed"] = 45.0
    conv = {"meta": {"train_feature_ranges": {"wind_speed": [0.3, 20.3]}}}
    note = _drift_note(df, [("기상->컨버터", conv), ("파생->분류기", _clf())])
    assert note is not None
    assert "wind_speed" in note and "capacity_factor" in note


def test_한_단계만_벗어나면_그것만_보고한다():
    df = _df(0)
    df["wind_speed"] = 45.0
    conv = {"meta": {"train_feature_ranges": {"wind_speed": [0.3, 20.3]}}}
    note = _drift_note(df, [("기상->컨버터", conv), ("파생->분류기", _clf())])
    assert note is not None
    assert "wind_speed" in note and "capacity_factor" not in note
