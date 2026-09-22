"""
데이터 로딩 및 전처리 공통 유틸리티.

이 프로젝트의 검증된 방법론(출력제어_팀검토대응_심화검증_요약, 출력제어예측_프로젝트계획서 05/07장)을
그대로 재사용한다. 원본 공공데이터 CSV -> 학습/추론용 시계열 DataFrame으로 변환하는 함수들을 모은다.

주의: 이 모듈은 "연구용 검증"에 쓰인 messy 예보 아카이브 파서(블록 파서+UTC보정 등)는 재현하지 않는다.
그 파서는 "과거 시점에 실제로 발표됐던 예보가 있었다면 어땠을까"를 검증하기 위한 1회성 분석 도구였고,
실제 운영에서는 Backend가 기상청 API 등에서 이미 정제된 24개 시간별 기상값을 넘겨주는 구조이기 때문이다
(계획서 02장 참고). 따라서 학습 데이터는 형식이 깨끗한 ASOS 실측 관측자료(종관기상관측)를 사용한다 —
I장에서 이미 "실측 풍속으로 컨버터를 학습시키는" 방식으로 검증된 것과 같은 접근이다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = "/sessions/gracious-dreamy-noether/mnt/uploads"


def _read_csv_any_encoding(path: str, **kwargs) -> pd.DataFrame:
    for enc in ("cp949", "utf-8", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"could not decode {path}")


def _find(pattern_substr: str) -> str:
    """uploads 폴더에서 파일명에 pattern_substr이 포함된 첫 파일 경로를 반환."""
    for f in os.listdir(DATA_DIR):
        if pattern_substr in f:
            return os.path.join(DATA_DIR, f)
    raise FileNotFoundError(f"no file containing '{pattern_substr}' in {DATA_DIR}")


def _melt_hourly(df: pd.DataFrame, date_col: str, value_name: str) -> pd.DataFrame:
    """'1시'~'24시' 와이드 포맷 -> (dt, value) 롱 포맷. 24시는 다음날 0시로 취급."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    hour_cols = [c for c in df.columns if str(c).endswith("시")]
    long = df.melt(id_vars=date_col, value_vars=hour_cols, var_name="hour", value_name=value_name)
    long["hour"] = long["hour"].str.replace("시", "", regex=False).astype(int)
    long["hour"] = long["hour"].replace(24, 0)
    long["dt"] = long[date_col] + pd.to_timedelta(long["hour"], unit="h")
    return long[["dt", value_name]]


# ---------------------------------------------------------------------------
# 발전량 실적 (제주 전체, 태양광/풍력)
# ---------------------------------------------------------------------------

def load_generation_actual(energy_type: str) -> pd.DataFrame:
    """energy_type: 'solar' | 'wind' -> DataFrame[dt, generation_mwh]"""
    fname_substr = "태양광 한시간단위실적" if energy_type == "solar" else "풍력 한시간단위실적"
    path = _find(fname_substr)
    df = _read_csv_any_encoding(path)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = df.columns[0]
    long = _melt_hourly(df, date_col, "generation_mwh")
    long["generation_mwh"] = pd.to_numeric(long["generation_mwh"], errors="coerce")
    return long.dropna(subset=["generation_mwh"]).sort_values("dt").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 출력제어 이력 (발생여부 + 풍력만 제어량 MWh)
# ---------------------------------------------------------------------------

def load_curtailment(energy_type: str) -> pd.DataFrame:
    """DataFrame[dt, is_curtailed(bool), curtailment_mwh(float, 풍력만 실값·태양광은 NaN)]"""
    if energy_type == "wind":
        path = _find("풍력 출력제어횟수 및 제어량")
        df = _read_csv_any_encoding(path)
        long = _melt_hourly(df, "일자", "curtailment_mwh")
        long["curtailment_mwh"] = pd.to_numeric(long["curtailment_mwh"], errors="coerce").fillna(0.0)
        long["is_curtailed"] = long["curtailment_mwh"] > 0
    else:
        # 태양광 파일이 여러 개 있을 수 있어 "월별" 같은 다른 파일은 제외
        candidates = [f for f in os.listdir(DATA_DIR) if "태양광 출력제어횟수" in f and "월별" not in f]
        path = os.path.join(DATA_DIR, candidates[0])
        df = _read_csv_any_encoding(path)
        long = _melt_hourly(df, "일자", "flag")
        long["is_curtailed"] = long["flag"].astype(str) == "출력제어"
        long["curtailment_mwh"] = np.nan  # 공식적으로 미산정 (전력거래소 공식 확인)
    long = long.groupby("dt", as_index=False).agg(
        is_curtailed=("is_curtailed", "max"),
        curtailment_mwh=("curtailment_mwh", "sum" if energy_type == "wind" else "first"),
    )
    return long.sort_values("dt").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 전력수요 실측 (제주, 2017~2024 여러 스냅샷 파일을 병합)
# ---------------------------------------------------------------------------

def load_demand_actual() -> pd.DataFrame:
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("한국전력거래소_시간별 제주전력수요")]
    frames = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        df = _read_csv_any_encoding(path)
        date_col = df.columns[0]
        long = _melt_hourly(df, date_col, "demand_mw")
        long["demand_mw"] = pd.to_numeric(long["demand_mw"], errors="coerce")
        med = long["demand_mw"].median()
        if med and med > 10000:  # kWh 스케일 -> MWh로 보정 (build_curtailment_proposal 검증 시 확인된 패턴)
            long["demand_mw"] = long["demand_mw"] / 1000.0
        frames.append(long.dropna(subset=["demand_mw"]))
    merged = pd.concat(frames).drop_duplicates(subset="dt", keep="last").sort_values("dt")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ASOS 실측 기상관측 (지점: 184=제주, 185=고산, 188=성산)
# ---------------------------------------------------------------------------

def load_asos(station: str = "184") -> pd.DataFrame:
    """station별 2021~2023 파일을 합쳐 DataFrame[dt, temp, wind_speed, humidity, cloud, solar_rad] 반환."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith(f"SURFACE_ASOS_{station}_HR_"))
    frames = []
    for f in files:
        df = _read_csv_any_encoding(os.path.join(DATA_DIR, f))
        frames.append(df)
    raw = pd.concat(frames).drop_duplicates(subset="일시")
    out = pd.DataFrame({
        "dt": pd.to_datetime(raw["일시"]),
        "temp": pd.to_numeric(raw["기온(°C)"], errors="coerce"),
        "wind_speed": pd.to_numeric(raw["풍속(m/s)"], errors="coerce"),
        "humidity": pd.to_numeric(raw["습도(%)"], errors="coerce"),
        "cloud": pd.to_numeric(raw["전운량(10분위)"], errors="coerce"),
        "solar_rad": pd.to_numeric(raw["일사(MJ/m2)"], errors="coerce").fillna(0.0),
    })
    return out.dropna(subset=["dt"]).sort_values("dt").drop_duplicates(subset="dt").reset_index(drop=True)


def load_asos_multi(stations: tuple[str, ...] = ("184", "185", "188")) -> pd.DataFrame:
    """I장 방식 — 여러 관측소(기본 제주184·고산185·성산188)의 실측을 평균해 위치 오차를 상쇄.

    I장 원문: "여러 지점을 평균하면 개별 지점의 위치 불일치가 서로 상쇄되는 것으로 해석된다"
    (corr 0.656->0.763, NMAE 15.58%->12.03% 개선 확인됨). 풍력 컨버터 재보정에 사용.
    """
    frames = [load_asos(s).set_index("dt") for s in stations]
    combined = pd.concat(frames, axis=1, keys=stations)
    avg = pd.DataFrame({
        "wind_speed": combined.xs("wind_speed", axis=1, level=1).mean(axis=1, skipna=True),
        "temp": combined.xs("temp", axis=1, level=1).mean(axis=1, skipna=True),
        "humidity": combined.xs("humidity", axis=1, level=1).mean(axis=1, skipna=True),
        "cloud": combined.xs("cloud", axis=1, level=1).mean(axis=1, skipna=True),
        "solar_rad": combined.xs("solar_rad", axis=1, level=1).mean(axis=1, skipna=True),
    }).dropna(subset=["wind_speed"]).reset_index()
    return avg.sort_values("dt").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 시간/월 사인·코사인 피처 (분류기 공통 입력)
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, dt_col: str = "dt") -> pd.DataFrame:
    df = df.copy()
    hour = df[dt_col].dt.hour
    month = df[dt_col].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


@dataclass
class TrainTestSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def leakage_free_split(df: pd.DataFrame, dt_col: str, test_start: str, test_end: str) -> TrainTestSplit:
    """테스트 구간을 시간으로 명확히 분리 (리키지-프리). test_end는 미포함(exclusive)."""
    mask = (df[dt_col] >= test_start) & (df[dt_col] < test_end)
    return TrainTestSplit(train=df[~mask].reset_index(drop=True), test=df[mask].reset_index(drop=True))
