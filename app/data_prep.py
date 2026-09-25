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

# 원본 공공데이터 CSV 폴더. 환경변수 DATA_DIR로 지정하거나, 기본값(프로젝트 루트/data/raw)을 사용한다.
# (수정 전: 다른 실행환경의 절대경로가 하드코딩돼 있어 로컬에서 재학습이 불가능했음)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_PROJECT_ROOT, "data", "raw"))

# 출력제어 이력 파일이 커버하는 라벨 유효 구간 [start, end).
# - start: 출력제어 제도가 실제로 시작된 시점(그 이전은 '제어 없음'이 아니라 '라벨 의미 없음')
# - end: 발전량 실적이 2023.12까지만 있으므로 2024-01-01에서 자른다
CURTAILMENT_COVERAGE = {
    "wind": ("2021-01-01", "2024-01-01"),
    "solar": ("2021-10-01", "2024-01-01"),
}


def _read_csv_any_encoding(path: str, **kwargs) -> pd.DataFrame:
    for enc in ("cp949", "utf-8", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"could not decode {path}")


def _find(pattern_substr: str) -> str:
    """uploads 폴더에서 파일명에 pattern_substr이 포함된 첫 파일 경로를 반환."""
    for f in sorted(os.listdir(DATA_DIR)):
        if pattern_substr in f:
            return os.path.join(DATA_DIR, f)
    raise FileNotFoundError(f"no file containing '{pattern_substr}' in {DATA_DIR}")


def _melt_hourly(df: pd.DataFrame, date_col: str, value_name: str) -> pd.DataFrame:
    """'1시'~'24시' 와이드 포맷 -> (dt, value) 롱 포맷. 24시는 '다음날 00:00'으로 취급.

    dt는 해당 1시간 구간의 '끝 시각'이다(예: 1시 = 00:00~01:00 구간 -> 01:00).
    ASOS의 일시(정시 관측, 일사량은 직전 1시간 누적)와 같은 기준이라 그대로 merge할 수 있다.

    [버그 수정] 이전 코드는 24를 0으로 바꾼 뒤 '같은 날' 00:00에 붙여서, D일 24시 값이 D일 00:00
    (= 실제로는 D-1일 24시 자리)에 들어가 하루 어긋났다. 24시간 timedelta를 그대로 더하면
    자연스럽게 다음날 00:00이 된다.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    hour_cols = [c for c in df.columns if str(c).endswith("시")]
    long = df.melt(id_vars=date_col, value_vars=hour_cols, var_name="hour", value_name=value_name)
    long["hour"] = long["hour"].astype(str).str.replace("시", "", regex=False).astype(int)
    long["dt"] = long[date_col] + pd.to_timedelta(long["hour"], unit="h")
    return long[["dt", value_name]]


# ---------------------------------------------------------------------------
# 발전량 실적 (제주 전체, 태양광/풍력)
# ---------------------------------------------------------------------------

# 롱 포맷 지역별 데이터셋(한국전력거래소_지역별 시간별 태양광 및 풍력 발전량)의 필수 컬럼.
# 기존 "제주지역 ... 한시간단위실적"은 1시~24시 와이드 포맷이고, 이쪽은 행 단위다.
# 롱 포맷(지역별 시간별 발전량) 배포본은 연도마다 스키마가 미묘하게 다르다.
#   날짜 컬럼 : 2024년판 '거래일자' / 2025년판 '거래일'
#   지역 표기 : 2024년판 태양광 '제주도' · 풍력 '제주' / 2025년판 둘 다 '제주도'
# 완전일치로 거르면 한 해가 통째로 조용히 빠지므로, 날짜 컬럼은 후보 목록에서 찾고
# 지역은 '제주' 부분일치로 받는다.
_LONG_GEN_DATE_COLS = ("거래일", "거래일자")
_LONG_GEN_COLS = ("거래시간", "지역", "연료원")

# [중요] 제주 풍력은 2024-06-01 재생에너지 입찰제도 본격 운영과 함께 이 데이터셋에서
# 사실상 사라진다 — 발전을 안 한 게 아니라 거래가 다른 경로로 정산되기 때문으로 보인다.
#   월 합계: 2024-05 42,657MWh -> 2024-06 5,421 -> 2025-03 이후 1,000~2,800 (정상치의 4%)
#   풍속 상관: 2023년 0.786 -> 2024년 0.624 -> 2025년 0.540
#   평균 발전: 58MWh -> 42 -> 8.6
# 이 구간을 그대로 학습에 넣으면 컨버터가 '바람이 불어도 발전하지 않는다'를 학습한다.
# 출력제어 이력이 2024-05에서 끊긴 것과 같은 원인이다(README '제도 전환' 참고).
GENERATION_VALID_END = {"wind": "2024-06-01", "solar": None}


def _read_generation_wide(energy_type: str) -> pd.DataFrame:
    """기존 포맷: '제주지역 태양광/풍력 한시간단위실적' (1시~24시 와이드)."""
    fname_substr = "태양광 한시간단위실적" if energy_type == "solar" else "풍력 한시간단위실적"
    path = _find(fname_substr)
    df = _read_csv_any_encoding(path)
    df.columns = [str(c).strip() for c in df.columns]
    long = _melt_hourly(df, df.columns[0], "generation_mwh")
    long["generation_mwh"] = pd.to_numeric(long["generation_mwh"], errors="coerce")
    return long.dropna(subset=["generation_mwh"])


def _hour_to_end_of_hour(hours: pd.Series) -> tuple[pd.Series, str]:
    """거래시간 컬럼을 이 프로젝트의 규약(구간 '끝 시각')에 맞춘 시간 오프셋으로 변환.

    이 저장소의 dt는 항상 1시간 구간의 끝 시각이다(1시 = 00:00~01:00 -> 01:00).
    공공데이터 문서에 1~24시인지 0~23시인지 명시가 없어 값 범위로 판별한다.
      - 1~24 : 기존 파일과 같은 규약 -> 그대로 더한다
      - 0~23 : 구간 '시작 시각' 표기로 보고 +1 시간
    [주의] 0~23이 '끝 시각'을 뜻하는 다른 해석도 가능하다. 겹치는 구간으로 반드시
    검증할 것 -- scripts/verify_generation_overlap.py 참고.
    """
    lo, hi = int(hours.min()), int(hours.max())
    if hi == 24:
        return hours, f"1~24시(구간 끝) 규약으로 해석 [관측 범위 {lo}~{hi}]"
    if lo == 0:
        return hours + 1, f"0~23시(구간 시작) 규약으로 해석해 +1시간 [관측 범위 {lo}~{hi}]"
    return hours, f"기본(구간 끝) 규약으로 해석 [관측 범위 {lo}~{hi}]"


def _read_generation_long(energy_type: str) -> pd.DataFrame:
    """롱 포맷 지역별 파일들에서 제주 + 해당 연료원만 추출. 없으면 빈 DataFrame."""
    fuel = "태양광" if energy_type == "solar" else "풍력"
    frames = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.lower().endswith(".csv"):
            continue
        try:
            df = _read_csv_any_encoding(os.path.join(DATA_DIR, f), nrows=5)
        except Exception:
            continue
        cols = {str(c).strip() for c in df.columns}
        date_col = next((c for c in _LONG_GEN_DATE_COLS if c in cols), None)
        if date_col is None or not set(_LONG_GEN_COLS).issubset(cols):
            continue
        df = _read_csv_any_encoding(os.path.join(DATA_DIR, f))
        df.columns = [str(c).strip() for c in df.columns]
        val_col = next((c for c in df.columns if "거래량" in c or "발전량" in c), None)
        if val_col is None:
            continue
        sub = df[df["지역"].astype(str).str.contains("제주", na=False)
                 & df["연료원"].astype(str).str.contains(fuel, na=False)].copy()
        if sub.empty:
            continue
        hours = pd.to_numeric(sub["거래시간"], errors="coerce")
        sub = sub[hours.notna()]
        hours, note = _hour_to_end_of_hour(hours.dropna().astype(int))
        sub["dt"] = pd.to_datetime(sub[date_col], errors="coerce") + pd.to_timedelta(hours, unit="h")
        sub["generation_mwh"] = pd.to_numeric(sub[val_col], errors="coerce")
        sub = sub.dropna(subset=["dt", "generation_mwh"])
        if not sub.empty:
            print(f"  [롱포맷] {f}: {fuel} {len(sub)}행 "
                  f"({sub['dt'].min():%Y-%m}~{sub['dt'].max():%Y-%m}) — {note}")
            frames.append(sub[["dt", "generation_mwh"]])
    if not frames:
        return pd.DataFrame(columns=["dt", "generation_mwh"])
    out = pd.concat(frames).groupby("dt", as_index=False)["generation_mwh"].sum()
    end = GENERATION_VALID_END.get(energy_type)
    if end:
        dropped = int((out["dt"] >= end).sum())
        if dropped:
            print(f"  [발전량:{energy_type}] 제도 전환({end}) 이후 {dropped}시간 제외 "
                  f"— 입찰시장 이전으로 집계가 끊긴 구간")
        out = out[out["dt"] < end]
    return out


def load_generation_actual(energy_type: str, source: str = "all") -> pd.DataFrame:
    """energy_type: 'solar' | 'wind' -> DataFrame[dt, generation_mwh]

    source:
      "all"    기존 + 신규 (기본). 라벨 구간(~2024-01)이 전부 기존 출처 안에 있어
               분류기·라벨 작업에는 이 값을 쓴다.
      "legacy" 기존 '제주지역 한시간단위실적'만
      "market" 신규 '지역별 시간별 전력거래량'만

    [두 출처는 모집단이 다르다 — 섞으면 안 되는 경우가 있다]
    신규 데이터셋은 "전력시장에 참여하는 발전기의 전력거래량"으로, 전기사업법 시행령
    제19조 1항 2호에 따른 한전 직접거래(PPA)·자가용을 **포함하지 않는다**. 또 송전단 기준이고
    ESS 충방전량이 섞여 있다. 그 결과 태양광 연간 총합이 2023년 487.2GWh(기존) ->
    2024년 448.5GWh(신규)로 설비가 늘었는데도 8% 줄어든다. 일사량 1MJ/m2당 발전량도
    2021->2023에 +17.5%, +24.6%로 늘다가 출처가 바뀌는 2024년 -2.0%, 2025년 -7.0%로 꺾인다.

    겹치는 구간이 없어 보정계수를 직접 구할 수 없고, 입찰제도 하의 제어 증가와도 섞여 있다.
    그래서 컨버터는 한 출처만 쓴다 — 실측 비교에서 신규 단독(NMAE 22.74%)이 혼합(25.09%)보다
    학습 데이터가 1/5인데도 더 정확했다.

    capacity_factor는 비율이라 출처 간 배율 차이가 분자·분모에서 상쇄되므로, 컨버터와
    서빙 대리지표가 같은 출처를 쓰기만 하면 분류기의 학습(기존 출처) 관계와 호환된다.
    """
    if source == "market":
        return _read_generation_long(energy_type).sort_values("dt").reset_index(drop=True)
    wide = _read_generation_wide(energy_type)
    if source == "legacy":
        return wide.sort_values("dt").drop_duplicates(subset="dt").reset_index(drop=True)
    extra = _read_generation_long(energy_type)
    if not extra.empty:
        new_only = extra[~extra["dt"].isin(set(wide["dt"]))]
        if not new_only.empty:
            print(f"  [발전량:{energy_type}] 기존 {wide['dt'].max():%Y-%m}까지 + "
                  f"롱포맷에서 {len(new_only)}시간 추가 (~{new_only['dt'].max():%Y-%m})")
            wide = pd.concat([wide, new_only])
    return wide.sort_values("dt").drop_duplicates(subset="dt").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 제주 하루전 SMP — 신제도 구간의 대리 라벨
# ---------------------------------------------------------------------------
# 2024-06 재생에너지 입찰제도 전환 이후 제어 실적이 공개되지 않는다. 대신 재생에너지 입찰
# 상한이 0원/kWh이므로 SMP <= 0은 '재생에너지가 낙찰되지 못한 시간' = 시장 기반 출력제어의
# 대리 지표가 된다. 구 라벨과 겹치는 2024-03~05 구간에서 검증한 결과
#   풍력   정밀도 0.665 / 재현율 0.626 / F1 0.645
#   태양광 정밀도 0.526 / 재현율 0.696 / F1 0.599
# 임계값을 5원·20원으로 올려도 F1이 거의 변하지 않아(0.649 / 0.604) 0이 자연스러운 경계다.
#
# [반드시 '하루전' 계열을 쓸 것] 전일 공개되므로 하루 전 예측 시점에 알 수 있다.
# EPSIS '시간별 SMP'(실시간·정산 계열)는 사후 확정값이고 하루전과 5.6% 불일치하며
# 음수가 76건뿐이다(하루전은 196건). 실시간 계열을 라벨로 쓰면 미래 정보가 새어 들어간다.
SMP_SURPLUS_THRESHOLD = 0.0


def load_smp_dayahead() -> pd.DataFrame:
    """전력거래소 제주시범사업 하루전 SMP -> DataFrame[dt, smp] (원/kWh).

    파일: data/raw/KPX_제주_하루전SMP_*.xlsx (scripts/fetch_smp.py로 내려받음)
    포맷: 1행 제목, 2행 헤더(구분 · 1h~24h · 최대 · 최소 · 평균), 이후 날짜별 행.
    """
    files = sorted(f for f in os.listdir(DATA_DIR) if "제주_하루전SMP" in f and f.endswith(".xlsx"))
    if not files:
        raise FileNotFoundError(
            f"{DATA_DIR}에 KPX_제주_하루전SMP_*.xlsx가 없습니다 — "
            f"`python -m scripts.fetch_smp`로 내려받으세요"
        )
    frames = []
    for f in files:
        d = pd.read_excel(os.path.join(DATA_DIR, f), header=1)
        d = d[pd.to_numeric(d["구분"], errors="coerce").notna()]
        base = pd.to_datetime(d["구분"].astype(int).astype(str), format="%Y%m%d")
        for h in range(1, 25):
            col = f"{h}h"
            if col not in d.columns:
                continue
            frames.append(pd.DataFrame({
                "dt": base + pd.to_timedelta(h, unit="h"),
                "smp": pd.to_numeric(d[col], errors="coerce"),
            }))
    out = pd.concat(frames).dropna(subset=["smp"])
    return out.sort_values("dt").drop_duplicates(subset="dt").reset_index(drop=True)


def load_surplus_proxy_label() -> pd.DataFrame:
    """DataFrame[dt, smp, is_surplus] — 신제도 구간의 대리 라벨."""
    d = load_smp_dayahead()
    d["is_surplus"] = (d["smp"] <= SMP_SURPLUS_THRESHOLD).astype(int)
    return d


# ---------------------------------------------------------------------------
# 설비용량 대리지표 / 정규화 피처
# ---------------------------------------------------------------------------

def capacity_proxy(gen: pd.DataFrame, window_days: int = 365, q: float = 0.99,
                   min_days: int = 60) -> pd.Series:
    """발전량 이력만으로 만든 설비용량 대리지표 (공개 설비용량 시계열이 없어서).

    과거 window_days 구간의 q분위수를 쓴다. 태양광은 절반이 야간(0)이라 99분위가
    사실상 '맑은 날 정오 최대 출력'에 해당하고, 이는 설비용량에 비례한다.

    [인과성] rolling 후 shift(1)로 현재 시각을 제외한다. 현재 발전량이 자기 분모에
    들어가면 미래 정보가 새고, 정규화된 값이 인위적으로 1 근처에 묶인다.

    [콜드스타트] 발전량 실적은 2019-12부터 있고 라벨은 2021-01부터라, 라벨 구간
    시작 시점에는 이미 1년 이상의 이력이 쌓여 있다. 따라서 학습 데이터 손실은 없다.
    """
    s = gen.set_index("dt")["generation_mwh"].sort_index()
    prox = s.rolling(f"{window_days}D", min_periods=24 * min_days).quantile(q).shift(1)
    return prox.ffill()


def latest_capacity_proxy(gen: pd.DataFrame) -> float:
    """서빙에서 쓰는 '고정' 설비용량 대리지표 = 전체 이력의 마지막 유효값.

    컨버터(이용률 -> MWh 환산)와 분류기(MWh -> 이용률)가 반드시 같은 상수를 써야 한다.
    다르면 왕복에서 비율만큼 왜곡이 남는다(예: 242 vs 232이면 capacity_factor가 4% 부풀려짐).
    """
    return round(float(capacity_proxy(gen).dropna().iloc[-1]), 2)


def add_normalized_features(df: pd.DataFrame, gen_full: pd.DataFrame,
                            demand: pd.DataFrame | None = None) -> pd.DataFrame:
    """capacity_factor(이용률)와 penetration(수요 대비 침투율)을 추가한다.

    왜 둘 다인가:
      - capacity_factor = 발전량 / 설비용량대리 -> '날씨가 얼마나 좋았나'. 증설이 있어도
        범위가 [0, 1] 근처로 안정돼 학습 구간 밖으로 나가지 않는다.
      - penetration = 발전량 / 수요 -> '수요 대비 잉여 압력'. 출력제어를 실제로 일으키는
        물리량이다. 증설과 함께 커지지만, 그 증가는 노이즈가 아니라 신호다.
      이용률만 쓰면 절대 규모 정보가 사라져 제어 요인을 잃고, 절대 발전량만 쓰면
      증설로 학습 범위를 벗어나 RandomForest가 경계에서 포화된다.

    gen_full: 대리지표 계산용 '전체 기간' 발전량(라벨 구간으로 자르기 전).
    """
    out = df.copy()
    prox = capacity_proxy(gen_full)
    out["capacity_proxy_mwh"] = out["dt"].map(prox)
    out["capacity_factor"] = (out["generation_mwh"] / out["capacity_proxy_mwh"]).replace(
        [np.inf, -np.inf], np.nan)
    if demand is not None:
        out = out.merge(demand, on="dt", how="left")
        out["penetration"] = (out["generation_mwh"] / out["demand_mw"]).replace(
            [np.inf, -np.inf], np.nan)
    return out


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
        candidates = sorted(f for f in os.listdir(DATA_DIR) if "태양광 출력제어횟수" in f and "월별" not in f)
        if not candidates:
            raise FileNotFoundError(f"no file containing '태양광 출력제어횟수' in {DATA_DIR}")
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
    # 파일명 정렬: 스냅샷이 겹치는 시간은 keep="last"로 처리하므로, 정렬하지 않으면
    # os.listdir 순서(환경마다 다름)에 따라 결과가 달라진다.
    files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("한국전력거래소_시간별 제주전력수요"))
    if not files:
        raise FileNotFoundError(f"no demand files in {DATA_DIR}")
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
    """station별 파일을 합쳐 DataFrame[dt, temp, wind_speed, humidity, cloud, solar_rad] 반환.

    두 가지 파일 형식을 모두 읽는다.
      - SURFACE_ASOS_{지점}_HR_*.csv : 지점 하나당 파일 하나 (기존)
      - 그 밖의 CSV 중 '지점'과 '일시' 컬럼을 가진 파일 : 여러 지점이 한 파일에 든 형식
        (기상자료개방포털에서 지점을 여러 개 골라 받으면 OBS_ASOS_TIM_*.csv로 나온다)
    파일명 규칙에만 의존하면 포털 다운로드 이름이 바뀔 때마다 조용히 실패하므로,
    컬럼 구조로도 판별한다.
    """
    frames = []
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.lower().endswith(".csv"):
            continue
        if f.startswith(f"SURFACE_ASOS_{station}_HR_"):
            frames.append(_read_csv_any_encoding(os.path.join(DATA_DIR, f)))
            continue
        try:
            head = _read_csv_any_encoding(os.path.join(DATA_DIR, f), nrows=5)
        except Exception:
            continue
        cols = {str(c).strip() for c in head.columns}
        if not {"지점", "일시", "기온(°C)"}.issubset(cols):
            continue
        df = _read_csv_any_encoding(os.path.join(DATA_DIR, f))
        df.columns = [str(c).strip() for c in df.columns]
        sub = df[pd.to_numeric(df["지점"], errors="coerce") == int(station)]
        if not sub.empty:
            frames.append(sub)
    if not frames:
        raise FileNotFoundError(f"지점 {station}의 ASOS 파일을 {DATA_DIR}에서 찾지 못했습니다")

    raw = pd.concat(frames)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.drop_duplicates(subset="일시")
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
# 학습용 라벨 프레임 (전체 달력 기준)
# ---------------------------------------------------------------------------

def build_labeled_hourly(energy_type: str) -> pd.DataFrame:
    """발전량 실적(모든 시간)을 기준 달력으로 삼아 출력제어 라벨을 붙인다.

    [버그 수정] 출력제어 이력 파일에는 '제어가 발생한 날'만 들어 있다. 이전 코드는 이 파일과
    inner join을 해서 제어가 없었던 날(음성 사례 대부분)이 통째로 빠졌고, 그 결과
      - 모델이 '오늘 제어가 있을까'가 아니라 '제어가 있는 날 몇 시에 있을까'를 학습했고
      - 테스트 양성 비율이 약 20%로 부풀어 상위5%포착률이 수학적 상한(약 0.25)에 막혔다.
    여기서는 left join 후 이력에 없는 시간은 '제어 없음'(제어량 0)으로 채우고,
    라벨이 의미 있는 구간(CURTAILMENT_COVERAGE)만 남긴다.

    반환: DataFrame[dt, generation_mwh, is_curtailed(int 0/1), curtailment_mwh]
          (태양광 curtailment_mwh는 공식 미산정이므로 NaN 유지)
    """
    gen = load_generation_actual(energy_type)
    curt = load_curtailment(energy_type)
    df = gen.merge(curt, on="dt", how="left")
    df["is_curtailed"] = df["is_curtailed"].fillna(False).astype(bool).astype(int)
    if energy_type == "wind":
        df["curtailment_mwh"] = df["curtailment_mwh"].fillna(0.0)
    start, end = CURTAILMENT_COVERAGE[energy_type]
    df = df[(df["dt"] >= start) & (df["dt"] < end)]
    return df.sort_values("dt").reset_index(drop=True)


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
