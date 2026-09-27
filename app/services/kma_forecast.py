"""기상청 단기예보 -> /predict 요청 본문 어댑터.

[왜 이게 '연동'의 전부인가]
/predict는 이미 호출자가 기상값 24개를 넣는 구조다. 그래서 연동은 새 엔드포인트가 아니라
**예보를 요청 본문으로 바꾸는 어댑터**다. 이 모듈은 순수 변환 + HTTP 호출만 하고, 모델은 건드리지
않는다.

[단기예보의 제약과 대응]
  거래일 D의 24시간을 D-1에 받아야 하고, 하루전시장 입찰마감이 D-1 11시다.
  발표시각(base_time)은 0200·0500·0800·1100·1400·1700·2000·2300이므로 **D-1 08시 발표**를 쓰면
  11시 마감에 맞는다(0500도 가능, 1100은 마감과 동시라 위험).

  항목 매핑:
    TMP -> temp          (그대로)
    WSD -> wind_speed    (그대로)
    SKY -> cloud         (1/3/4 -> 대표 전운량 2.5/7/9.5. 양자화 손실이 남는다)
    REH -> humidity      (일사량 추정 입력)
    일사량 -> **없음**    -> radiation_estimator로 추정 (app/solar_radiation.py 주석 참고)

  시각 규약: 예보의 fcstTime은 '그 시각'을 뜻하지만 이 저장소는 `h시`를 '그 구간의 끝'으로 쓴다
  (13시 = 12:00~13:00). 예보 fcstTime HH00은 구간 [HH-1, HH]의 대표값으로 보고 hour=HH로 넣는다 —
  ASOS 적산 규약과 같은 자리에 놓이게 하려는 것이다.

[검증 상태] 격자 변환은 공식 예시(서울 종로구 60,127 / 제주시청 53,38)로 확인했다.
HTTP 호출 경로는 API 키가 필요해 이 저장소에서 검증하지 못했다 — `sample_path`로 저장된 응답을
넣어 변환부만 시험할 수 있다.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from app.data_prep import add_time_features
from app.model_io import load_artifact
from app.solar_radiation import add_solar_geometry, cloud_from_sky, latlon_to_grid

ENDPOINT = ("https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst")
# 발표시각. 하루전시장 입찰마감(D-1 11시) 전에 거래일 24시간을 다 받으려면 0800이 기본이다.
BASE_TIMES = ("0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300")
DEFAULT_BASE_TIME = "0800"
NEEDED = ("TMP", "WSD", "SKY", "REH")


def _rows_to_frame(items: list[dict], target: date) -> pd.DataFrame:
    """예보 항목 리스트(category별 행) -> 시간별 와이드 프레임."""
    df = pd.DataFrame(items)
    if df.empty:
        raise ValueError("예보 응답에 항목이 없습니다")
    df = df[df.category.isin(NEEDED)].copy()
    ymd = target.strftime("%Y%m%d")
    df = df[df.fcstDate == ymd]
    if df.empty:
        raise ValueError(f"응답에 {ymd} 예보가 없습니다 — base_date/base_time을 확인하세요")
    df["hour"] = df.fcstTime.str[:2].astype(int)
    df["value"] = pd.to_numeric(df.fcstValue, errors="coerce")
    wide = df.pivot_table(index="hour", columns="category", values="value", aggfunc="first")
    missing = [c for c in NEEDED if c not in wide.columns]
    if missing:
        raise ValueError(f"예보에 필요한 항목이 없습니다: {missing}")
    return wide.reset_index()


def _to_weather(wide: pd.DataFrame, target: date, station: str = "184") -> list[dict]:
    """예보 프레임 -> /predict의 weather 24개. 일사량은 추정해서 채운다."""
    base = datetime.combine(target, datetime.min.time())
    # hour 0은 자정을 뜻하고 이 저장소 규약에서는 전날 24시다 — 거래일 24시로 옮긴다.
    wide = wide.copy()
    wide.loc[wide.hour == 0, "hour"] = 24
    wide = wide[(wide.hour >= 1) & (wide.hour <= 24)].sort_values("hour")

    df = pd.DataFrame({"dt": [base + timedelta(hours=int(h)) for h in wide.hour]})
    df["temp"] = wide.TMP.to_numpy()
    df["humidity"] = wide.REH.to_numpy()
    df["wind_speed"] = wide.WSD.to_numpy()
    df["sky"] = wide.SKY.to_numpy()
    df["cloud"] = cloud_from_sky(df["sky"])
    df = add_solar_geometry(add_time_features(df), station)

    art = load_artifact("radiation_estimator")
    night = float(art["meta"].get("night_threshold_mj", 0.02))
    rad = np.zeros(len(df))
    mask = (df["clear_sky_mj"] > night).to_numpy()
    if mask.any():
        pred = art["model"].predict(df.loc[mask, art["features"]])
        if (art["meta"].get("target") or "").startswith("clear_sky_index"):
            pred = np.clip(pred, 0, 1.2) * df.loc[mask, "clear_sky_mj"].to_numpy()
        rad[mask] = np.clip(pred, 0, None)
    df["solar_rad"] = rad

    out = []
    # r.dt는 pandas의 .dt 접근자와 이름이 겹치므로 반드시 r["dt"]로 꺼낸다.
    for _, r in df.iterrows():
        hour = int(pd.Timestamp(r["dt"]).hour) or 24
        out.append({"hour": hour, "temp": round(float(r["temp"]), 1),
                    "cloud": round(float(r["cloud"]), 1),
                    "wind_speed": round(float(r["wind_speed"]), 1),
                    "solar_rad": round(float(r["solar_rad"]), 3)})
    if len(out) != 24:
        raise ValueError(f"예보에서 만든 시간이 {len(out)}개입니다 — 24개가 필요합니다 "
                         f"(발표시각을 더 이른 것으로 바꿔보세요)")
    return out


def fetch(target: date, lat: float = 33.5141, lon: float = 126.5297,
          base_time: str = DEFAULT_BASE_TIME, service_key: str | None = None,
          sample_path: str | None = None, timeout: float = 10.0) -> list[dict]:
    """거래일 target의 예보를 받아 /predict의 weather 24개로 만든다.

    service_key는 공공데이터포털 인증키다. 환경변수 KMA_SERVICE_KEY로도 받는다.
    sample_path에 저장된 JSON 응답을 주면 HTTP 호출 없이 변환만 수행한다(오프라인 시험용).
    """
    if sample_path:
        with open(sample_path, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        import urllib.parse
        import urllib.request
        key = service_key or os.environ.get("KMA_SERVICE_KEY")
        if not key:
            raise RuntimeError(
                "기상청 인증키가 없습니다 — 공공데이터포털에서 '기상청_단기예보 조회서비스'를 "
                "신청하고 KMA_SERVICE_KEY 환경변수에 넣으세요")
        nx, ny = latlon_to_grid(lat, lon)
        base_date = (target - timedelta(days=1)).strftime("%Y%m%d")
        q = urllib.parse.urlencode({
            "serviceKey": key, "dataType": "JSON", "numOfRows": "1000", "pageNo": "1",
            "base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny})
        with urllib.request.urlopen(f"{ENDPOINT}?{q}", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

    body = (payload.get("response") or {}).get("body") or {}
    header = (payload.get("response") or {}).get("header") or {}
    if header.get("resultCode") not in (None, "00"):
        raise RuntimeError(f"기상청 API 오류 {header.get('resultCode')}: {header.get('resultMsg')}")
    items = ((body.get("items") or {}).get("item")) or []
    return _to_weather(_rows_to_frame(items, target), target)


def build_predict_request(energy_type: str, target: date, region: str = "제주",
                          demand_forecast_mw: list[float] | None = None,
                          **kw) -> dict:
    """/predict에 그대로 POST할 수 있는 요청 본문.

    풍력에도 태양광 기상값(solar_rad·temp·cloud)이 함께 들어간다 — 출력제어는 재생에너지
    합계로 결정되므로 계통 전체 침투율 모델로 자동 전환되게 하려는 것이다
    (README '계통 전체 침투율').
    """
    body = {"energy_type": energy_type, "region": region,
            "target_date": target.isoformat(), "weather": fetch(target, **kw)}
    if demand_forecast_mw is not None:
        body["demand_forecast_mw"] = demand_forecast_mw
    return body
