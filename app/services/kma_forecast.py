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
from app.services import kim_forecast
from app.services.credentials import resolve_secret, setup_hint
from app.solar_radiation import add_solar_geometry, cloud_from_sky, latlon_to_grid

# 같은 서비스(VilageFcstInfoService_2.0 / 동네예보 > 단기예보조회)를 두 포털이 각자 제공한다.
# 응답 스키마가 동일하므로 바뀌는 것은 호스트와 인증 파라미터 이름뿐이다.
#
#   API허브        인증 파라미터 authKey     — 수치모델 API가 여기에만 있어 키를 통일하기 좋다
#   공공데이터포털   인증 파라미터 serviceKey
#
# 환경변수 이름을 각 포털의 파라미터 이름과 똑같이 맞췄다 — 어느 포털의 키인지 헷갈릴 자리를
# 없애려는 것이다. 둘 다 있으면 API허브를 쓴다.
PORTALS = {
    "apihub": {
        "url": "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst",
        "key_param": "authKey",
        "env": "KMA_AUTH_KEY",
        "label": "기상청 API허브 (예보 > 동네예보 > 단기예보조회)",
    },
    "data.go.kr": {
        "url": "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst",
        "key_param": "serviceKey",
        "env": "KMA_SERVICE_KEY",
        "label": "공공데이터포털 (기상청_단기예보 조회서비스)",
    },
}
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


def _to_weather(wide: pd.DataFrame, target: date, station: str = "184",
                rad_override: dict[int, float] | None = None) -> list[dict]:
    """예보 프레임 -> /predict의 weather 24개.

    rad_override가 오면(KIM 국지모델 일사량 예보) 그 값을 쓰고, 없으면 radiation_estimator로
    추정한다. 24시간이 다 채워지지 않은 override는 호출부에서 이미 None으로 바뀌어 온다 —
    일부는 예보값, 일부는 추정값인 상태를 만들지 않기 위한 것이다.
    """
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

    if rad_override is not None:
        hours = [int(pd.Timestamp(t).hour) or 24 for t in df["dt"]]
        df["solar_rad"] = np.clip([rad_override[h] for h in hours], 0, None)
    else:
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


def _ssl_context():
    """TLS 검증용 컨텍스트. CA 번들을 명시한다.

    [왜 필요한가] macOS의 python.org 설치판은 시스템 키체인을 쓰지 않고, 'Install
    Certificates.command'를 실행하지 않으면 **CA 저장소가 비어 있다**(ssl.get_default_verify_paths()의
    cafile이 None). 그 상태에서는 모든 HTTPS가 CERTIFICATE_VERIFY_FAILED로 막힌다 — 기상청뿐
    아니라 google.com도 막히므로 서버 문제로 오해하기 쉽다.

    검증을 끄는 것은 답이 아니다(중간자 공격을 못 막는다). certifi가 들어 있으면 그 번들을
    쓰고, 없으면 시스템 기본값으로 둔다.
    """
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def resolve_key(portal: str | None = None, key: str | None = None) -> tuple[str, str]:
    """(포털 이름, 인증키). 환경변수 -> macOS 키체인 순으로 찾는다.

    키체인을 함께 보는 이유는 매일 도는 자동 수집(scripts/kim_validation_log.py) 때문이다 —
    cron/launchd는 로그인 셸을 거치지 않아 ~/.zshrc의 export가 보이지 않는다.
    """
    if portal and portal not in PORTALS:
        raise ValueError(f"portal은 {list(PORTALS)} 중 하나여야 합니다 (받은 값: {portal!r})")
    if key:
        return portal or "apihub", key
    order = [portal] if portal else list(PORTALS)
    for name in order:
        v = resolve_secret(PORTALS[name]["env"])
        if v:
            return name, v
    labels = "\n".join(f"  {PORTALS[n]['env']:18} {PORTALS[n]['label']}" for n in order)
    raise RuntimeError(
        "기상청 인증키를 찾지 못했습니다 (환경변수·키체인 모두 없음).\n"
        f"쓸 수 있는 이름:\n{labels}\n\n" + setup_hint(PORTALS[order[0]]["env"]))


def fetch(target: date, lat: float = 33.5141, lon: float = 126.5297,
          base_time: str = DEFAULT_BASE_TIME, key: str | None = None,
          portal: str | None = None, sample_path: str | None = None,
          timeout: float = 10.0, use_kim: bool | None = None,
          radiation_source: list | None = None,
          kim_sample_path: str | None = None) -> list[dict]:
    """거래일 target의 예보를 받아 /predict의 weather 24개로 만든다.

    인증키는 환경변수에서 읽는다 — KMA_AUTH_KEY(API허브) 또는 KMA_SERVICE_KEY(공공데이터포털).
    key 인자로 직접 줄 수도 있지만 **소스나 로그에 키를 남기지 말 것**.
    sample_path에 저장된 JSON 응답을 주면 HTTP 호출 없이 변환만 수행한다(오프라인 시험용).
    """
    if sample_path:
        with open(sample_path, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        import urllib.error
        import urllib.parse
        import urllib.request
        name, auth = resolve_key(portal, key)
        cfg = PORTALS[name]
        nx, ny = latlon_to_grid(lat, lon)
        base_date = (target - timedelta(days=1)).strftime("%Y%m%d")
        # [주의] 공공데이터포털 인증키는 '일반 인증키(Encoding)'과 '(Decoding)' 두 형태로 나온다.
        # urlencode가 한 번 더 인코딩하므로 여기에는 **Decoding 키**를 넣어야 한다. Encoding 키를
        # 넣으면 %2B가 %252B가 되어 SERVICE_KEY_IS_NOT_REGISTERED_ERROR가 난다 — 가장 흔한 실패다.
        q = urllib.parse.urlencode({
            cfg["key_param"]: auth, "dataType": "JSON", "numOfRows": "1000", "pageNo": "1",
            "base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny})
        url = f"{cfg['url']}?{q}"
        try:
            with urllib.request.urlopen(url, timeout=timeout,
                                        context=_ssl_context()) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            # 401/403은 키가 틀렸거나 활용신청이 아직 승인되지 않은 상태다 — 원인을 짚어준다.
            hint = ""
            if e.code in (401, 403):
                hint = (f"\n  인증이 거부됐습니다({e.code}). 확인할 것:\n"
                        f"  1) {cfg['env']}에 넣은 키가 이 포털({cfg['label']})의 키인지\n"
                        f"  2) 활용신청이 '승인' 상태인지 (신청 직후에는 대기일 수 있습니다)\n")
                if name == "data.go.kr":
                    hint += ("  3) 공공데이터포털은 키가 Encoding/Decoding 두 형태로 나옵니다 — "
                             "**Decoding 키**를 넣어야 합니다\n")
            raise RuntimeError(f"{cfg['label']} HTTP {e.code} {e.reason}{hint}") from None
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise RuntimeError(
                    "TLS 인증서 검증에 실패했습니다. 이 파이썬에 CA 번들이 연결돼 있지 않은 "
                    "상태로 보입니다(기상청만이 아니라 모든 HTTPS가 막힙니다).\n"
                    "  해결 1) pip install certifi  — 설치돼 있으면 이 코드가 자동으로 씁니다\n"
                    "  해결 2) macOS python.org 설치판이면 "
                    "'/Applications/Python 3.x/Install Certificates.command' 실행\n"
                    "  확인)  python -c \"import ssl; print(ssl.get_default_verify_paths())\"\n"
                    "         cafile이 None이면 위 상태입니다.\n"
                    f"  원본: {e}") from None
            raise
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            # 인증 실패·미승인 상태에서는 JSON이 아니라 XML 오류 문서가 온다.
            raise RuntimeError(
                f"{cfg['label']} 응답이 JSON이 아닙니다 — 인증키나 활용신청 상태를 확인하세요.\n"
                f"응답 앞부분: {raw[:300]}") from None

    body = (payload.get("response") or {}).get("body") or {}
    header = (payload.get("response") or {}).get("header") or {}
    if header.get("resultCode") not in (None, "00"):
        raise RuntimeError(f"기상청 API 오류 {header.get('resultCode')}: {header.get('resultMsg')}")
    items = ((body.get("items") or {}).get("item")) or []

    # 일사량은 단기예보에 없다. KIM 국지모델(1.5km) 예보값을 먼저 시도하고, 못 받으면 추정한다.
    # 실패 사유를 삼키지 않도록 어느 쪽을 썼는지 radiation_source에 남긴다.
    # sample_path(오프라인 재생)에서는 KIM 샘플을 따로 주지 않는 한 네트워크를 타지 않는다.
    if use_kim is None:
        use_kim = sample_path is None or kim_sample_path is not None
    override, note = None, []
    if use_kim:
        override = kim_forecast.radiation_override(
            target, note=note, lat=lat, lon=lon, key=key, sample_path=kim_sample_path)
    if radiation_source is not None:
        radiation_source.append("kim_l010" if override is not None
                                else "estimator" + (f" (KIM 폴백: {note[0]})" if note else ""))
    return _to_weather(_rows_to_frame(items, target), target, rad_override=override)


def self_check(target: date | None = None, **kw) -> None:
    """인증키가 실제로 동작하는지 한 번에 확인한다 — 키를 넣은 직후 이걸 돌려볼 것.

    실행: python -m app.services.kma_forecast
    """
    from datetime import date as _date, timedelta as _td
    target = target or (_date.today() + _td(days=1))
    print("=" * 70)
    for name, cfg in PORTALS.items():
        has = "있음" if os.environ.get(cfg["env"]) else "없음"
        print(f"  {cfg['env']:18} {has:4}  {cfg['label']}")
    name, _ = resolve_key(kw.get("portal"), kw.get("key"))
    lat, lon = kw.get("lat", 33.5141), kw.get("lon", 126.5297)
    print(f"\n사용 포털: {PORTALS[name]['label']}")
    print(f"거래일 {target} · 좌표 ({lat}, {lon}) -> 격자 {latlon_to_grid(lat, lon)} · "
          f"발표시각 {kw.get('base_time', DEFAULT_BASE_TIME)}")
    src: list[str] = []
    weather = fetch(target, radiation_source=src, **kw)
    print(f"\n✅ 예보 {len(weather)}시간 수신")
    if src:
        ok = src[0] == "kim_l010"
        print(f"  일사량 출처: {'KIM 국지예보모델 1.5km (예보값)' if ok else src[0]}")
        if not ok:
            print("     KIM을 못 받아 추정값으로 돌아갔습니다. 정보 손실 +2.99%p가 그대로 남습니다.\n"
                  "     scripts/verify_kim_radiation.py로 분석시각 가용성을 확인해보세요.")
    print(f"  {'시':>3} {'기온':>6} {'풍속':>6} {'운량':>6} {'일사량':>8}")
    for w in weather:
        if w["hour"] in (1, 6, 9, 12, 13, 15, 18, 21, 24):
            print(f"  {w['hour']:3} {w['temp']:6.1f} {w['wind_speed']:6.1f} "
                  f"{w['cloud']:6.1f} {w['solar_rad']:8.3f}")
    rad = sum(w["solar_rad"] for w in weather)
    print(f"\n  일사량 일적산 {rad:.2f} MJ/m² · 최대 풍속 "
          f"{max(w['wind_speed'] for w in weather):.1f} m/s")
    print("  -> build_predict_request()로 /predict에 바로 넣을 수 있습니다.")


def build_predict_request(energy_type: str, target: date, region: str = "제주",
                          demand_forecast_mw: list[float] | None = None,
                          **kw) -> dict:
    """/predict에 그대로 POST할 수 있는 요청 본문.

    풍력에도 태양광 기상값(solar_rad·temp·cloud)이 함께 들어간다 — 출력제어는 재생에너지
    합계로 결정되므로 계통 전체 침투율 모델로 자동 전환되게 하려는 것이다
    (README '계통 전체 침투율').
    """
    src: list[str] = []
    body = {"energy_type": energy_type, "region": region, "target_date": target.isoformat(),
            "weather": fetch(target, radiation_source=src, **kw)}
    # 요청 본문에는 넣지 않는다(스키마에 없는 필드는 거부된다). 호출부가 보려면 반환값 대신
    # fetch(radiation_source=...)를 직접 쓰면 된다. 여기서는 표준출력에만 남긴다.
    if src:
        print(f"  일사량 출처: {src[0]}")
    if demand_forecast_mw is not None:
        body["demand_forecast_mw"] = demand_forecast_mw
    return body


if __name__ == "__main__":
    import sys
    from datetime import date as _date

    tgt = _date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        self_check(tgt)
    except Exception as e:                      # 사용자에게 원인을 그대로 보여준다
        print(f"\n❌ {type(e).__name__}: {e}")
        sys.exit(1)
