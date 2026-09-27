"""KIM 국지예보모델(1.5km)에서 시간별 일사량과 허브높이 풍속을 받아온다.

[왜 필요한가]
기상청 단기예보에는 일사량이 없어 지금은 청천일사량 x 감쇠계수로 추정한다
(app/solar_radiation.py). KIM 국지모델 표준화 NetCDF에는 실제 예보값이 있다.

  SWDDIR2  직달 일사     W/m^2   순간값
  SWDDIF2  산란 일사     W/m^2   순간값
  ACSWDNB  누적 하향단파  MJ/m^2  모델 시작부터의 누적 -> 차분이 필요
  U80/V80  80m 풍속      m/s     허브높이에 가깝다

**직달 + 산란**을 쓴다. 순간값이라 차분이 필요 없고, 수평면 전일사량이 바로 나온다.
ACSWDNB 차분은 교차검증용으로만 쓴다(hf=0에서 0, hf=24에서 9.32로 단조 증가하는 것을 확인했다).

[시각 변환이 이 모듈의 핵심이다]
  tmfc  모델 분석시각, **UTC**
  hf    tmfc 기준 경과 시간
  이 저장소의 `h시`는 **KST이고 그 시간 구간의 끝**이다(13시 = 12:00~13:00).
  따라서 거래일 D의 h시에 해당하는 hf는

      hf = (D일 h시 KST - tmfc UTC) 시간차 = (KST -> UTC는 -9h)

  거래일 24시는 D+1일 00시이므로 마지막 hf가 가장 크다.

[자료 가용 시점 — 하루전시장 입찰마감이 D-1 11시 KST다]
  KIM 분석시각은 00/06/12/18 UTC이고 생산에 수 시간 걸린다. D-1 00 UTC(= D-1 09시 KST)
  자료는 마감 후에야 나올 수 있으므로, 안전하게 **D-2 18 UTC**(= D-1 03시 KST)를 기본으로 둔다.
  실제 가용 시점은 운영에서 확인해야 한다 — scripts/probe_kma_kim_avail.py 참고.

[요청 수] hf에 범위를 넣을 수 없다(1,24,1 / 1-24 / all 모두 첫 숫자만 읽는다). 그래서 24번
호출하고, 순차로는 느려서 스레드로 병렬 처리한다.
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib
import ssl
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

ENDPOINT = "https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_nc_pt_txt2_std"
GROUP, NWP, DATA = "KIML", "L010", "U"
NAMES = ("SWDDIR2", "SWDDIF2", "ACSWDNB", "U80", "V80", "T2")
W_TO_MJ_PER_HOUR = 0.0036
KST = dt.timezone(dt.timedelta(hours=9))

# 기본 분석시각 = D-2 18 UTC (= D-1 03시 KST). 하루전시장 입찰마감이 D-1 11시 KST(= D-1 02 UTC)인데
# KIM은 분석 후 생산에 수 시간 걸린다. D-1 00 UTC 자료는 마감 후에야 나오므로 한 사이클 앞을 쓴다.
# D일 00시 KST 기준으로 21시간 거슬러 올라가면 D-1 03시 KST = D-2 18 UTC다.
DEFAULT_TMFC_OFFSET_HOURS = 21


@dataclass
class KimHour:
    hour: int                 # 1~24 (KST, 구간의 끝)
    ghi_mj: float             # 직달 + 산란, MJ/m^2
    direct_w: float
    diffuse_w: float
    wind80: float             # sqrt(U80^2 + V80^2), m/s
    temp_c: float
    acc_mj: float             # ACSWDNB 누적값 (차분 교차검증용)


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def default_tmfc(target: dt.date) -> str:
    """거래일 target에 쓸 기본 분석시각 (UTC, YYYYMMDDHH). D-2 18 UTC로 떨어진다."""
    base_kst = dt.datetime.combine(target, dt.time(0), tzinfo=KST)
    t = (base_kst - dt.timedelta(hours=DEFAULT_TMFC_OFFSET_HOURS)).astimezone(dt.timezone.utc)
    # 분석시각은 00/06/12/18 UTC뿐이므로 내림한다
    return t.replace(hour=t.hour // 6 * 6).strftime("%Y%m%d%H")


def hf_for(target: dt.date, hour: int, tmfc: str) -> int:
    """거래일 target의 hour시(KST, 구간 끝)에 해당하는 hf. hour=24는 다음날 00시다."""
    t0 = dt.datetime.strptime(tmfc, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)
    t1 = dt.datetime.combine(target, dt.time(0), tzinfo=KST) + dt.timedelta(hours=hour)
    return int((t1 - t0).total_seconds() // 3600)


def _parse(body: str) -> dict[str, float]:
    """응답 -> {변수명: 값}. 단위에 공백이 있어(U80(m s-1)) 앞 5개 토큰만 고정으로 쓴다."""
    out: dict[str, float] = {}
    for line in body.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        f = s.split()
        if len(f) >= 6 and len(f[0]) == 10 and f[0].isdigit():
            name = " ".join(f[5:]).split("(")[0]
            try:
                out[name] = float(f[4])
            except ValueError:
                continue
    return out


def load_sample(path: str) -> dict[int, dict[str, float]]:
    """오프라인 테스트용. 한 파일에 `#hf=<n>` 구분선으로 여러 응답을 이어 붙인 형식을 읽는다."""
    blocks: dict[int, dict[str, float]] = {}
    cur, buf = None, []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#hf="):
            if cur is not None:
                blocks[cur] = _parse("\n".join(buf))
            cur, buf = int(line[4:].strip()), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        blocks[cur] = _parse("\n".join(buf))
    return blocks


def _one(key: str, ctx, tmfc: str, hf: int, lat: float, lon: float,
         timeout: float) -> dict[str, float]:
    q = urllib.parse.urlencode({
        "group": GROUP, "nwp": NWP, "data": DATA, "name": ",".join(NAMES), "tmfc": tmfc,
        "hf": str(hf), "disp": "A", "help": "1",
        "lat": f"{lat}", "lon": f"{lon}", "authKey": key})
    with urllib.request.urlopen(f"{ENDPOINT}?{q}", timeout=timeout, context=ctx) as r:
        return _parse(r.read().decode("cp949", "replace"))


def fetch_day(target: dt.date, lat: float = 33.5141, lon: float = 126.5297,
              tmfc: str | None = None, key: str | None = None, workers: int = 8,
              timeout: float = 60.0, sample_path: str | None = None,
              errors: list | None = None) -> list[KimHour]:
    """거래일 24시간의 일사량·풍속을 받아온다. hf를 24번 호출하며 병렬로 처리한다.

    누락된 시각이 있으면 그 시각을 빼고 돌려준다 — 호출자가 24개인지 확인해야 한다.
    """
    tmfc = tmfc or default_tmfc(target)
    hours = list(range(1, 25))
    # hf=h-1도 함께 받아 ACSWDNB 차분을 낼 수 있게 한다(교차검증용)
    need = sorted({hf_for(target, h, tmfc) for h in hours}
                  | {hf_for(target, h, tmfc) - 1 for h in hours})

    if sample_path:
        raw = load_sample(sample_path)
    else:
        key = key or os.environ.get("KMA_AUTH_KEY")
        if not key:
            raise RuntimeError("KMA_AUTH_KEY 환경변수가 필요합니다")
        ctx = _ctx()

        def job(hf: int):
            try:
                return hf, _one(key, ctx, tmfc, hf, lat, lon, timeout)
            except Exception as e:                               # noqa: BLE001
                if errors is not None:
                    errors.append(f"hf={hf} {type(e).__name__}: {e}")
                return hf, {}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            raw = dict(ex.map(job, need))

    out: list[KimHour] = []
    for h in hours:
        hf = hf_for(target, h, tmfc)
        v = raw.get(hf) or {}
        if not {"SWDDIR2", "SWDDIF2"} <= v.keys():
            continue
        prev = (raw.get(hf - 1) or {}).get("ACSWDNB")
        acc = v.get("ACSWDNB", float("nan"))
        out.append(KimHour(
            hour=h,
            ghi_mj=round((v["SWDDIR2"] + v["SWDDIF2"]) * W_TO_MJ_PER_HOUR, 4),
            direct_w=v["SWDDIR2"], diffuse_w=v["SWDDIF2"],
            wind80=round(float(np.hypot(v.get("U80", np.nan), v.get("V80", np.nan))), 2),
            temp_c=round(v.get("T2", float("nan")) - 273.15, 1),
            acc_mj=round(acc - prev, 4) if prev is not None and acc == acc else float("nan"),
        ))
    return out


def radiation_override(target: dt.date, note: list | None = None,
                       **kw) -> dict[int, float] | None:
    """{시각: 일사량 MJ/m^2}. 24시간을 다 받지 못하면 None — 그러면 추정 모델로 폴백한다.

    부분적으로만 받은 값을 섞으면 어느 시각이 예보이고 어느 시각이 추정인지 알 수 없게 되므로,
    전부 받거나 전부 폴백한다. 폴백 자체는 정상 동작이지만 **사유를 삼키면 안 된다** —
    미승인·오타·분석시각 미생산을 조용한 성능 저하로 넘기게 된다. note에 사유를 남긴다.
    """
    errs: list[str] = []
    try:
        rows = fetch_day(target, errors=errs, **kw)
    except Exception as e:                                       # noqa: BLE001
        if note is not None:
            note.append(f"{type(e).__name__}: {e}")
        return None
    if len(rows) != 24:
        if note is not None:
            note.append(f"24시간 중 {len(rows)}개만 수신"
                        + (f" — 첫 오류: {errs[0]}" if errs else ""))
        return None
    return {r.hour: r.ghi_mj for r in rows}
