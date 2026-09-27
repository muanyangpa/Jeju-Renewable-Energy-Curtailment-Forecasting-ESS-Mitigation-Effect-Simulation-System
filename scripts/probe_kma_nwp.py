"""기상청 API허브 수치모델(KIM) 탐색 — DSWRF(지표면 하향단파복사)를 뽑을 수 있는지 확인한다.

[왜 필요한가]
기상청 단기예보에는 일사량이 없어 현재는 청천일사량 x 감쇠계수로 추정한다
(app/solar_radiation.py). 그런데 nph-nwp_header로 확인해 보니 KIM 전구모델 단일면 GRIB에
DSWRF(Downward Short-Wave Radiation Flux, 지표면, W/m^2)가 실재한다. 이 스크립트는 그것을
지점 시계열로 실제로 받을 수 있는지 확인한다.

[변수 지정 방식] GRIB 조회 API는 varn = discipline x 100000 + category x 1000 + parameterNumber.
  DSWRF 지표면   D=0 C=4 P=7  -> 4007
  TCDC  전운량   D=0 C=6 P=1  -> 6001
  UGRD  동서바람 D=0 C=2 P=2  -> 2002   (국지모델은 level=80, 허브높이에 가깝다)
  TMP   기온     D=0 C=0 P=0  -> 0

[NC와 GRIB의 차이] nph-kim_nc_pt_txt2_std(NetCDF)는 GRIB 약어를 받지 않는다 — DSWRF·TCDC·
UGRD·TMP 모두 'Variable not found'였다. 그래서 GRIB 계열 API를 쓴다.

실행: KMA_AUTH_KEY=... python scripts/probe_kma_nwp.py
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib
import ssl
import urllib.error
import urllib.parse
import urllib.request

OUT = pathlib.Path(os.environ.get(
    "KMA_PROBE_OUT",
    "/private/tmp/claude-501/-Users-yangjiho-Downloads-ai-server/"
    "5a4d5486-1c70-4374-854a-80249c7c756e/scratchpad/nwp"))

PT = "https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_grib_pt_txt1"
TS = "https://apihub.kma.go.kr/api/typ06/url/kim_grib_pt_tmfc.php"

# X, Y는 앞선 NC 응답이 알려준 제주(Lon 126.5 / Lat 33.5) 격자점이다.
JEJU_GLOBAL = ("1519", "1482")
JEJU_LOCAL = ("637", "280")


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def main() -> None:
    key = os.environ.get("KMA_AUTH_KEY")
    if not key:
        raise SystemExit("KMA_AUTH_KEY 환경변수가 필요합니다")
    OUT.mkdir(parents=True, exist_ok=True)
    tmfc = (dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).strftime("%Y%m%d") + "00"
    gx, gy = JEJU_GLOBAL
    lx, ly = JEJU_LOCAL

    jobs = [
        ("grib_g_DSWRF_ts", TS, {"group": "KIMG", "nwp": "g128", "data": "U", "varn": "4007",
                                 "tmfc": tmfc, "ef": "0,48,3", "X": gx, "Y": gy, "level": "0"}),
        ("grib_g_DSWRF_pt", PT, {"group": "KIMG", "nwp": "g128", "data": "U", "varn": "4007",
                                 "tmfc": tmfc, "hf": "24", "X": gx, "Y": gy}),
        ("grib_g_TCDC_ts", TS, {"group": "KIMG", "nwp": "g128", "data": "U", "varn": "6001",
                                "tmfc": tmfc, "ef": "0,48,3", "X": gx, "Y": gy, "level": "0"}),
        ("grib_l_UGRD80_ts", TS, {"group": "KIML", "nwp": "l010", "data": "U", "varn": "2002",
                                  "tmfc": tmfc, "ef": "0,48,3", "X": lx, "Y": ly, "level": "80"}),
        ("grib_l_TMP2_ts", TS, {"group": "KIML", "nwp": "l010", "data": "U", "varn": "0",
                                "tmfc": tmfc, "ef": "0,48,3", "X": lx, "Y": ly, "level": "2"}),
    ]
    ctx = _ctx()
    for tag, base, params in jobs:
        q = urllib.parse.urlencode(dict(params, disp="A", help="1", authKey=key))
        try:
            with urllib.request.urlopen(f"{base}?{q}", timeout=90, context=ctx) as r:
                raw = r.read()
        except Exception as e:                                   # noqa: BLE001
            print(f"{tag:18} {type(e).__name__}: {e}")
            continue
        (OUT / f"{tag}.txt").write_bytes(raw)
        body = raw.decode("cp949", "replace")
        data = [l for l in body.splitlines() if l.strip() and not l.startswith("#")]
        errs = [l for l in body.splitlines() if "ERROR" in l]
        status = ("오류: " + errs[0][:55]) if errs else "정상"
        print(f"{tag:18} {len(raw):7d}B  데이터행 {len(data):4d}  {status}")
        if data:
            print(f"      첫 행: {data[0][:110]}")
    print(f"\ntmfc = {tmfc}\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
