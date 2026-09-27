"""KIM 표준화 NetCDF의 단일면 변수명을 찾는다 — 일사량(하향단파복사)이 목표.

[배경]
GRIB 조회 API는 403(별도 활용신청 필요)이지만 **NetCDF 지점 조회는 이미 승인돼 있다**.
게다가 전구 NC 파일명이 `glob_etc.2byte.1hr.ft024...nc`로 **1시간 간격**이라 GRIB(3시간)보다
우리 용도에 더 맞다. 막힌 것은 변수명뿐이다.

앞서 GRIB 약어(DSWRF·TCDC·UGRD·TMP)를 넣었더니 모두 'Variable not found'였다. API허브 예시가
`t2m`을 쓰는 것으로 보아 NC는 **소문자 관용 표기**를 쓴다. 그래서 후보를 넓게 시도한다.

성공하면 어떤 이름이 통하는지 한 줄로 드러나고, 전부 실패하면 '변수정보 PDF를 봐야 한다'가
확정된다 — 어느 쪽이든 결론이다.

실행: KMA_AUTH_KEY=... python scripts/probe_kma_nc_vars.py
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib
import ssl
import urllib.parse
import urllib.request

P = "https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_nc_pt_txt2_std"
OUT = pathlib.Path(os.environ.get(
    "KMA_PROBE_OUT",
    "/private/tmp/claude-501/-Users-yangjiho-Downloads-ai-server/"
    "5a4d5486-1c70-4374-854a-80249c7c756e/scratchpad/nwp"))

# 대조군(되는 것이 하나는 있어야 방법 자체가 맞는지 안다) + 일사량/운량/풍속 후보.
# 관용 표기, CF 표준명 줄임, ECMWF 스타일(ssrd), WRF 스타일(swdown)을 섞었다.
CANDIDATES = [
    "t2m",                                                    # 대조군 — API허브 예시에 나온 이름
    "swdown", "SWDOWN", "swd", "sw_down", "dswrf", "dswrf_sfc",
    "ssrd", "ssr", "rsds", "sw", "solar", "srad", "glob",     # 일사량 후보
    "tcc", "tcdc", "cld", "cloud",                            # 운량 후보
    "u10m", "v10m", "ws10m", "u80m", "wspd",                  # 풍속 후보
]
DATA_SETS = ["U", "E", "S"]   # U=단일면(예시). E/S는 다른 파일 세트가 있는지 탐색


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def probe(ctx, key, group, nwp, data, name, tmfc, xy) -> tuple[str, int, str]:
    q = urllib.parse.urlencode({
        "group": group, "nwp": nwp, "data": data, "name": name, "tmfc": tmfc,
        "hf": "24", "disp": "A", "help": "1", "lat": xy[0], "lon": xy[1], "authKey": key})
    try:
        with urllib.request.urlopen(f"{P}?{q}", timeout=60, context=ctx) as r:
            raw = r.read()
    except Exception as e:                                       # noqa: BLE001
        return "예외", 0, f"{type(e).__name__}: {e}"
    body = raw.decode("cp949", "replace")
    rows = [l for l in body.splitlines() if l.strip() and not l.startswith("#")]
    if rows:
        (OUT / f"nc_{group}_{data}_{name}.txt").write_bytes(raw)
        return "정상", len(rows), rows[0][:100]
    if "Variable not found" in body:
        return "변수없음", 0, ""
    errs = [l for l in body.splitlines() if "ERROR" in l]
    return "오류", 0, (errs[0][:70] if errs else "")


def main() -> None:
    key = os.environ.get("KMA_AUTH_KEY")
    if not key:
        raise SystemExit("KMA_AUTH_KEY 환경변수가 필요합니다")
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = _ctx()
    tmfc = (dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).strftime("%Y%m%d") + "00"
    jeju = ("33.5141", "126.5297")
    hits = []
    for group, nwp in (("KIMG", "NE57"), ("KIML", "L010")):
        for data in DATA_SETS:
            found_any = False
            for name in CANDIDATES:
                st, n, first = probe(ctx, key, group, nwp, data, name, tmfc, jeju)
                if st == "정상":
                    found_any = True
                    hits.append((group, nwp, data, name, n))
                    print(f"  ✅ {group}/{nwp} data={data} name={name:10} 행 {n:3d}  {first}")
                elif st in ("오류", "예외") and name == CANDIDATES[0]:
                    # 첫 후보에서 구조적 오류면 이 조합 자체가 잘못된 것이므로 건너뛴다
                    print(f"  -- {group}/{nwp} data={data} 건너뜀 ({st}: {first[:60]})")
                    break
            if not found_any:
                print(f"  ✗  {group}/{nwp} data={data}: {len(CANDIDATES)}개 후보 모두 변수없음")
    print()
    if hits:
        print("통하는 변수명:")
        for g, w, d, n, c in hits:
            print(f"  group={g} nwp={w} data={d} name={n}  (행 {c})")
    else:
        print("통하는 변수명을 찾지 못했습니다 — API허브의 '한국형 수치모델(KIM) 변수정보' PDF를")
        print("내려받아 단일면 변수명을 확인해야 합니다.")
    print(f"\ntmfc = {tmfc}\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
