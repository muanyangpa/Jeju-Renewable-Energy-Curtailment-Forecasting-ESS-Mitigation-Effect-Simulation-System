"""KIM 표준화 NetCDF 단일면 변수명 일괄 탐색 — 일사량을 찾는다.

[알아낸 것]
KIMG/NE57/data=U 에서 t2m·u10m·v10m·u80m 이 통했다. 응답 형식이

    TMFC       TMEF       VARN  LEVEL  VALUS        NAME
    2026092600 2026092700   25      0  2.96090e+02  t2m(K)

이고, VARN이 파일 내부 인덱스다(t2m=25, u10m=20, v10m=21, u80m=22). 그리고 name 파라미터는
쉼표로 여러 개를 받는다(API허브 예시 `name=U,W`). 그래서 후보를 묶어 한 번에 던지면
**존재하는 것만 행으로 돌아온다** — 요청 수를 줄이면서 전수 탐색에 가까워진다.

명명 규칙이 `변수+고도+m`(t2m, u10m, u80m)이라 지표 플럭스도 관용 표기일 가능성이 높다.
일사량 어휘를 넓게 깐다: WRF(swdown/swdnb), ECMWF(ssrd/ssr), CF(rsds), GRIB 소문자(dswrf),
그리고 축약형(sw, swd, gsw, rad, srad, solrad, swsfc, dswsfc).

실행: KMA_AUTH_KEY=... python scripts/probe_kma_nc_batch.py
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

KNOWN = ["t2m", "u10m", "v10m", "u80m"]          # 통하는 것 — 대조군으로 매 묶음에 넣는다
BATCHES = {
    "일사·복사": ["swdown", "swdnb", "swdn", "swd", "sw", "gsw", "dswrf", "dswsfc", "swsfc",
                "ssrd", "ssr", "rsds", "rad", "srad", "solrad", "sol", "glob", "ghi",
                "lwdown", "lwdn", "rlds", "netsw", "nswrs"],
    "운량·습도": ["tcc", "tcdc", "cld", "cloud", "lcc", "mcc", "hcc", "rh2m", "q2m", "td2m",
                "rh", "spfh"],
    "바람·기타": ["v80m", "w80m", "ws80m", "u100m", "v100m", "gust", "u925", "psfc", "pmsl",
                "slp", "prcp", "rain", "apcp", "tsfc", "skt", "sst"],
}


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def ask(ctx, key, group, nwp, data, names, tmfc, hf="24") -> tuple[list[str], str]:
    q = urllib.parse.urlencode({
        "group": group, "nwp": nwp, "data": data, "name": ",".join(names), "tmfc": tmfc,
        "hf": hf, "disp": "A", "help": "1", "lat": "33.5141", "lon": "126.5297", "authKey": key})
    with urllib.request.urlopen(f"{P}?{q}", timeout=90, context=ctx) as r:
        body = r.read().decode("cp949", "replace")
    rows = [l.strip() for l in body.splitlines() if l.strip() and not l.startswith("#")]
    return rows, body


def main() -> None:
    key = os.environ.get("KMA_AUTH_KEY")
    if not key:
        raise SystemExit("KMA_AUTH_KEY 환경변수가 필요합니다")
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = _ctx()
    tmfc = (dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).strftime("%Y%m%d") + "00"

    print(f"tmfc={tmfc} · 제주 (33.5141, 126.5297) · KIMG/NE57/data=U\n")
    found = {}
    for label, cands in BATCHES.items():
        names = KNOWN + cands
        try:
            rows, body = ask(ctx, key, "KIMG", "NE57", "U", names, tmfc)
        except Exception as e:                                   # noqa: BLE001
            print(f"[{label}] {type(e).__name__}: {e}")
            continue
        (OUT / f"nc_batch_{label}.txt").write_text(body, encoding="utf-8")
        got = {}
        for r in rows:
            f = r.split()
            if len(f) >= 6:
                got[f[-1]] = (f[2], f[-2])        # NAME -> (VARN, VALUS)
        new = {k: v for k, v in got.items() if not any(k.startswith(x) for x in KNOWN)}
        print(f"[{label}] 요청 {len(cands)}개 -> 수신 {len(got)}개 (대조군 제외 {len(new)}개)")
        for n, (varn, val) in sorted(new.items(), key=lambda x: int(x[1][0])):
            print(f"    varn={varn:>4}  {n:22} = {val}")
        found.update(new)

    print()
    if found:
        rad = {k: v for k, v in found.items()
               if any(s in k.lower() for s in ("sw", "rad", "sol", "ghi", "lw"))}
        print(f"✅ 새로 찾은 변수 {len(found)}개 · 그중 복사 관련 {len(rad)}개")
        for n, (varn, val) in rad.items():
            print(f"    {n:22} varn={varn}  값={val}")
        if not rad:
            print("   복사 관련은 없습니다 — 단일면 NC에 일사량이 없을 수 있습니다.")
    else:
        print("대조군 외에는 아무것도 없습니다.")
    print(f"\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
