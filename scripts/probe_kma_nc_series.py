"""시간별 시계열을 한 번에 받을 수 있는지 — hf 파라미터의 범위 지정 여부를 확인한다.

[왜 중요한가]
/predict는 24시간을 한 번에 받는다. hf에 단일 값만 넣을 수 있으면 요청이 24번 필요하고,
입찰마감(D-1 11시) 전에 여러 발전원·여러 단지를 돌리려면 부담이 된다. 범위 지정이 되면
요청 한 번으로 끝난다.

GRIB 시계열 API는 `ef=0,48,3`(시작,종료,간격) 형식을 쓴다 — NC API의 hf도 같은 관례를
받아들일 가능성이 있다. 여러 표기를 시도해 본다.

기준값: hf=24 단일 요청이 1행을 돌려준다(이미 확인). 범위가 통하면 여러 행이 온다.

실행: KMA_AUTH_KEY=... python scripts/probe_kma_nc_series.py
"""
from __future__ import annotations

import datetime as dt
import os
import pathlib
import re
import ssl
import urllib.parse
import urllib.request

P = "https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-kim_nc_pt_txt2_std"
OUT = pathlib.Path(os.environ.get(
    "KMA_PROBE_OUT",
    "/private/tmp/claude-501/-Users-yangjiho-Downloads-ai-server/"
    "5a4d5486-1c70-4374-854a-80249c7c756e/scratchpad/nwp"))

# 국지모델 단일면. 일사량(MJ/m2)과 허브높이 풍속을 함께 요청한다.
NAMES = "ACSWDNB,SWDDIR2,SWDDIF2,U80,V80,T2"
# hf 표기 후보. GRIB의 ef 관례(시작,종료,간격)와 그 변형들.
HF_FORMS = ["24", "1,24,1", "0,24,1", "1-24", "1:24", "1,24", "0,48,3", "all", ""]


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def parse(body: str) -> list[tuple[str, str, str, str]]:
    """(TMEF, VARN, VALUS, NAME). 단위에 공백이 있어(예: 'U80(m s-1)') 고정 열로 못 자른다 —
    앞 5개 토큰을 취하고 나머지를 이름으로 붙인다."""
    out = []
    for l in body.splitlines():
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        f = s.split()
        if len(f) >= 6 and re.fullmatch(r"\d{10}", f[0]):
            out.append((f[1], f[2], f[4], " ".join(f[5:])))
    return out


def main() -> None:
    key = os.environ.get("KMA_AUTH_KEY")
    if not key:
        raise SystemExit("KMA_AUTH_KEY 환경변수가 필요합니다")
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = _ctx()
    tmfc = (dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).strftime("%Y%m%d") + "00"
    print(f"tmfc={tmfc} · 국지 1.5km · 변수 {NAMES}\n")
    print(f"  {'hf 표기':10} {'행수':>5}  {'서로 다른 예측시각':>18}  비고")

    best = None
    for hf in HF_FORMS:
        q = urllib.parse.urlencode({
            "group": "KIML", "nwp": "L010", "data": "U", "name": NAMES, "tmfc": tmfc,
            "hf": hf, "disp": "A", "help": "1",
            "lat": "33.5141", "lon": "126.5297", "authKey": key})
        try:
            with urllib.request.urlopen(f"{P}?{q}", timeout=120, context=ctx) as r:
                body = r.read().decode("cp949", "replace")
        except Exception as e:                                    # noqa: BLE001
            print(f"  {hf or '(빈값)':10} {'-':>5}  {'-':>18}  {type(e).__name__}: {e}")
            continue
        (OUT / f"nc_series_hf_{hf.replace(',', '_') or 'blank'}.txt").write_text(body, encoding="utf-8")
        rows = parse(body)
        tmefs = sorted({r[0] for r in rows})
        err = [l for l in body.splitlines() if "ERROR" in l]
        note = (err[0].split("ERROR :")[-1].strip()[:44] if err else "")
        print(f"  {hf or '(빈값)':10} {len(rows):5d}  {len(tmefs):18d}  {note}")
        if len(tmefs) > 1 and (best is None or len(tmefs) > best[1]):
            best = (hf, len(tmefs), rows)

    print()
    if best:
        hf, n, rows = best
        print(f"✅ 범위 지정이 됩니다 — hf={hf} 로 예측시각 {n}개를 한 번에 받습니다.")
        var = "ACSWDNB"
        ser = [(r[0], r[2]) for r in rows if r[3].startswith(var)]
        print(f"   {var} 시계열 (앞 12개):")
        for tmef, val in ser[:12]:
            print(f"     {tmef}  {float(val):8.3f}")
    else:
        print("범위 지정이 안 됩니다 — hf를 1~24로 24번 요청해야 합니다.")
        print("  24번이면 순차 호출로 수십 초가 걸립니다. 입찰마감(D-1 11시) 전 산출에는")
        print("  문제없지만, 병렬 호출이나 캐시를 두는 편이 낫습니다.")
    print(f"\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
