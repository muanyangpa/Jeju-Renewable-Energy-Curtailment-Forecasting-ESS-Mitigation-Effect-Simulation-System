"""변수정보 PDF에서 확인한 이름으로 실제 값을 받아온다 — 일사량·허브높이 풍속.

[PDF에서 확인한 것] '한국형 수치모델(KIM) 변수정보'는 모델마다 **다른 명명 체계**를 쓴다.
전구는 소문자 관용명(t2m, dswrsfc), 국지는 대문자 GRIB식(T2, SWDDIR2). 앞서 국지모델에서
t2m조차 실패한 이유가 이것이었다.

  전구/단일면  dswrsfc  지표면 하향단파복사   W/m2      tcld  전운량 0~1
  국지/단일면  ACSWDNB  누적 하향단파 플럭스  MJ/m2  <- 컨버터와 같은 단위
              SWDDIR2  직달                W/m2
              SWDDIF2  산란                W/m2
              SWDDNI2  법선직달             W/m2
              U80/V80  80m 풍속            m/s    <- 허브높이. 140/220m도 있다

국지모델(1.5km)이 전구(12km)보다 해상도가 높고 직달·산란 분해까지 있어 태양광에 유리하다.
풍력은 U80/V80이 ASOS 지상 10m를 대체할 후보다.

실행: KMA_AUTH_KEY=... python scripts/probe_kma_nc_confirm.py
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

# (레이블, group, nwp, data, 변수 목록)
TARGETS = [
    ("전구 단일면", "KIMG", "NE57", "U",
     ["dswrsfc", "dswrtoa", "rss", "tcld", "lcld", "mcld", "hcld", "t2m", "u80m", "v80m"]),
    ("국지 단일면", "KIML", "L010", "U",
     ["ACSWDNB", "SWDDIR2", "SWDDIF2", "SWDDNI2", "T2", "U10", "V10",
      "U80", "V80", "U140", "V140", "LCDC", "MCDC", "HCDC", "PSFC", "RH2"]),
]


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
    ctx = _ctx()
    tmfc = (dt.datetime.now(dt.UTC) - dt.timedelta(days=1)).strftime("%Y%m%d") + "00"
    print(f"tmfc={tmfc} · 제주 (33.5141, 126.5297)\n")

    for label, group, nwp, data, names in TARGETS:
        q = urllib.parse.urlencode({
            "group": group, "nwp": nwp, "data": data, "name": ",".join(names),
            "tmfc": tmfc, "hf": "24", "disp": "A", "help": "1",
            "lat": "33.5141", "lon": "126.5297", "authKey": key})
        try:
            with urllib.request.urlopen(f"{P}?{q}", timeout=90, context=ctx) as r:
                body = r.read().decode("cp949", "replace")
        except Exception as e:                                   # noqa: BLE001
            print(f"[{label}] {type(e).__name__}: {e}\n")
            continue
        (OUT / f"nc_confirm_{group}.txt").write_text(body, encoding="utf-8")
        rows = [l.split() for l in body.splitlines() if l.strip() and not l.startswith("#")]
        got = {r[-1]: (r[2], r[-2]) for r in rows if len(r) >= 6}
        miss = [n for n in names if not any(g.startswith(n) for g in got)]
        print(f"[{label}] 요청 {len(names)}개 -> 수신 {len(got)}개")
        for nm, (varn, val) in sorted(got.items(), key=lambda x: int(x[1][0])):
            print(f"    varn={varn:>4}  {nm:24} = {val}")
        if miss:
            print(f"    없음: {', '.join(miss)}")
        print()

    print("일사량이 나왔으면 다음은 시간별 시계열(hf를 1~24로 바꿔 반복)이다.")
    print(f"저장 위치: {OUT}")


if __name__ == "__main__":
    main()
