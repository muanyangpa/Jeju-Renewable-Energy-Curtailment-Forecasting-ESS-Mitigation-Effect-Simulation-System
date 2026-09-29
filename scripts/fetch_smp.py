"""전력거래소 제주시범사업 '하루전 SMP' 연간 엑셀을 내려받는다.

왜 하루전 계열인가:
  제주는 2024-06 재생에너지 입찰제도 전환 이후 제어 실적이 공개되지 않는다. 대신 재생에너지
  입찰 상한이 0원/kWh이므로 SMP <= 0은 '재생에너지가 낙찰되지 못한 시간' = 시장 기반 출력제어의
  대리 지표가 된다(기후솔루션·PLANiT 보고서, 해줌 DR-XGB 논문 모두 같은 논리를 쓴다).

  SMP 계열이 두 가지 있고 반드시 '하루전'을 써야 한다.
    하루전(이 스크립트)  : 전일 공개 -> 하루 전 예측 시점에 이미 알 수 있다. 음수 196건.
    EPSIS 시간별(실시간) : 사후 확정값. 하루전과 5.6% 불일치하고 음수가 76건뿐이다.
  실시간 계열을 라벨로 쓰면 미래 정보가 새어 들어간다.

출처: 전력거래소 > 주요사업 > 제주시범사업 정보 > 계통한계가격(SMP) > 하루전
실행: python -m scripts.fetch_smp [연도 ...]     (기본 2024 2025 2026)
"""
from __future__ import annotations

import os
import re
import sys
import urllib.parse
import urllib.request
import http.cookiejar

PAGE = ("https://new.kpx.or.kr/bidSmpLfdDataDa.es"
        "?mid=a10406020100&device=pc&division=smpDataDa&gubun=today")
XLS = "https://new.kpx.or.kr/xlsxdownload.es?act=smpLfdDataDa&gubun=year&division=smpDataDa"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def fetch_year(year: int) -> str:
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    op.addheaders = [("User-Agent", "Mozilla/5.0")]
    html = op.open(PAGE, timeout=60).read().decode("utf-8", "replace")
    m = re.search(r'name="_csrf" value="([^"]+)"', html)
    if not m:
        raise RuntimeError("CSRF 토큰을 찾지 못했습니다 — 페이지 구조가 바뀐 것 같습니다")
    body = urllib.parse.urlencode({"yy": str(year), "issue_date": f"{year}-01-01",
                                   "_csrf": m.group(1)}).encode()
    data = op.open(XLS, data=body, timeout=180).read()
    if len(data) < 5000 or data[:2] != b"PK":
        raise RuntimeError(f"{year}: 엑셀이 아닌 응답({len(data)}바이트)")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"KPX_제주_하루전SMP_{year}.xlsx")
    with open(path, "wb") as f:
        f.write(data)
    return path


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2024, 2025, 2026]
    for y in years:
        try:
            print(f"  ✓ {fetch_year(y)}")
        except Exception as e:
            print(f"  ✗ {y}: {type(e).__name__} {e}")
