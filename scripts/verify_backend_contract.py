"""Java Backend DTO와 AI 서버 응답이 맞는지 실제 HTTP로 확인한다.

docs/통신규격_확정서_v1.1.md 05장의 근거. tests/test_api_contract.py는 TestClient(인프로세스)로
도는 반면 이 스크립트는 **실제로 띄운 서버에 HTTP 요청**을 보내므로, 직렬화·인코딩·포트까지
포함해 검증한다. Java 쪽에서 같은 JSON을 보내면 같은 응답이 와야 한다.

  .venv/bin/python -m uvicorn app.main:app --port 8000 &
  .venv/bin/python scripts/verify_backend_contract.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"

# backend/src/main/java/com/ami/curtailment/dto/*.java 의 필드명 그대로
JAVA_RESPONSE_KEYS = {"energy_type", "region", "target_date", "hourly", "model_used",
                      "operational_threshold", "operational_threshold_reliable", "note"}
JAVA_HOURLY_KEYS = {"hour", "generation_forecast_mwh", "curtailment_probability",
                    "expected_curtailment_mwh"}


def call(path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def weather(temp: float | None = 15.0) -> list[dict]:
    return [{"hour": h, "solar_rad": 0.5, "temp": temp, "cloud": 4.0, "wind_speed": 6.0}
            for h in range(1, 25)]


def main() -> int:
    fails = []

    s, j = call("/predict", {"energy_type": "wind", "region": "남원읍",
                             "target_date": "2026-10-01", "weather": weather(),
                             "demand_forecast_mw": [620.0] * 24})
    print(f"[1] 응답 키 일치 (Java PredictionResponse)")
    if s != 200:
        print(f"    ❌ HTTP {s}: {j}"); return 1
    d1, d2 = set(j) - JAVA_RESPONSE_KEYS, JAVA_RESPONSE_KEYS - set(j)
    ok = not d1 and not d2
    print(f"    {'✅' if ok else '❌'} 서버에만: {d1 or '없음'} / Java에만: {d2 or '없음'}")
    fails += [] if ok else ["응답 키"]

    print(f"[2] hourly 키 일치 (Java HourlyPrediction)")
    h1, h2 = set(j["hourly"][0]) - JAVA_HOURLY_KEYS, JAVA_HOURLY_KEYS - set(j["hourly"][0])
    ok = not h1 and not h2
    print(f"    {'✅' if ok else '❌'} 서버에만: {h1 or '없음'} / Java에만: {h2 or '없음'}")
    fails += [] if ok else ["hourly 키"]

    n = len(j.get("note") or "")
    print(f"[3] note 길이 {n}자 — VARCHAR(255) 초과 여부")
    print(f"    {'⚠ 초과 (TEXT 필요)' if n > 255 else '✅ 255 이내'}")

    print(f"[4] model_used / operational_threshold 수신")
    print(f"    model_used={j['model_used']}")
    print(f"    threshold={j['operational_threshold']} reliable={j['operational_threshold_reliable']}")
    fails += [] if j.get("model_used") else ["model_used 누락"]

    print(f"[5] temp를 null로 보내면 422여야 한다 (Java가 double이면 0.0이 되어 통과한다)")
    s2, j2 = call("/predict", {"energy_type": "solar", "region": "남원읍",
                               "target_date": "2026-10-01", "weather": weather(None)})
    ok = s2 == 422 and j2.get("error_code") == "MISSING_REQUIRED_FIELD"
    print(f"    {'✅' if ok else '❌'} HTTP {s2} {j2.get('error_code')}")
    fails += [] if ok else ["null temp 검증"]

    print(f"[6] temp=0.0(Jackson 기본값)이 조용히 통과하는 영향")
    tot = {}
    for t in (15.0, 0.0):
        _, jj = call("/predict", {"energy_type": "solar", "region": "남원읍",
                                  "target_date": "2026-10-01", "weather": weather(t)})
        tot[t] = sum(x["generation_forecast_mwh"] for x in jj["hourly"])
    gap = (tot[0.0] / tot[15.0] - 1) * 100
    print(f"    temp=15 -> {tot[15.0]:,.1f} MWh / temp=0 -> {tot[0.0]:,.1f} MWh  ({gap:+.0f}%)")
    print(f"    ⚠ Java WeatherHour.temp가 Double이어야 하는 이유 (v1.1 05장)")

    print(f"\n{'✅ 계약 일치' if not fails else '❌ 불일치: ' + ', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    try:
        urllib.request.urlopen(BASE + "/health", timeout=5)
    except Exception:
        sys.exit(f"AI 서버가 {BASE}에 없습니다 — "
                 f".venv/bin/python -m uvicorn app.main:app --port 8000 으로 먼저 띄우세요")
    sys.exit(main())
