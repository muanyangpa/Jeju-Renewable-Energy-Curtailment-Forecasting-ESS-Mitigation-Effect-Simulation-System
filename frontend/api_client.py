"""
Spring Boot Backend와 통신하는 API 클라이언트.
Frontend(Streamlit)는 이 모듈을 통해서만 Backend를 호출한다.
Python AI 서버는 Backend 뒤에 숨어있어 Frontend가 직접 호출하지 않음(IF-01 규격).
"""

import requests

BASE_URL = "http://localhost:8080"
TIMEOUT = 5


def get_meters():
    """전체 세대(계량기) 목록 조회"""
    try:
        res = requests.get(f"{BASE_URL}/api/meters", timeout=TIMEOUT)
        res.raise_for_status()
        return res.json(), None
    except requests.exceptions.RequestException as e:
        return [], f"Backend 연결 실패: {e}"


def get_anomalies(meter_id: int):
    """특정 세대의 이상탐지 결과 조회"""
    try:
        res = requests.get(f"{BASE_URL}/api/anomalies/{meter_id}", timeout=TIMEOUT)
        res.raise_for_status()
        return res.json(), None
    except requests.exceptions.RequestException as e:
        return [], f"Backend 연결 실패: {e}"


def get_recent_readings(consumer_no: str):
    """특정 세대의 최근 측정값(시계열) 조회"""
    try:
        res = requests.get(f"{BASE_URL}/api/meters/{consumer_no}/readings/recent", timeout=TIMEOUT)
        res.raise_for_status()
        return res.json(), None
    except requests.exceptions.RequestException as e:
        return [], f"Backend 연결 실패: {e}"


def register_meter(consumer_no: str, contract_power_kw: float,
                    idle_requested: bool, reserve_power_contract: bool):
    """세대(계량기) 신규 등록 (테스트/시연용 데이터 준비 용도)"""
    payload = {
        "consumerNo": consumer_no,
        "contractPowerKw": contract_power_kw,
        "idleRequested": idle_requested,
        "reservePowerContract": reserve_power_contract,
    }
    try:
        res = requests.post(f"{BASE_URL}/api/meters", json=payload, timeout=TIMEOUT)
        res.raise_for_status()
        return res.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Backend 연결 실패: {e}"
