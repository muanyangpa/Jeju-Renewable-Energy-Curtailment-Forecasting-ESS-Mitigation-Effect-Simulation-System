"""
AMI 계량기 데이터 시뮬레이터

실제 스마트 계량기가 없으므로, 이 스크립트가 "계량기 역할"을 대신해서
일정 간격(기본 15분 -> 데모용으로는 초 단위로 빠르게 조정 가능)마다
Spring Boot Backend의 수신 엔드포인트로 전력사용량 데이터를 전송한다.

사용법:
    python simulate_ami.py --mode synthetic --consumer 1001 --interval 2
    python simulate_ami.py --mode csv --csv sgcc_sample.csv --consumer 1001 --interval 2

--mode synthetic : 정상 패턴 + 무작위 이상치를 섞은 가상 데이터 생성
--mode csv       : SGCC 등 실제 데이터셋 CSV를 읽어서 그대로 순서대로 전송
--interval       : 전송 간격(초). 실제로는 15분(900초)이 맞지만 데모/개발 중엔 짧게 설정
"""

import argparse
import csv
import random
import time
from datetime import datetime, timedelta

import requests

DEFAULT_BASE_URL = "http://localhost:8080"


def post_reading(base_url: str, consumer_no: str, usage_kwh: float, recorded_at: datetime):
    url = f"{base_url}/api/meters/{consumer_no}/readings"
    payload = {
        "usageKwh": round(usage_kwh, 3),
        "recordedAt": recorded_at.isoformat(timespec="seconds"),
    }
    try:
        res = requests.post(url, json=payload, timeout=5)
        status = "OK" if res.status_code == 200 else f"FAIL({res.status_code})"
    except requests.exceptions.RequestException as e:
        status = f"ERROR({e})"
    print(f"[{recorded_at}] consumer={consumer_no} usage={payload['usageKwh']}kWh -> {status}")


def generate_synthetic_value(hour: int, inject_anomaly: bool) -> float:
    """시간대별 평균 사용량 + 노이즈. 저녁(18~22시) 피크 패턴을 흉내냄."""
    base_by_hour = {
        h: (0.3 if h < 6 else 0.5 if h < 18 else 1.2 if h < 22 else 0.6)
        for h in range(24)
    }
    base = base_by_hour[hour]
    noise = random.uniform(-0.1, 0.1)
    value = max(0.0, base + noise)

    if inject_anomaly:
        # 계약전력초과 은폐형(비정상적으로 튀는 값) 흉내
        value *= random.uniform(2.5, 4.0)

    return value


def run_synthetic(base_url: str, consumer_no: str, interval: float, anomaly_rate: float):
    print(f"[시작] synthetic 모드 - consumer={consumer_no}, interval={interval}s, anomaly_rate={anomaly_rate}")
    current_time = datetime.now()
    while True:
        hour = current_time.hour
        inject_anomaly = random.random() < anomaly_rate
        value = generate_synthetic_value(hour, inject_anomaly)
        post_reading(base_url, consumer_no, value, current_time)

        current_time += timedelta(minutes=15)  # 실제 시간 흐름은 15분 단위로 가정
        time.sleep(interval)


def run_csv(base_url: str, consumer_no: str, interval: float, csv_path: str):
    """
    간단한 CSV 포맷 가정: recorded_at,usage_kwh 두 컬럼.
    SGCC 원본은 날짜가 컬럼으로 나열된 형태라 별도 전처리가 필요함(전처리 스크립트는 AI 담당자와 협의 후 추가).
    """
    print(f"[시작] csv 모드 - consumer={consumer_no}, interval={interval}s, file={csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            recorded_at = datetime.fromisoformat(row["recorded_at"])
            usage_kwh = float(row["usage_kwh"])
            post_reading(base_url, consumer_no, usage_kwh, recorded_at)
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="AMI 계량기 데이터 시뮬레이터")
    parser.add_argument("--mode", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--consumer", required=True, help="Meter.consumerNo (Backend에 미리 등록되어 있어야 함)")
    parser.add_argument("--interval", type=float, default=2.0, help="전송 간격(초), 데모용 기본 2초")
    parser.add_argument("--anomaly-rate", type=float, default=0.05, help="synthetic 모드에서 이상치 주입 확률")
    parser.add_argument("--csv", help="csv 모드일 때 읽을 파일 경로")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Backend 서버 주소")
    args = parser.parse_args()

    if args.mode == "synthetic":
        run_synthetic(args.base_url, args.consumer, args.interval, args.anomaly_rate)
    else:
        if not args.csv:
            parser.error("--mode csv 사용 시 --csv 경로가 필요합니다.")
        run_csv(args.base_url, args.consumer, args.interval, args.csv)


if __name__ == "__main__":
    main()
