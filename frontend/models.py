"""
공통 데이터 모델 — 위반유형 표시 이름/색상 매핑.
Backend의 ViolationType enum(CONTRACT_POWER_EXCEEDED 등)과 1:1 대응.
목록·상세·통계·지도 화면에서 공통으로 사용.
"""

VIOLATION_TYPE_DISPLAY = {
    "CONTRACT_POWER_EXCEEDED": {"label": "계약전력초과", "color": "#E74C3C"},
    "IDLE_PERIOD_USAGE": {"label": "휴지기간사용", "color": "#F39C12"},
    "RESERVE_POWER_MISUSE": {"label": "예비전원무단사용", "color": "#8E44AD"},
    "SIMPLE_ANOMALY": {"label": "단순이상", "color": "#7F8C8D"},
}


def label_for(violation_type: str) -> str:
    return VIOLATION_TYPE_DISPLAY.get(violation_type, {}).get("label", violation_type or "미판정")


def color_for(violation_type: str) -> str:
    return VIOLATION_TYPE_DISPLAY.get(violation_type, {}).get("color", "#95A5A6")
