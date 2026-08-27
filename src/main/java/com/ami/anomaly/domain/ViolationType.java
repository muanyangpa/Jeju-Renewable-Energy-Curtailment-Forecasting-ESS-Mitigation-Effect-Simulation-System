package com.ami.anomaly.domain;

/**
 * 위반유형 4분류 (한전 전기공급약관 제44조 기준)
 * - 조사 결과 문서 "위반유형_판정기준_조사결과" 참고
 */
public enum ViolationType {
    CONTRACT_POWER_EXCEEDED,   // 계약전력초과: 저압 월 450시간 초과, 고압 15분단위 최대수요전력제
    IDLE_PERIOD_USAGE,         // 휴지기간사용: 휴지신청 상태인데 실사용 발생
    RESERVE_POWER_MISUSE,      // 예비전원무단사용: 예비전력 계약 상태에서 상시 사용
    SIMPLE_ANOMALY             // 단순이상: 위 3개로 명확히 분류 안 되는 이상 패턴
}
