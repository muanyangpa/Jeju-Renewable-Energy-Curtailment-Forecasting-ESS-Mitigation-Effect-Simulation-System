package com.ami.anomaly.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * Python AI 서버 -> Spring Boot 응답 (통신규격 초안)
 */
@Getter
@Setter
@NoArgsConstructor
public class AiPredictResponse {
    private double anomalyScore;   // 0~1 사이 이상점수
    private String violationType;  // CONTRACT_POWER_EXCEEDED / IDLE_PERIOD_USAGE / RESERVE_POWER_MISUSE / SIMPLE_ANOMALY
    private String explanation;    // SHAP 등 XAI 설명 (JSON 문자열 등)
}
