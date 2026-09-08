package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * Python AI 서버 -> Spring Boot 응답 (통신규격 초안)
 */
@Getter
@Setter
@NoArgsConstructor
public class PredictionResponse {
    private double curtailmentProbability; // 출력제어 확률 (0~1)
    private Double excessGenerationMwh;    // 초과발전량 크기 — ESS 충전량 계산용 (04장)
}
