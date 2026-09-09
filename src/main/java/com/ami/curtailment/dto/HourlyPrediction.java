package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * 시간별 예측 결과 (통신규격 확정서 v1.0, 03장) - PredictionResponse.predictions 배열 요소
 */
@Getter
@Setter
@NoArgsConstructor
public class HourlyPrediction {
    private int hour;                       // 0~23
    private double curtailmentProbability;  // 출력제어 확률 (0.0~1.0)
    private String riskLevel;               // "HIGH" | "MEDIUM" | "LOW" (임계값은 AI 서버 관리)
    private double forecastGenerationMwh;   // AI 서버 변환모델 산출값 (대시보드 참고 표시용)
    private Double curtailmentMwh;          // 예상 출력제어량 - SOLAR는 항상 null
}
