package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * AI 서버 응답의 hourly 배열 요소. app/schemas.py 실제 필드명 그대로.
 */
@Getter
@Setter
@NoArgsConstructor
public class HourlyPrediction {
    private int hour;                          // 1~24
    private double generation_forecast_mwh;    // 컨버터가 산출한 발전량 예측치
    private double curtailment_probability;    // 출력제어 확률 (0.0~1.0)
    private Double expected_curtailment_mwh;   // 예상 출력제어량 - WIND만 값 존재, SOLAR는 항상 null
}
