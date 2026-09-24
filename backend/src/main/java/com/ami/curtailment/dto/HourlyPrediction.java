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
    // 출력제어 확률 (0.0~1.0). sigmoid 보정된 값이라 보정 전 모델과 '크기'가 다르다.
    // 0.5 같은 고정 임계값이나 예전 등급 구간을 그대로 쓰지 말 것 - 탐지 기준은 AI 서버
    // README '운영 임계값'을 따르고, 모델 재학습 시마다 재산정해야 한다. (2026-09-24)
    private double curtailment_probability;

    // 제어량 기댓값 = curtailment_probability x E[제어량|제어 발생]. WIND + demand_forecast_mw가
    // 있을 때만 값 존재, SOLAR는 항상 null. 기댓값이라 개별 시간의 제어량 크기가 아니므로
    // /ess/simulate의 hourly_curtailment_mwh로 넘기지 말 것 (응답 note의 경고 참고).
    private Double expected_curtailment_mwh;
}
