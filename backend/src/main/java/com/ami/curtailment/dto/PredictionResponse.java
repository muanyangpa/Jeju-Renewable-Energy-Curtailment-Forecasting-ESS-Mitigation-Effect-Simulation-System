package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * AI 서버 POST /predict 응답. app/schemas.py 실제 필드명 그대로.
 * note는 태양광 응답에만 존재 - "expected_curtailment_mwh를 제공하지 않는다"는 안내 문구.
 * Frontend에서 태양광 화면에 이 필드를 다룰 때 note를 반드시 같이 노출해야 함(AI 서버 README 명시).
 */
@Getter
@Setter
@NoArgsConstructor
public class PredictionResponse {
    private String energy_type;
    private String region;
    private String target_date;
    private List<HourlyPrediction> hourly;
    private String model_used; // 실제 사용된 분류모델 (예: classifier_wind_demand) - AI 서버 2026-09-23 추가
    private String note; // solar 응답, 또는 wind에서 demand_forecast_mw 생략 시 존재
}
