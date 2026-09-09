package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * 단건(특정 시각) 재조회·디버깅용 요청 (통신규격 확정서 v1.0, 04장)
 * PredictionRequest와 달리 weather가 배열이 아니라 객체 하나.
 */
@Getter
@Setter
@NoArgsConstructor
public class HourlyPredictionRequest {
    private String regionName;
    private String energySource;
    private String targetDate;
    private WeatherHour weather; // 단건이라 배열이 아닌 단일 객체
}
