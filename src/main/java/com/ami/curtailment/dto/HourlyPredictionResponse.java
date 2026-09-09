package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * 단건(특정 시각) 응답 (통신규격 확정서 v1.0, 04장)
 */
@Getter
@Setter
@NoArgsConstructor
public class HourlyPredictionResponse {
    private String regionName;
    private String energySource;
    private String targetDate;
    private String modelVersion;
    private boolean curtailmentMwhAvailable;
    private HourlyPrediction prediction; // 단건이라 배열이 아닌 단일 객체
}
