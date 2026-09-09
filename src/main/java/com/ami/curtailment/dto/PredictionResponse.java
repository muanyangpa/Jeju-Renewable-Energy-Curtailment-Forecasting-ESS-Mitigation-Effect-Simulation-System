package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * Python AI 서버 -> Spring Boot 배치 응답 (통신규격 확정서 v1.0, 03장)
 */
@Getter
@Setter
@NoArgsConstructor
public class PredictionResponse {
    private String regionName;
    private String energySource;
    private String targetDate;
    private String modelVersion;              // 재현성 추적용 - 발표 시 근거 자료
    private boolean curtailmentMwhAvailable;   // SOLAR면 false. false면 predictions[].curtailmentMwh 전부 null
    private List<HourlyPrediction> predictions; // 24개(0~23시)
}
