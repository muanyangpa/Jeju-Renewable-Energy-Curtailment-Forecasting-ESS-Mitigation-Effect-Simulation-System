package com.ami.anomaly.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * AMI(계량기 시뮬레이터 포함)로부터 들어오는 단일 측정값 요청 바디
 */
@Getter
@Setter
@NoArgsConstructor
public class MeterReadingRequest {
    private double usageKwh;
    private LocalDateTime recordedAt;
}
