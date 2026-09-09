package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * 시간별 기상 원본 데이터 (통신규격 확정서 v1.0, 03장)
 * Backend는 이 값을 가공하지 않고 AI 서버에 그대로 전달한다 (변환은 AI 서버 담당).
 */
@Getter
@Setter
@NoArgsConstructor
public class WeatherHour {
    private int hour;              // 0~23, 필수
    private Double windSpeedMs;    // 풍속(m/s) - WIND 필수, SOLAR는 null
    private Double irradianceWm2;  // 일사량(W/m^2) - SOLAR 필수, WIND는 null
    private double temperatureC;   // 기온(℃) - 공통 필수
    private Double demandMwh;      // 전력수요 - WIND 필수, SOLAR는 null
}
