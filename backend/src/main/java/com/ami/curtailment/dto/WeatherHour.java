package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * 시간별 기상 원본 데이터. AI 서버 app/schemas.py 실제 필드명 그대로 사용 (snake_case 아님 -
 * Jackson이 자동 매핑하도록 JSON 프로퍼티명을 그대로 필드명으로 사용, 별도 @JsonProperty 불필요
 * 하게 필드명 자체를 스펙과 동일하게 맞춤).
 */
@Getter
@Setter
@NoArgsConstructor
public class WeatherHour {
    private int hour;          // 1~24
    private Double solar_rad;  // 일사량 - SOLAR 필수, WIND는 0.0 또는 null 가능(AI 서버 스펙 참고)
    private double temp;       // 기온(℃) - 공통 필수
    private Double cloud;      // 구름량 - SOLAR 컨버터 입력
    private Double wind_speed; // 풍속(m/s) - WIND 필수
}
