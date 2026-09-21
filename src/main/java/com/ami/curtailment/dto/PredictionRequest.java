package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * Spring Boot -> AI 서버 POST /predict 요청. AI 서버 README(app/schemas.py 요약) 기준.
 *
 * 검증 책임 분담(AI 서버 README 그대로):
 * - Backend(여기): weather 필드 존재 여부·JSON 타입만 확인
 * - AI 서버: weather가 정확히 24개인지, 1~24시 중복·누락 없는지 최종 검증
 *   (위반 시 422 INVALID_HOUR_SET, Backend는 그대로 릴레이)
 */
@Getter
@Setter
@NoArgsConstructor
public class PredictionRequest {
    private String energy_type;              // "solar" | "wind", 필수
    private String region;                   // 예: "남원읍", 필수
    private String target_date;              // "YYYY-MM-DD", 필수
    private List<WeatherHour> weather;        // 24개(1~24시), 필수
    private List<Double> demand_forecast_mw;  // 24개, 풍력 전용(선택) - solar는 생략 가능
}
