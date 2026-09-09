package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * Spring Boot -> Python AI 서버 배치 요청 (통신규격 확정서 v1.0, 03장)
 * POST /api/v1/predictions/daily 에서 사용.
 *
 * 원칙: 모델 입력으로 들어가는 가공(발전량 변환, sin/cos 등)은 전부 AI 서버가 한다.
 * Backend는 기상 원본과 조회 조건만 전달한다.
 */
@Getter
@Setter
@NoArgsConstructor
public class PredictionRequest {
    private String regionName;         // 필수, 예: "제주"
    private String energySource;       // 필수, "WIND" | "SOLAR"
    private String targetDate;         // 필수, ISO-8601 (YYYY-MM-DD, KST) - 월 sin/cos 계산에 AI 서버가 사용
    private List<WeatherHour> weather; // 필수, 24개(0~23시)
}
