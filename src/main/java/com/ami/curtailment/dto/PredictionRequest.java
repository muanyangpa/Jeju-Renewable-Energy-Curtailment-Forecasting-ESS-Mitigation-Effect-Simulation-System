package com.ami.curtailment.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * Spring Boot -> Python AI 서버 요청 (통신규격 초안 — AI 담당자와 합의 후 확정 필요)
 *
 * 계획서 04장 01: 모델 입력은 "발전량(예보 기반) · 시간(sin/cos) · 월(sin/cos)"
 * 풍력은 여기에 전력수요(하루전 예보) 피처가 정식 채택됨 — demandMwh 필드로 반영.
 * 태양광은 아직 수요 피처 미채택이라 null 허용.
 */
@Getter
@NoArgsConstructor
@AllArgsConstructor
public class PredictionRequest {
    private String regionName;
    private String energySource;      // "SOLAR" / "WIND"
    private String targetHour;        // ISO 8601, 예측 대상 시각
    private double forecastGenerationMwh; // 기상청 예보 기반 발전량 예측치
    private Double demandMwh;         // 하루전 전력수요 예보 (풍력만 사용, 태양광은 null)
}
