package com.ami.anomaly.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Spring Boot -> Python AI 서버로 보낼 요청 (통신규격 초안)
 * 실제 필드는 AI 담당자와 합의 후 확정 필요
 */
@Getter
@NoArgsConstructor
@AllArgsConstructor
public class AiPredictRequest {
    private String consumerNo;       // 세대 식별자
    private List<Double> loadSeries; // 시계열 전력사용량 (예: 96포인트/24시간, 15분단위)
}
