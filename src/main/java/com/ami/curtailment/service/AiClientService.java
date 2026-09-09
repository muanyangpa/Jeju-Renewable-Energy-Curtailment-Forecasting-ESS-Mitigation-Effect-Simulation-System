package com.ami.curtailment.service;

import com.ami.curtailment.dto.HourlyPredictionRequest;
import com.ami.curtailment.dto.HourlyPredictionResponse;
import com.ami.curtailment.dto.PredictionRequest;
import com.ami.curtailment.dto.PredictionResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

/**
 * Python AI 서버(내부 전용)와 통신. 외부 요청은 Controller -> 이 Service -> Python 서버 순으로만 호출.
 * 통신규격 확정서 v1.0 04장 엔드포인트 표 반영: 배치(/predictions/daily)가 주 경로,
 * 대시보드 기본 화면(하루 24시간)이 이걸 쓰므로 HTTP 왕복을 24회에서 1회로 줄이는 목적.
 */
@Service
@RequiredArgsConstructor
public class AiClientService {

    private final WebClient aiServerWebClient;

    /** 하루치(24시간) 배치 예측 - 대시보드 기본 화면에서 사용 */
    public PredictionResponse predictDaily(PredictionRequest request) {
        return aiServerWebClient.post()
                .uri("/api/v1/predictions/daily")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(PredictionResponse.class)
                .block(); // 프로토타입 단계라 블로킹 호출로 단순화
    }

    /** 단건(특정 시각) 재조회·디버깅용 */
    public HourlyPredictionResponse predictHourly(HourlyPredictionRequest request) {
        return aiServerWebClient.post()
                .uri("/api/v1/predictions/hourly")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(HourlyPredictionResponse.class)
                .block();
    }

    /** AI 서버 기동 확인 (헬스체크) */
    public boolean checkHealth() {
        try {
            aiServerWebClient.get()
                    .uri("/api/v1/health")
                    .retrieve()
                    .toBodilessEntity()
                    .block();
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
