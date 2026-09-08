package com.ami.curtailment.service;

import com.ami.curtailment.dto.PredictionRequest;
import com.ami.curtailment.dto.PredictionResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

/**
 * Python AI 서버(내부 전용)와 통신. 외부 요청은 Controller -> 이 Service -> Python 서버 순으로만 호출.
 */
@Service
@RequiredArgsConstructor
public class AiClientService {

    private final WebClient aiServerWebClient;

    public PredictionResponse predict(PredictionRequest request) {
        return aiServerWebClient.post()
                .uri("/predict") // Python 측 엔드포인트 경로 - AI 담당자와 합의 후 확정
                .bodyValue(request)
                .retrieve()
                .bodyToMono(PredictionResponse.class)
                .block(); // 프로토타입 단계라 블로킹 호출로 단순화
    }
}
