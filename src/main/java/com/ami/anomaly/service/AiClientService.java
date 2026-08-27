package com.ami.anomaly.service;

import com.ami.anomaly.dto.AiPredictRequest;
import com.ami.anomaly.dto.AiPredictResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

/**
 * Python AI 서버(내부 전용)와 통신하는 서비스.
 * 외부 요청은 이 서비스를 거치지 않고, Controller -> 이 Service -> Python 서버 순으로만 호출됨.
 */
@Service
@RequiredArgsConstructor
public class AiClientService {

    private final WebClient aiServerWebClient;

    public AiPredictResponse predict(AiPredictRequest request) {
        return aiServerWebClient.post()
                .uri("/predict") // Python 측 엔드포인트 - AI 담당자와 합의 후 확정
                .bodyValue(request)
                .retrieve()
                .bodyToMono(AiPredictResponse.class)
                .block(); // 데모/프로토타입 단계라 블로킹 호출로 단순화
    }
}
