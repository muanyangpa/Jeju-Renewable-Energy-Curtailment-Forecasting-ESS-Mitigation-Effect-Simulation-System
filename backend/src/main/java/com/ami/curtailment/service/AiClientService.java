package com.ami.curtailment.service;

import com.ami.curtailment.dto.EssSimulateRequest;
import com.ami.curtailment.dto.EssSimulateResponse;
import com.ami.curtailment.dto.PredictionRequest;
import com.ami.curtailment.dto.PredictionResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

/**
 * Python AI 서버(내부 전용)와 통신. 외부 요청은 Controller -> 이 Service -> Python 서버 순으로만 호출.
 * 엔드포인트 경로는 AI 서버 README(app/main.py) 실제 구현 기준 - 확정서 v1.0(/api/v1/...)이 아님.
 */
@Service
@RequiredArgsConstructor
public class AiClientService {

    private final WebClient aiServerWebClient;

    /** 하루치(24시간) 예측 */
    public PredictionResponse predict(PredictionRequest request) {
        return aiServerWebClient.post()
                .uri("/predict")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(PredictionResponse.class)
                .block(); // 프로토타입 단계라 블로킹 호출로 단순화
    }

    /** ESS 충방전 시뮬레이션 (04장 02·03 공유 엔드포인트) */
    public EssSimulateResponse simulateEss(EssSimulateRequest request) {
        return aiServerWebClient.post()
                .uri("/ess/simulate")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(EssSimulateResponse.class)
                .block();
    }
}
