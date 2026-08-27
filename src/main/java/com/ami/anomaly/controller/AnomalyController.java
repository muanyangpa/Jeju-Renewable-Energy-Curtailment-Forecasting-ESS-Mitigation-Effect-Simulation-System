package com.ami.anomaly.controller;

import com.ami.anomaly.domain.AnomalyResult;
import com.ami.anomaly.domain.Meter;
import com.ami.anomaly.domain.ViolationType;
import com.ami.anomaly.dto.AiPredictRequest;
import com.ami.anomaly.dto.AiPredictResponse;
import com.ami.anomaly.repository.AnomalyResultRepository;
import com.ami.anomaly.repository.MeterRepository;
import com.ami.anomaly.service.AiClientService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 외부에 노출되는 이상탐지 API (인증/검증은 SecurityConfig에서 처리)
 */
@RestController
@RequestMapping("/api/anomalies")
@RequiredArgsConstructor
public class AnomalyController {

    private final AiClientService aiClientService;
    private final MeterRepository meterRepository;
    private final AnomalyResultRepository anomalyResultRepository;

    // 세대별 이상탐지 결과 조회
    @GetMapping("/{meterId}")
    public List<AnomalyResult> getResults(@PathVariable Long meterId) {
        return anomalyResultRepository.findByMeterId(meterId);
    }

    // 특정 세대의 최신 시계열 데이터를 AI 서버에 보내 판정받고 결과 저장
    @PostMapping("/predict/{consumerNo}")
    public AnomalyResult predictAndSave(@PathVariable String consumerNo,
                                         @RequestBody List<Double> loadSeries) {
        Meter meter = meterRepository.findByConsumerNo(consumerNo);

        AiPredictRequest request = new AiPredictRequest(consumerNo, loadSeries);
        AiPredictResponse response = aiClientService.predict(request);

        AnomalyResult result = new AnomalyResult();
        result.setMeter(meter);
        result.setAnomalyScore(response.getAnomalyScore());
        result.setViolationType(ViolationType.valueOf(response.getViolationType()));
        result.setExplanation(response.getExplanation());
        result.setDetectedAt(LocalDateTime.now());

        return anomalyResultRepository.save(result);
    }
}
