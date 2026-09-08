package com.ami.curtailment.controller;

import com.ami.curtailment.domain.CurtailmentPrediction;
import com.ami.curtailment.domain.Region;
import com.ami.curtailment.dto.PredictionRequest;
import com.ami.curtailment.dto.PredictionResponse;
import com.ami.curtailment.repository.CurtailmentPredictionRepository;
import com.ami.curtailment.repository.RegionRepository;
import com.ami.curtailment.service.AiClientService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 외부 노출 API: 출력제어 예측 요청 및 결과 조회.
 */
@RestController
@RequestMapping("/api/curtailment-predictions")
@RequiredArgsConstructor
public class CurtailmentPredictionController {

    private final AiClientService aiClientService;
    private final RegionRepository regionRepository;
    private final CurtailmentPredictionRepository curtailmentPredictionRepository;

    // 지역별 예측 결과 조회
    @GetMapping("/{regionId}")
    public List<CurtailmentPrediction> getPredictions(@PathVariable Long regionId) {
        return curtailmentPredictionRepository.findByRegionIdOrderByTargetHourDesc(regionId);
    }

    // AI 서버에 예측 요청 후 결과 저장
    @PostMapping("/predict")
    public CurtailmentPrediction predictAndSave(@RequestBody PredictionRequest request) {
        Region region = regionRepository.findByName(request.getRegionName());

        PredictionResponse response = aiClientService.predict(request);

        CurtailmentPrediction prediction = new CurtailmentPrediction();
        prediction.setRegion(region);
        prediction.setTargetHour(LocalDateTime.parse(request.getTargetHour()));
        prediction.setCurtailmentProbability(response.getCurtailmentProbability());
        prediction.setExcessGenerationMwh(response.getExcessGenerationMwh());
        prediction.setPredictedAt(LocalDateTime.now());

        return curtailmentPredictionRepository.save(prediction);
    }
}
