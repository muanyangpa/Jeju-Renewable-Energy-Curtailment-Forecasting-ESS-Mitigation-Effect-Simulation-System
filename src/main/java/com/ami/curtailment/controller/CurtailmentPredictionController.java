package com.ami.curtailment.controller;

import com.ami.curtailment.domain.CurtailmentPrediction;
import com.ami.curtailment.domain.Region;
import com.ami.curtailment.dto.HourlyPrediction;
import com.ami.curtailment.dto.HourlyPredictionRequest;
import com.ami.curtailment.dto.HourlyPredictionResponse;
import com.ami.curtailment.dto.PredictionRequest;
import com.ami.curtailment.dto.PredictionResponse;
import com.ami.curtailment.repository.CurtailmentPredictionRepository;
import com.ami.curtailment.repository.RegionRepository;
import com.ami.curtailment.service.AiClientService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

/**
 * 외부 노출 API: 출력제어 예측 요청 및 결과 조회.
 * 통신규격 확정서 v1.0 04장 엔드포인트 표 반영.
 */
@RestController
@RequestMapping("/api/v1/predictions")
@RequiredArgsConstructor
public class CurtailmentPredictionController {

    private final AiClientService aiClientService;
    private final RegionRepository regionRepository;
    private final CurtailmentPredictionRepository curtailmentPredictionRepository;

    // 지역별 저장된 예측 결과 조회 (DB 조회, AI 서버 호출 아님)
    @GetMapping("/{regionId}")
    public List<CurtailmentPrediction> getPredictions(@PathVariable Long regionId) {
        return curtailmentPredictionRepository.findByRegionIdOrderByTargetHourDesc(regionId);
    }

    /**
     * 하루치(24시간) 배치 예측. 대시보드 기본 화면에서 사용하는 주 엔드포인트.
     * AI 서버 응답의 24개 시간대를 각각 CurtailmentPrediction으로 저장.
     */
    @PostMapping("/daily")
    public List<CurtailmentPrediction> predictDaily(@RequestBody PredictionRequest request) {
        Region region = regionRepository.findByName(request.getRegionName());
        LocalDate targetDate = LocalDate.parse(request.getTargetDate());

        PredictionResponse response = aiClientService.predictDaily(request);

        return response.getPredictions().stream()
                .map(hp -> saveOne(region, targetDate, response, hp))
                .toList();
    }

    /** 단건(특정 시각) 재조회·디버깅용 */
    @PostMapping("/hourly")
    public CurtailmentPrediction predictHourly(@RequestBody HourlyPredictionRequest request) {
        Region region = regionRepository.findByName(request.getRegionName());
        LocalDate targetDate = LocalDate.parse(request.getTargetDate());

        HourlyPredictionResponse response = aiClientService.predictHourly(request);

        return saveOne(region, targetDate,
                response.getModelVersion(), response.isCurtailmentMwhAvailable(),
                response.getPrediction());
    }

    private CurtailmentPrediction saveOne(Region region, LocalDate targetDate,
                                            PredictionResponse response, HourlyPrediction hp) {
        return saveOne(region, targetDate, response.getModelVersion(),
                response.isCurtailmentMwhAvailable(), hp);
    }

    private CurtailmentPrediction saveOne(Region region, LocalDate targetDate,
                                            String modelVersion, boolean curtailmentMwhAvailable,
                                            HourlyPrediction hp) {
        CurtailmentPrediction prediction = new CurtailmentPrediction();
        prediction.setRegion(region);
        prediction.setTargetHour(targetDate.atTime(hp.getHour(), 0));
        prediction.setCurtailmentProbability(hp.getCurtailmentProbability());
        prediction.setRiskLevel(hp.getRiskLevel());
        prediction.setForecastGenerationMwh(hp.getForecastGenerationMwh());
        prediction.setCurtailmentMwh(hp.getCurtailmentMwh());
        prediction.setCurtailmentMwhAvailable(curtailmentMwhAvailable);
        prediction.setModelVersion(modelVersion);
        prediction.setPredictedAt(LocalDateTime.now());

        return curtailmentPredictionRepository.save(prediction);
    }
}
