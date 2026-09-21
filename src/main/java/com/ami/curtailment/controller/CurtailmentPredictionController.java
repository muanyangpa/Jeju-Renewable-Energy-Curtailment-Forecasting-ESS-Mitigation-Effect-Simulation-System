package com.ami.curtailment.controller;

import com.ami.curtailment.domain.CurtailmentPrediction;
import com.ami.curtailment.domain.Region;
import com.ami.curtailment.dto.HourlyPrediction;
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
 * AI 서버 실제 엔드포인트(POST /predict) 기준 - 확정서 v1.0의 /api/v1/predictions/daily가 아님.
 */
@RestController
@RequestMapping("/api/predictions")
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
     * 하루치(1~24시) 예측. AI 서버 POST /predict 호출 후 24개 결과를 각각 저장.
     */
    @PostMapping
    public List<CurtailmentPrediction> predict(@RequestBody PredictionRequest request) {
        Region region = regionRepository.findByName(request.getRegion());
        LocalDate targetDate = LocalDate.parse(request.getTarget_date());

        PredictionResponse response = aiClientService.predict(request);

        return response.getHourly().stream()
                .map(hp -> saveOne(region, targetDate, response.getNote(), hp))
                .toList();
    }

    private CurtailmentPrediction saveOne(Region region, LocalDate targetDate, String note, HourlyPrediction hp) {
        CurtailmentPrediction prediction = new CurtailmentPrediction();
        prediction.setRegion(region);
        // AI 서버는 hour를 1~24로 사용 - LocalDateTime 시(hour)는 0~23이라 -1 보정
        prediction.setTargetHour(targetDate.atStartOfDay().plusHours(hp.getHour() - 1));
        prediction.setCurtailmentProbability(hp.getCurtailment_probability());
        prediction.setGenerationForecastMwh(hp.getGeneration_forecast_mwh());
        prediction.setCurtailmentMwh(hp.getExpected_curtailment_mwh());
        prediction.setNote(note);
        prediction.setPredictedAt(LocalDateTime.now());

        return curtailmentPredictionRepository.save(prediction);
    }
}
