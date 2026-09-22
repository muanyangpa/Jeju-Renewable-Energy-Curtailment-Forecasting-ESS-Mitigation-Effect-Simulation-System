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
     * 하루치(1~24시, 원본 데이터셋 관행 그대로) 예측. AI 서버 POST /predict 호출 후 24개 결과를 각각 저장.
     * weather 요청 필드의 hour 값도 호출하는 쪽(Frontend 등)이 1~24 그대로 채워서 보내면 되고,
     * Backend는 이 값을 가공하지 않고 그대로 AI 서버에 전달한다 (지호님 확인 사항).
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
        // 원본 데이터셋 관행: 1~23시는 그날 해당 시각, 24시는 "자정"=다음날 00:00.
        // (지호님 확인: "현재 데이터셋 파싱시에 자정 24시를 0시로 바꾸고 있는데, AI에서 이를
        //  처리하므로 Backend는 원본 1~24 그대로 보내면 됨" — 응답 저장 시에도 같은 관행 적용)
        LocalDateTime hourStart = (hp.getHour() == 24)
                ? targetDate.plusDays(1).atStartOfDay()
                : targetDate.atTime(hp.getHour(), 0);
        prediction.setTargetHour(hourStart);
        prediction.setCurtailmentProbability(hp.getCurtailment_probability());
        prediction.setGenerationForecastMwh(hp.getGeneration_forecast_mwh());
        prediction.setCurtailmentMwh(hp.getExpected_curtailment_mwh());
        prediction.setNote(note);
        prediction.setPredictedAt(LocalDateTime.now());

        return curtailmentPredictionRepository.save(prediction);
    }
}
