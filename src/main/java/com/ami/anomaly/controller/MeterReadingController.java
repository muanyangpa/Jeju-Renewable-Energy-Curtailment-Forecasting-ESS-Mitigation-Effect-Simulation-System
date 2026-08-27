package com.ami.anomaly.controller;

import com.ami.anomaly.domain.Meter;
import com.ami.anomaly.domain.MeterReading;
import com.ami.anomaly.dto.MeterReadingRequest;
import com.ami.anomaly.repository.MeterReadingRepository;
import com.ami.anomaly.repository.MeterRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * AMI 계량기(또는 시뮬레이터)가 실시간으로 전력사용량을 보내는 수신 엔드포인트.
 * "IoT 파트"에 해당 — 계량기 -> 이 API -> DB 저장 -> (추후) AI 분석 요청 흐름의 입구.
 */
@RestController
@RequestMapping("/api/meters/{consumerNo}/readings")
@RequiredArgsConstructor
public class MeterReadingController {

    private final MeterRepository meterRepository;
    private final MeterReadingRepository meterReadingRepository;

    // 계량기 1건 측정값 수신
    @PostMapping
    public ResponseEntity<MeterReading> receive(@PathVariable String consumerNo,
                                                 @RequestBody MeterReadingRequest request) {
        Meter meter = meterRepository.findByConsumerNo(consumerNo);
        if (meter == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
        }

        MeterReading reading = new MeterReading();
        reading.setMeter(meter);
        reading.setUsageKwh(request.getUsageKwh());
        reading.setRecordedAt(request.getRecordedAt());

        return ResponseEntity.ok(meterReadingRepository.save(reading));
    }

    // 최근 96개(24시간, 15분단위) 조회 - AI 판정 요청 시 이 데이터를 loadSeries로 사용
    @GetMapping("/recent")
    public List<MeterReading> getRecent(@PathVariable String consumerNo) {
        Meter meter = meterRepository.findByConsumerNo(consumerNo);
        if (meter == null) {
            return List.of();
        }
        return meterReadingRepository.findTop96ByMeterIdOrderByRecordedAtDesc(meter.getId());
    }
}
