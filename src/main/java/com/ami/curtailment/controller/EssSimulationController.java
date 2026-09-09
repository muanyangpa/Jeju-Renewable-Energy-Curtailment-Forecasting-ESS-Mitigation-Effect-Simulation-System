package com.ami.curtailment.controller;

import com.ami.curtailment.domain.EssSimulationResult;
import com.ami.curtailment.domain.Region;
import com.ami.curtailment.repository.EssSimulationResultRepository;
import com.ami.curtailment.repository.RegionRepository;
import com.ami.curtailment.service.EssSimulationService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

/**
 * ESS 완화효과 시뮬레이션 API. 04장 03 "ESS 용량 조정 시뮬레이터" —
 * 대시보드 슬라이더가 essCapacityMw를 바꿔가며 이 엔드포인트를 반복 호출하는 구조.
 * 통신규격 확정서 v1.0 04장: 경로는 /api/v1/ess/simulate.
 *
 * 여기 쓰이는 curtailmentMwh는 CurtailmentPrediction.curtailmentMwh(AI 서버가 내려준 값)와
 * 동일한 정의(출력제어량, 초과발전량 아님)를 그대로 사용한다 — 지호님 문서 04장 표 하단 주석 대응.
 * SOLAR 지역은 curtailmentMwh가 항상 null이므로, Frontend에서 이 화면 자체를 비활성 처리해야 한다.
 */
@RestController
@RequestMapping("/api/v1/ess")
@RequiredArgsConstructor
public class EssSimulationController {

    private final EssSimulationService essSimulationService;
    private final EssSimulationResultRepository essSimulationResultRepository;
    private final RegionRepository regionRepository;

    @GetMapping("/{regionId}")
    public List<EssSimulationResult> getResults(@PathVariable Long regionId) {
        return essSimulationResultRepository.findByRegionIdOrderByTargetHourDesc(regionId);
    }

    /**
     * ESS 용량을 바꿔가며 흡수율을 즉시 재계산 (슬라이더 what-if 기능).
     */
    @PostMapping("/simulate")
    public EssSimulationResult simulate(@RequestParam String regionName,
                                          @RequestParam String targetHour,
                                          @RequestParam double curtailmentMwh,
                                          @RequestParam double essCapacityMw) {
        Region region = regionRepository.findByName(regionName);
        return essSimulationService.simulate(region, LocalDateTime.parse(targetHour),
                curtailmentMwh, essCapacityMw);
    }
}
