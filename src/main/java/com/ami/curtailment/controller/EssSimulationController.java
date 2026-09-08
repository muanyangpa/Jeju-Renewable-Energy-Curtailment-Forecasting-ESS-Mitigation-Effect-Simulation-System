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
 */
@RestController
@RequestMapping("/api/ess-simulations")
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
     * curtailmentMwh는 CurtailmentPrediction의 excessGenerationMwh를 그대로 넘겨받는 것을 전제로 함
     * — 실제 값 전달 방식은 Frontend 구현 시 확정 필요.
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
