package com.ami.curtailment.controller;

import com.ami.curtailment.domain.EssSimulationResult;
import com.ami.curtailment.domain.Region;
import com.ami.curtailment.dto.EssSimulateRequest;
import com.ami.curtailment.repository.EssSimulationResultRepository;
import com.ami.curtailment.repository.RegionRepository;
import com.ami.curtailment.service.EssSimulationService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;

/**
 * ESS 완화효과 시뮬레이션 API. 04장 02(기본)·03(용량 조정 슬라이더)이 공유.
 * 슬라이더는 ratedPowerMw만 바꿔가며 이 엔드포인트를 반복 호출.
 * 실제 계산은 AI 서버 POST /ess/simulate가 수행 (Backend는 호출+저장만).
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
        return essSimulationResultRepository.findByRegionIdOrderByTargetDateDesc(regionId);
    }

    @PostMapping
    public EssSimulationResult simulate(@RequestParam String regionName,
                                          @RequestParam String targetDate,
                                          @RequestBody EssSimulateRequest request) {
        Region region = regionRepository.findByName(regionName);
        return essSimulationService.simulate(region, LocalDate.parse(targetDate), request);
    }
}
