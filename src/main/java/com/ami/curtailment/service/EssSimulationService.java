package com.ami.curtailment.service;

import com.ami.curtailment.domain.EssSimulationResult;
import com.ami.curtailment.domain.Region;
import com.ami.curtailment.repository.EssSimulationResultRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

/**
 * ESS 충방전 시뮬레이션. 계획서 07장 "물리적으로 더 정확한 방식" 그대로 구현:
 * 시간별 min(그 시각의 실제 출력제어량, ESS 정격출력)을 합산.
 *
 * 04장 03 "ESS 용량 조정 시뮬레이터": essCapacityMw를 대시보드 슬라이더 값으로 받아
 * 별도 모델 없이 이 계산만 재사용 — 정책적 질문("증설하면 흡수율이 얼마나 오르나")에 즉시 응답.
 */
@Service
@RequiredArgsConstructor
public class EssSimulationService {

    private final EssSimulationResultRepository essSimulationResultRepository;

    public EssSimulationResult simulate(Region region, LocalDateTime targetHour,
                                          double curtailmentMwh, double essCapacityMw) {
        // 07장 방식: 그 시각 제어량이 ESS 정격출력보다 작으면 그만큼만, 크면 정격출력만큼만 흡수
        double absorbedMwh = Math.min(curtailmentMwh, essCapacityMw);

        EssSimulationResult result = new EssSimulationResult();
        result.setRegion(region);
        result.setTargetHour(targetHour);
        result.setEssCapacityMw(essCapacityMw);
        result.setCurtailmentMwh(curtailmentMwh);
        result.setAbsorbedMwh(absorbedMwh);
        result.setSimulatedAt(LocalDateTime.now());

        return essSimulationResultRepository.save(result);
    }
}
