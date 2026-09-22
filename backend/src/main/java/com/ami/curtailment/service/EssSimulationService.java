package com.ami.curtailment.service;

import com.ami.curtailment.domain.EssSimulationResult;
import com.ami.curtailment.domain.Region;
import com.ami.curtailment.dto.EssSimulateRequest;
import com.ami.curtailment.dto.EssSimulateResponse;
import com.ami.curtailment.repository.EssSimulationResultRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;

/**
 * ESS 충방전 시뮬레이션. 계산(min(그 시각 제어량, 정격출력) 합산, 07장 두 방식)은
 * AI 서버 POST /ess/simulate가 수행 - 이 서비스는 호출과 결과 저장만 담당.
 * (이전 버전은 Backend에서 직접 min() 계산했으나, 실제 AI 서버 구현에서 그 역할을
 * AI 서버가 갖게 되어 정정함.)
 */
@Service
@RequiredArgsConstructor
public class EssSimulationService {

    private final AiClientService aiClientService;
    private final EssSimulationResultRepository essSimulationResultRepository;

    public EssSimulationResult simulate(Region region, LocalDate targetDate, EssSimulateRequest request) {
        EssSimulateResponse response = aiClientService.simulateEss(request);

        EssSimulationResult result = new EssSimulationResult();
        result.setRegion(region);
        result.setTargetDate(targetDate);
        result.setRatedPowerMw(response.getRated_power_mw());
        result.setMethod(response.getMethod());
        result.setTotalCurtailmentMwh(response.getTotal_curtailment_mwh());
        result.setTotalAbsorbedMwh(response.getTotal_absorbed_mwh());
        result.setAbsorptionRate(response.getAbsorption_rate());
        result.setSimulatedAt(LocalDateTime.now());

        return essSimulationResultRepository.save(result);
    }
}
