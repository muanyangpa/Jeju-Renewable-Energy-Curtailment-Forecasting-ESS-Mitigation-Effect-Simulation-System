package com.ami.curtailment.dto;

import com.ami.curtailment.domain.EssSimulationResult;
import lombok.Getter;

import java.time.LocalDate;
import java.time.LocalDateTime;

/**
 * GET /api/ess-simulations/{regionId} 응답 전용 DTO.
 * CurtailmentPredictionView와 같은 이유(Hibernate 지연 로딩 프록시 직렬화 실패, 통신규격
 * v1.1 07장)로 EssSimulationResult 엔티티를 그대로 반환하지 않고 이 DTO로 변환해서 내려준다.
 */
@Getter
public class EssSimulationResultView {
    private final Long id;
    private final String regionName;
    private final LocalDate targetDate;
    private final double ratedPowerMw;
    private final String method;
    private final Double energyCapacityMwh;   // storage_constrained에서만 값 존재
    private final Double usableCapacityMwh;
    private final Integer hoursFull;
    private final Double annualCycles;
    private final double totalCurtailmentMwh;
    private final double totalAbsorbedMwh;
    private final double absorptionRate;
    private final LocalDateTime simulatedAt;

    public EssSimulationResultView(EssSimulationResult entity) {
        this.id = entity.getId();
        this.regionName = entity.getRegion().getName(); // 트랜잭션 안에서 프록시를 초기화해 값만 꺼냄
        this.targetDate = entity.getTargetDate();
        this.ratedPowerMw = entity.getRatedPowerMw();
        this.method = entity.getMethod();
        this.energyCapacityMwh = entity.getEnergyCapacityMwh();
        this.usableCapacityMwh = entity.getUsableCapacityMwh();
        this.hoursFull = entity.getHoursFull();
        this.annualCycles = entity.getAnnualCycles();
        this.totalCurtailmentMwh = entity.getTotalCurtailmentMwh();
        this.totalAbsorbedMwh = entity.getTotalAbsorbedMwh();
        this.absorptionRate = entity.getAbsorptionRate();
        this.simulatedAt = entity.getSimulatedAt();
    }
}
