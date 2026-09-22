package com.ami.curtailment.repository;

import com.ami.curtailment.domain.EssSimulationResult;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface EssSimulationResultRepository extends JpaRepository<EssSimulationResult, Long> {
    List<EssSimulationResult> findByRegionIdOrderByTargetDateDesc(Long regionId);
}
