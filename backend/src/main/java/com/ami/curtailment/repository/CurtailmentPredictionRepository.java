package com.ami.curtailment.repository;

import com.ami.curtailment.domain.CurtailmentPrediction;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface CurtailmentPredictionRepository extends JpaRepository<CurtailmentPrediction, Long> {
    List<CurtailmentPrediction> findByRegionIdOrderByTargetHourDesc(Long regionId);
}
