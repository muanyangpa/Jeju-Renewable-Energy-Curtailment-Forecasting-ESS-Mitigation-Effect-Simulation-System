package com.ami.curtailment.repository;

import com.ami.curtailment.domain.PowerGeneration;
import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDateTime;
import java.util.List;

public interface PowerGenerationRepository extends JpaRepository<PowerGeneration, Long> {
    List<PowerGeneration> findByRegionIdAndRecordedAtBetween(
            Long regionId, LocalDateTime start, LocalDateTime end);
}
