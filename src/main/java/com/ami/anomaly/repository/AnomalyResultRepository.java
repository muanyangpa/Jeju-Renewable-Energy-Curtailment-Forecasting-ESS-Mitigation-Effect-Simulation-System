package com.ami.anomaly.repository;

import com.ami.anomaly.domain.AnomalyResult;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface AnomalyResultRepository extends JpaRepository<AnomalyResult, Long> {
    List<AnomalyResult> findByMeterId(Long meterId);
}
