package com.ami.anomaly.repository;

import com.ami.anomaly.domain.Meter;
import org.springframework.data.jpa.repository.JpaRepository;

public interface MeterRepository extends JpaRepository<Meter, Long> {
    Meter findByConsumerNo(String consumerNo);
}
