package com.ami.anomaly.repository;

import com.ami.anomaly.domain.MeterReading;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface MeterReadingRepository extends JpaRepository<MeterReading, Long> {
    List<MeterReading> findTop96ByMeterIdOrderByRecordedAtDesc(Long meterId); // 최근 96개(24시간, 15분단위) 조회용
}
