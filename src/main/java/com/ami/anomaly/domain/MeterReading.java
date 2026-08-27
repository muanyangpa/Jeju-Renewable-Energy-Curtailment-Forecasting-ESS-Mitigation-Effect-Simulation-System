package com.ami.anomaly.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * AMI 계량기로부터 수신한 원시 전력사용량 데이터 (1건 = 특정 시각의 사용량 1포인트)
 */
@Entity
@Table(name = "meter_readings")
@Getter
@Setter
@NoArgsConstructor
public class MeterReading {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "meter_id", nullable = false)
    private Meter meter;

    @Column(nullable = false)
    private double usageKwh; // 해당 시점 전력사용량(kWh)

    @Column(nullable = false)
    private LocalDateTime recordedAt; // 계량기가 측정한 시각 (15분 단위 등)
}
