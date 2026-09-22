package com.ami.curtailment.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * 시간 단위 발전량 데이터. 실측(ACTUAL)과 예보 기반 재구성(FORECAST) 둘 다 저장.
 * 계획서 05장: "실측 입력"과 "예보/재구성 입력"을 나란히 비교하는 방식과 대응.
 */
@Entity
@Table(name = "power_generations")
@Getter
@Setter
@NoArgsConstructor
public class PowerGeneration {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "region_id", nullable = false)
    private Region region;

    @Column(nullable = false)
    private LocalDateTime recordedAt; // 발전 시각(시간 단위)

    @Column(nullable = false)
    private double generationMwh; // 발전량(MWh)

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private DataSourceType sourceType; // ACTUAL(실측) / FORECAST(예보 기반 재구성)

    public enum DataSourceType {
        ACTUAL, FORECAST
    }
}
