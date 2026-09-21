package com.ami.curtailment.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDate;
import java.time.LocalDateTime;

/**
 * ESS 충방전 시뮬레이션 결과. 계획서 04장 02·03, 07장(흡수율) 대응.
 * AI 서버 POST /ess/simulate는 시간별이 아니라 "하루치 요청 -> 합계 응답" 구조라
 * 이 Entity도 1건 = 1회 시뮬레이션 실행(하루 단위 합계)으로 저장.
 * ratedPowerMw는 대시보드 슬라이더 조정값이라 매 시뮬레이션마다 함께 저장.
 */
@Entity
@Table(name = "ess_simulation_results")
@Getter
@Setter
@NoArgsConstructor
public class EssSimulationResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "region_id", nullable = false)
    private Region region;

    @Column(nullable = false)
    private LocalDate targetDate;

    @Column(nullable = false)
    private double ratedPowerMw; // ESS 정격출력 (슬라이더 조정값)

    @Column(length = 30, nullable = false)
    private String method; // "hourly_capped"(정식) | "naive_upper_bound"(단순 상한)

    @Column(nullable = false)
    private double totalCurtailmentMwh;

    @Column(nullable = false)
    private double totalAbsorbedMwh;

    @Column(nullable = false)
    private double absorptionRate;

    @Column(nullable = false)
    private LocalDateTime simulatedAt;
}
