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
 * ratedPowerMw/energyCapacityMwh는 대시보드 슬라이더 2축 조정값이라 매 시뮬레이션마다 함께 저장 -
 * 같은 정격출력이라도 저장용량에 따라 흡수율이 달라지므로 둘을 함께 남기지 않으면 결과를 재현할 수 없다.
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
    private double ratedPowerMw; // ESS 정격출력 (슬라이더 1축)

    @Column(length = 30, nullable = false)
    private String method; // "storage_constrained"(정식) | "hourly_capped"(이론 상한) | "naive_upper_bound"(단순 상한)

    // 아래 4개는 storage_constrained 실행에서만 채워진다 (다른 방식은 저장용량을 쓰지 않으므로 null)
    @Column
    private Double energyCapacityMwh; // ESS 저장용량 (슬라이더 2축)

    @Column
    private Double usableCapacityMwh; // SoC 범위를 적용한 가용용량

    @Column
    private Integer hoursFull; // ESS가 가득 차서 더 흡수하지 못한 시간 수

    @Column
    private Double annualCycles;

    @Column(nullable = false)
    private double totalCurtailmentMwh;

    @Column(nullable = false)
    private double totalAbsorbedMwh;

    @Column(nullable = false)
    private double absorptionRate;

    @Column(nullable = false)
    private LocalDateTime simulatedAt;
}
