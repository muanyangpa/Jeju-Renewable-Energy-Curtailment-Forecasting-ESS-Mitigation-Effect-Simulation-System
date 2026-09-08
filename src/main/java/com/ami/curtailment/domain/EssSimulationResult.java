package com.ami.curtailment.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * ESS 충방전 시뮬레이션 결과. 계획서 04장 02·03, 07장(흡수율 36.8~48.4%) 대응.
 * essCapacityMw는 대시보드 슬라이더로 조정 가능한 값이라 매 시뮬레이션마다 함께 저장.
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
    private LocalDateTime targetHour;

    @Column(nullable = false)
    private double essCapacityMw; // 시뮬레이션에 사용한 ESS 정격출력 (슬라이더 조정값)

    @Column(nullable = false)
    private double curtailmentMwh; // 해당 시간 출력제어량

    @Column(nullable = false)
    private double absorbedMwh; // ESS가 흡수 가능한 양 = min(curtailmentMwh, essCapacityMw)

    @Column(nullable = false)
    private LocalDateTime simulatedAt;
}
