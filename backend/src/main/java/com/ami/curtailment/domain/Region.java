package com.ami.curtailment.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * 지역(제주 세부 권역 또는 전남 등). 계획서 05장 기준 한림·남원·표선·구좌 등.
 */
@Entity
@Table(name = "regions")
@Getter
@Setter
@NoArgsConstructor
public class Region {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String name; // 예: "한림", "남원", "표선", "구좌", "제주전체"

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private EnergySource energySource; // SOLAR / WIND
}
