package com.ami.anomaly.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.math.BigDecimal;

/**
 * 세대(계량기) 정보
 */
@Entity
@Table(name = "meters")
@Getter
@Setter
@NoArgsConstructor
public class Meter {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String consumerNo; // 계량기/세대 고유번호 (SGCC의 CONS_NO 대응)

    @Column(nullable = false)
    private BigDecimal contractPowerKw; // 계약전력(kW)

    @Column(nullable = false)
    private boolean idleRequested; // 휴지신청 여부

    @Column(nullable = false)
    private boolean reservePowerContract; // 예비전력 계약 여부

    private String address;
}
