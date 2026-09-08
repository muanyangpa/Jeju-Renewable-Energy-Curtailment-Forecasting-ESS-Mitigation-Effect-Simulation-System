package com.ami.curtailment.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * Python AI 서버(RandomForestClassifier)로부터 받은 출력제어 확률 예측 결과 저장.
 * 계획서 04장 01 "출력제어 확률 예측 엔진" 대응.
 */
@Entity
@Table(name = "curtailment_predictions")
@Getter
@Setter
@NoArgsConstructor
public class CurtailmentPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "region_id", nullable = false)
    private Region region;

    @Column(nullable = false)
    private LocalDateTime targetHour; // 예측 대상 시각 (하루 전 예측이므로 미래 시점)

    @Column(nullable = false)
    private double curtailmentProbability; // 출력제어 확률 (0~1)

    private Double excessGenerationMwh; // 초과발전량 크기 (04장: ESS 충전량 계산에 필요, nullable)

    @Column(nullable = false)
    private LocalDateTime predictedAt; // 예측이 생성된 시각
}
