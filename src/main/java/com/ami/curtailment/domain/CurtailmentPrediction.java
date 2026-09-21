package com.ami.curtailment.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * AI 서버(POST /predict)로부터 받은 출력제어 확률 예측 결과 저장.
 * 계획서 04장 01 "출력제어 확률 예측 엔진" 대응.
 *
 * 필드는 AI 서버 실제 응답(app/schemas.py) 기준 - riskLevel·modelVersion은
 * 실제 응답에 없어 제거함 (이전 확정서 v1.0 문서 기준이었으나 구현체와 달라 정정).
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
    private LocalDateTime targetHour; // 예측 대상 시각

    @Column(nullable = false)
    private double curtailmentProbability; // 출력제어 확률 (0~1)

    private double generationForecastMwh; // 컨버터 산출 발전량 예측치

    private Double curtailmentMwh; // 예상 출력제어량 (nullable) - SOLAR는 항상 null, WIND만 값 존재

    @Column(length = 255)
    private String note; // SOLAR 응답에만 존재하는 안내 문구 (curtailment 미제공 사유)

    @Column(nullable = false)
    private LocalDateTime predictedAt; // 예측이 생성된 시각
}
