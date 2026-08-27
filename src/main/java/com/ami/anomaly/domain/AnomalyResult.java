package com.ami.anomaly.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * AI 이상탐지 결과 저장 (Python AI 서버로부터 받은 응답을 저장)
 */
@Entity
@Table(name = "anomaly_results")
@Getter
@Setter
@NoArgsConstructor
public class AnomalyResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "meter_id", nullable = false)
    private Meter meter;

    @Column(nullable = false)
    private double anomalyScore; // 이상점수 (0~1)

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private ViolationType violationType;

    @Column(columnDefinition = "TEXT")
    private String explanation; // SHAP 등 XAI 설명 (JSON 또는 텍스트)

    @Column(nullable = false)
    private LocalDateTime detectedAt;
}
