package com.ami.curtailment.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * Python AI 서버(RandomForestClassifier)로부터 받은 출력제어 확률 예측 결과 저장.
 * 계획서 04장 01 "출력제어 확률 예측 엔진" 대응.
 * 통신규격 확정서 v1.0 반영: excessGenerationMwh -> curtailmentMwh로 필드명 정정
 * (07장 ESS 계산에 쓰이는 값은 "초과발전량"이 아니라 "출력제어량"이라는 지적 반영).
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

    @Column(length = 20)
    private String riskLevel; // "HIGH" | "MEDIUM" | "LOW"

    private Double forecastGenerationMwh; // AI 서버 변환모델 산출 발전량 (대시보드 참고 표시용)

    private Double curtailmentMwh; // 예상 출력제어량 (nullable) - SOLAR는 항상 null

    @Column(nullable = false)
    private boolean curtailmentMwhAvailable; // false면 curtailmentMwh가 null. SOLAR는 항상 false

    @Column(length = 50)
    private String modelVersion; // 재현성 추적용 - 발표 시 근거 자료로 사용

    @Column(nullable = false)
    private LocalDateTime predictedAt; // 예측이 생성된 시각
}
