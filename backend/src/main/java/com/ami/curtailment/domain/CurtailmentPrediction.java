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
 *
 * 2026-09-25 AI 서버 변경(커밋 f7f0525→0787d28) 반영:
 * - curtailmentProbability는 sigmoid 확률 보정이 적용된 값.
 * - note는 SOLAR 전용이 아니라 세 경로 모두 올 수 있음(note 컬럼 주석 참고).
 * - ESS 흡수율 계산에는 curtailmentMwh(예측값)를 쓰지 말 것 - 실측 제어량 기반만 사용
 *   (예측값 사용 시 실측 36.8%가 78%로 부풀려짐, AI 서버팀 확인).
 *
 * 2026-09-26 AI 서버 추가 변경 반영:
 * - 0.03 같은 고정 임계값 사용 금지로 방침 변경. modelUsed별로 operationalThreshold가
 *   다르게 내려오므로(0.02~0.43, 최대 20배 차이) 응답값을 그대로 저장해서 써야 함.
 * - operationalThresholdReliable=false인 경우 등급(낮음/보통/높음) 확정 표시 대신
 *   "상위 5%" 방식으로 대체 표시할 것 (Frontend 작업 시 참고).
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
    private double curtailmentProbability; // 출력제어 확률 (0~1, sigmoid 보정값)

    @Column(length = 60)
    private String modelUsed; // 2026-09-26: 5경로 - classifier_{solar,solar_demand,wind,wind_demand,wind_demand_crossp}_calibrated_sigmoid

    private Double operationalThreshold; // 모델별 운영 임계값(0.02~0.43, 최대 20배 차이) - 0.03 고정 사용 금지

    private Boolean operationalThresholdReliable; // false면 임계값 확정 대신 상위 5%로만 표시 (note에도 안내)

    private double generationForecastMwh; // 컨버터 산출 발전량 예측치

    private Double curtailmentMwh; // 예상 출력제어량 (nullable) - SOLAR는 항상 null, WIND만 값 존재

    @Column(columnDefinition = "TEXT")
    private String note; // 세 경로(SOLAR·WIND·WIND+수요) 모두 존재 가능 - energy_type과
    // 무관하게 비어있지 않으면 저장/노출해야 함 (2026-09-25 AI 서버 변경사항 반영,
    // 특히 WIND+수요 경로엔 "ESS 계산에 쓰지 말라"는 경고문이 새로 추가돼 길어질 수 있어
    // length 제한(255) 대신 TEXT로 변경)

    @Column(nullable = false)
    private LocalDateTime predictedAt; // 예측이 생성된 시각
}
