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
 * - curtailmentProbability는 sigmoid 확률 보정이 적용된 값. 0.5 같은 고정 임계값 금지,
 *   제어 발생 판정 기준은 0.03 (Frontend 등급 표시 시 참고).
 * - note는 SOLAR 전용이 아니라 세 경로 모두 올 수 있음(note 컬럼 주석 참고).
 * - ESS 흡수율 계산에는 curtailmentMwh(예측값)를 쓰지 말 것 - 실측 제어량 기반만 사용
 *   (예측값 사용 시 실측 36.8%가 78%로 부풀려짐, AI 서버팀 확인).
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

    @Column(columnDefinition = "TEXT")
    private String note; // 세 경로(SOLAR·WIND·WIND+수요) 모두 존재 가능 - energy_type과
    // 무관하게 비어있지 않으면 저장/노출해야 함 (2026-09-25 AI 서버 변경사항 반영,
    // 특히 WIND+수요 경로엔 "ESS 계산에 쓰지 말라"는 경고문이 새로 추가돼 길어질 수 있어
    // length 제한(255) 대신 TEXT로 변경)

    @Column(nullable = false)
    private LocalDateTime predictedAt; // 예측이 생성된 시각
}
