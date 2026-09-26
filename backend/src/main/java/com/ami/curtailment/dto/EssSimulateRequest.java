package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * AI 서버 POST /ess/simulate 요청. 04장 02(기본 시뮬레이션)와 03(용량 조정 슬라이더)이 공유.
 * 슬라이더는 정격출력(MW)·저장용량(MWh) 2축을 바꿔가며 이 엔드포인트를 반복 호출.
 *
 * method 기본값 storage_constrained는 저장용량·왕복효율·SoC를 반영한 현실 추정이다.
 * 정격출력만 보는 hourly_capped는 ESS가 무한히 받는다고 가정하므로 이론적 상한이며,
 * 같은 65MW 설비에서 두 방식의 차이가 72.7% vs 52.5%(2023년 실측 풍력 제어량 기준)로
 * 20%p 벌어진다 - 저장용량을 빼고 말하면 흡수율이 과대평가된다.
 */
@Getter
@Setter
@NoArgsConstructor
public class EssSimulateRequest {
    private List<Double> hourly_curtailment_mwh; // 시간별 출력제어량
    private double rated_power_mw;                // ESS 정격출력 (슬라이더 1축). 기본 65 = 제주 장주기 BESS 중앙계약시장 물량
    private String method;                         // "storage_constrained"(정식) | "hourly_capped"(이론 상한) | "naive_upper_bound"(단순 상한)

    // --- method="storage_constrained" 전용. null로 보내면 AI 서버 기본값(4시간, 효율 0.90, SoC 0.10~0.90) 적용 ---
    private Double energy_capacity_mwh;    // ESS 저장용량 (슬라이더 2축). 미지정 시 rated_power_mw x 4시간
    private Double round_trip_efficiency;  // 왕복효율. 상용 BESS AC 85~94% 중 0.90
    private Double soc_min;                // 최소 충전상태
    private Double soc_max;                // 최대 충전상태
    private List<Integer> hour_of_day;     // 각 시간의 시각(1~24). 미지정 시 1시 시작 연속 시계열로 간주
}
