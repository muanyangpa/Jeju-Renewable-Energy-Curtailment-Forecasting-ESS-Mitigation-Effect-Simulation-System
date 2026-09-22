package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * AI 서버 POST /ess/simulate 요청. 04장 02(기본 시뮬레이션)와 03(용량 조정 슬라이더)이 공유.
 * 슬라이더는 rated_power_mw만 바꿔가며 이 엔드포인트를 반복 호출.
 */
@Getter
@Setter
@NoArgsConstructor
public class EssSimulateRequest {
    private List<Double> hourly_curtailment_mwh; // 시간별 출력제어량
    private double rated_power_mw;                // ESS 정격출력 (슬라이더 조정값)
    private String method;                         // "hourly_capped"(정식) | "naive_upper_bound"(단순 상한)
}
