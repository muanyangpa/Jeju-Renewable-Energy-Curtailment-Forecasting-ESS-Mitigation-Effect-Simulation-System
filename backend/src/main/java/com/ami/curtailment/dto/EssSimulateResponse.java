package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class EssSimulateResponse {
    private double rated_power_mw;
    private String method;
    private double total_curtailment_mwh;
    private double total_absorbed_mwh;
    private double absorption_rate;

    // storage_constrained에서만 채워진다. 정격출력만 쓰는 방식은 저장용량 개념이 없어 null.
    private Double energy_capacity_mwh;
    private Double usable_capacity_mwh;
    private Integer hours_full;   // ESS가 가득 차서 더 흡수하지 못한 시간 수
    private Double annual_cycles; // 가용용량 기준 환산 사이클 수
}
