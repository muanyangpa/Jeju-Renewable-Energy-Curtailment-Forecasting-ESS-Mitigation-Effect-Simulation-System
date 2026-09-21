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
}
