package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

/**
 * AI 서버 POST /predict 응답. app/schemas.py 실제 필드명 그대로.
 *
 * [2026-09-24 변경] note는 더 이상 태양광 전용이 아니다. 풍력 + demand_forecast_mw 경로에서도
 * ESS 경고문("expected_curtailment_mwh는 기댓값이므로 /ess/simulate에 넣지 말 것")이 실려 온다.
 * Frontend가 "태양광일 때만 note를 본다"는 전제로 분기하면 이 경고가 화면에 노출되지 않는다.
 * energy_type과 무관하게 note가 비어있지 않으면 노출할 것(AI 서버 README 명시).
 */
@Getter
@Setter
@NoArgsConstructor
public class PredictionResponse {
    private String energy_type;
    private String region;
    private String target_date;
    private List<HourlyPrediction> hourly;
    // curtailment_probability를 산출한 보정 분류모델 이름. 값을 하드코딩해 분기하지 말 것.
    // 2026-09-24 현재: classifier_{solar,wind,wind_demand}_calibrated_sigmoid
    // (2026-09-23 최초 추가 시에는 보정 전 이름이었고, 이후 _calibrated -> _calibrated_sigmoid로 두 번 바뀜)
    private String model_used;
    // 존재 조건: (1) 태양광 - expected_curtailment_mwh가 null인 이유,
    //           (2) 풍력 + 수요예측 생략 - 수요 미포함 모델로 전환됐다는 안내,
    //           (3) 풍력 + 수요예측 포함 - ESS 계산에 쓰지 말라는 경고 (2026-09-24 추가)
    private String note;
}
