package com.ami.curtailment.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

/**
 * 시간별 기상 원본 데이터. AI 서버 app/schemas.py 실제 필드명 그대로 사용 (snake_case 아님 -
 * Jackson이 자동 매핑하도록 JSON 프로퍼티명을 그대로 필드명으로 사용, 별도 @JsonProperty 불필요
 * 하게 필드명 자체를 스펙과 동일하게 맞춤).
 *
 * 통신규격 v1.1 05장 지적 반영 (2026-09-30): temp를 double(프리미티브)로 뒀더니 null/누락 시
 * Jackson이 0.0으로 채워 AI 서버의 422 검증을 우회, 발전량이 54% 부풀려진 채 200으로 응답됨
 * (실측 1,032.6MWh → temp=0.0일 때 1,588.7MWh). Double(래퍼)로 바꿔 null이 그대로 전달되게 함
 * — null이면 AI 서버가 의도대로 422 MISSING_REQUIRED_FIELD를 반환한다.
 *
 * 필수 여부는 energy_type에 따라 갈림(v1.1 02장 1번):
 * - "solar": solar_rad, temp, cloud 필수 / wind_speed는 선택
 * - "wind": wind_speed 필수 / 나머지는 선택이지만, 함께 보내면 crossp(계통 전체 침투율) 모델로
 *   전환되어 성능이 오름(PR-AUC 0.743→0.831) — 권장 사양이라 실질적으로는 매번 다 보내는 게 맞음
 */
@Getter
@Setter
@NoArgsConstructor
public class WeatherHour {
    private int hour;          // 1~24
    private Double solar_rad;  // 일사량
    private Double temp;       // 기온(℃)
    private Double cloud;      // 구름량
    private Double wind_speed; // 풍속(m/s)
}
