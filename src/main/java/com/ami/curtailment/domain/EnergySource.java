package com.ami.curtailment.domain;

/**
 * 발전원 구분. 계획서 전체에서 태양광·풍력을 별도 모델로 다루므로 구분 필수.
 */
public enum EnergySource {
    SOLAR,  // 태양광
    WIND    // 풍력
}
