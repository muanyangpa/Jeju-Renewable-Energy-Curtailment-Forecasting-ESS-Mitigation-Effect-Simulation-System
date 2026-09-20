"""
ESS 충방전 시뮬레이션 로직 — 계획서 04장 02번(ESS 충방전 시뮬레이션), 03번(ESS 용량 조정 시뮬레이터) 및
07장 "핵심 지표 — ESS 완화효과 정량화"에서 검증한 두 가지 계산 방식을 그대로 코드화한다.

두 계산 방식 (07장 원문 그대로):
  ① 단순 상한(naive upper bound): ESS가 출력제어 발생 시간 내내 정격출력으로 풀가동된다고 가정
     = rated_power_mw * curtailed_hours
  ② 물리적으로 더 정확한 방식(정식 채택): 시간별로 min(그 시각의 실제 출력제어량, ESS 정격출력)을 합산
     = sum(min(hourly_curtailment_mwh, rated_power_mw))

04장 03번(ESS 용량 조정 시뮬레이터)은 ②번 계산 로직을 다른 rated_power_mw 값으로 재실행하는 것과
정확히 같다 — 그래서 별도 모델 없이 이 함수 하나만 재사용한다.

주의: 이 모듈은 시간별 출력제어량(MWh)이 입력으로 주어졌을 때의 흡수율만 계산한다.
태양광은 이 MWh 값 자체가 없으므로(07장 참고), 태양광에 대해 이 모듈을 호출할 때는
호출자가 예측 확률 등으로 만든 "추정치"임을 명확히 하고 04장/07장의 한계를 함께 안내해야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HourlyEssResult:
    hour_index: int
    curtailment_mwh: float
    absorbed_mwh: float


@dataclass
class EssSimulationResult:
    rated_power_mw: float
    method: str  # "naive_upper_bound" | "hourly_capped"
    total_curtailment_mwh: float
    total_absorbed_mwh: float
    absorption_rate: float  # 0~1
    hourly: list[HourlyEssResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rated_power_mw": self.rated_power_mw,
            "method": self.method,
            "total_curtailment_mwh": round(self.total_curtailment_mwh, 1),
            "total_absorbed_mwh": round(self.total_absorbed_mwh, 1),
            "absorption_rate": round(self.absorption_rate, 4),
            "hourly": [h.__dict__ for h in self.hourly],
        }


def simulate_naive_upper_bound(curtailed_hours: int, rated_power_mw: float, total_curtailment_mwh: float) -> EssSimulationResult:
    """① 단순 상한 — ESS가 제어 발생 시간 내내 정격출력 풀가동된다고 가정 (07장 방식①)."""
    absorbed = rated_power_mw * curtailed_hours
    absorbed = min(absorbed, total_curtailment_mwh) if total_curtailment_mwh > 0 else absorbed
    rate = (absorbed / total_curtailment_mwh) if total_curtailment_mwh > 0 else 0.0
    return EssSimulationResult(
        rated_power_mw=rated_power_mw, method="naive_upper_bound",
        total_curtailment_mwh=total_curtailment_mwh, total_absorbed_mwh=absorbed,
        absorption_rate=rate,
    )


def simulate_hourly_capped(hourly_curtailment_mwh: list[float], rated_power_mw: float) -> EssSimulationResult:
    """② 물리적으로 더 정확한 방식 — 매 시간 min(제어량, ESS 정격출력) (07장 방식②, 정식 채택).

    04장 03번 'ESS 용량 조정 시뮬레이터'는 rated_power_mw만 바꿔가며 이 함수를 재호출하는 것.
    """
    hourly: list[HourlyEssResult] = []
    total_curt = 0.0
    total_abs = 0.0
    for i, c in enumerate(hourly_curtailment_mwh):
        c = max(0.0, c)
        absorbed = min(c, rated_power_mw)
        hourly.append(HourlyEssResult(hour_index=i, curtailment_mwh=c, absorbed_mwh=absorbed))
        total_curt += c
        total_abs += absorbed
    rate = (total_abs / total_curt) if total_curt > 0 else 0.0
    return EssSimulationResult(
        rated_power_mw=rated_power_mw, method="hourly_capped",
        total_curtailment_mwh=total_curt, total_absorbed_mwh=total_abs,
        absorption_rate=rate, hourly=hourly,
    )


def capacity_sweep(hourly_curtailment_mwh: list[float], capacities_mw: list[float]) -> list[dict]:
    """04장 03번 슬라이더용 — 여러 ESS 용량에 대해 한 번에 흡수율 계산."""
    return [simulate_hourly_capped(hourly_curtailment_mwh, cap).to_dict() for cap in capacities_mw]


if __name__ == "__main__":
    # 07장 수치 재현 검증: 2023년 풍력 26,197MWh/563시간, 22.5MW ESS
    from app.data_prep import load_curtailment

    curt = load_curtailment("wind")
    y2023 = curt[(curt["dt"] >= "2023-01-01") & (curt["dt"] < "2024-01-01")]
    hourly_mwh = y2023["curtailment_mwh"].tolist()
    curtailed_hours = int((y2023["curtailment_mwh"] > 0).sum())
    total = float(y2023["curtailment_mwh"].sum())

    naive = simulate_naive_upper_bound(curtailed_hours, 22.5, total)
    capped = simulate_hourly_capped(hourly_mwh, 22.5)
    print("① 단순 상한:", naive.to_dict()["total_absorbed_mwh"], "MWh",
          f"({naive.absorption_rate:.1%})", "— 07장 문서값: 12,668MWh(48.4%)")
    print("② min() 방식:", capped.to_dict()["total_absorbed_mwh"], "MWh",
          f"({capped.absorption_rate:.1%})", "— 07장 문서값: 9,632MWh(36.8%)")
