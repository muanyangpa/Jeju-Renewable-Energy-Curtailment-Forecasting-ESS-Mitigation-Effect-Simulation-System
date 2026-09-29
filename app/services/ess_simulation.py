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

import math
from dataclasses import dataclass, field


@dataclass
class HourlyEssResult:
    hour_index: int
    curtailment_mwh: float
    absorbed_mwh: float


@dataclass
class EssSimulationResult:
    rated_power_mw: float
    method: str  # "naive_upper_bound" | "hourly_capped" | "storage_constrained"
    total_curtailment_mwh: float
    total_absorbed_mwh: float
    absorption_rate: float  # 0~1
    hourly: list[HourlyEssResult] = field(default_factory=list)
    # storage_constrained 전용 진단값 (다른 방식에서는 None)
    energy_capacity_mwh: float | None = None
    usable_capacity_mwh: float | None = None
    hours_full: int | None = None
    annual_cycles: float | None = None
    delivered_mwh: float | None = None  # 계통에 실제로 되팔린 양 = 흡수량 x 왕복효율

    def to_dict(self) -> dict:
        d = {
            "rated_power_mw": self.rated_power_mw,
            "method": self.method,
            "total_curtailment_mwh": round(self.total_curtailment_mwh, 1),
            "total_absorbed_mwh": round(self.total_absorbed_mwh, 1),
            "absorption_rate": round(self.absorption_rate, 4),
            "hourly": [h.__dict__ for h in self.hourly],
        }
        if self.energy_capacity_mwh is not None:
            d.update({
                "energy_capacity_mwh": round(self.energy_capacity_mwh, 1),
                "usable_capacity_mwh": round(self.usable_capacity_mwh, 1),
                "hours_full": self.hours_full,
                "annual_cycles": round(self.annual_cycles, 1),
                "delivered_mwh": round(self.delivered_mwh, 1),
            })
        return d


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


# 2023년 제주 평균 SMP(EPSIS 연평균). 충전 시점 가격은 별도로 둔다 — 아래 참고.
SMP_2023_AVG_KRW_PER_KWH = 172.5
# 하루전 SMP(제주시범사업, 2024-03~2026-09) 기준 저녁 18~22시 평균은 전체 평균의 1.1449배다.
# 2023년의 시간별 하루전 계열은 존재하지 않으므로 이 비율로 환산해 방전 시점 가격을 만든다.
EVENING_SMP_RATIO = 1.1449
# 제주 용량요금 단가. 본 계산에는 포함하지 않는다(아래 주석 참고).
JEJU_CAPACITY_PAYMENT_KRW_PER_KWH = 22.05


def arbitrage_value_krw(result: EssSimulationResult,
                        discharge_price_krw_per_kwh: float | None = None,
                        charge_price_krw_per_kwh: float = 0.0) -> dict:
    """ESS가 제어량을 흡수해서 얻는 가치 — 차익거래(load shifting) 기준.

    [왜 '흡수량 x 연평균 SMP'가 아닌가]
    두 가지가 빠진다. 첫째, 흡수량은 충전측 양이고 왕복효율만큼 손실되므로 계통에 되팔리는
    것은 흡수량 x 왕복효율이다. 둘째, 충전과 방전의 가격이 다르다 — 이 프로젝트 자신의
    발견대로 제어가 일어나는 시간의 하루전 SMP는 0 이하이고(대리 라벨의 전제), 방전은
    저녁이라 가격이 더 높다.

    [문헌 근거] 전력거래소 공저 논문(Journal of Climate Change Research 13(1), 전우영·김진이·
    이성우)은 ESS 편익을 네 가지로 나눈다 — ①차익거래 수익(SMP 낮을 때 충전·높을 때 방전,
    "출력제한이 발생할 경우 SMP는 0원 혹은 육지 SMP가 적용되기 때문에 차익거래를 최적으로
    추구하다보면 출력제한은 자연스럽게 최소화된다"), ②용량요금(제주 22.05원/kWh),
    ③주파수제어 예비력 용량가치, ④실제 주파수제어 정산금. 왕복효율(Roundtrip Efficiency)도
    그 최적화 모형의 명시적 변수다. 에너지경제연구원 이슈페이퍼 23-09도 같은 구조를 전제로
    "BESS 용량이 증가할 경우 SMP가 0원인 시간이 줄어들어 차익거래의 편익이 감소"한다고 본다.

    이 함수는 ①만 계산한다. ②~④는 계통 서비스 대가로 출력제어 흡수와 별개이며, 포함하면
    '제어 회피로 생긴 가치'가 부풀려진다. 즉 이 값은 보수적인 하한이다.

    charge_price_krw_per_kwh 기본값 0은 제어 시간의 하루전 SMP가 0 이하라는 실측에 근거한다.
    음수 SMP 구간에서는 충전이 오히려 수익이므로 0은 보수적인 가정이다.
    """
    if result.delivered_mwh is None:
        raise ValueError("delivered_mwh가 없습니다 — simulate_with_storage 결과만 지원합니다 "
                         "(정격출력만 보는 방식은 방전을 모델링하지 않습니다)")
    price = (discharge_price_krw_per_kwh
             if discharge_price_krw_per_kwh is not None
             else SMP_2023_AVG_KRW_PER_KWH * EVENING_SMP_RATIO)
    revenue = result.delivered_mwh * 1000 * price
    cost = result.total_absorbed_mwh * 1000 * charge_price_krw_per_kwh
    return {
        "absorbed_mwh": round(result.total_absorbed_mwh, 1),
        "delivered_mwh": round(result.delivered_mwh, 1),
        "discharge_price_krw_per_kwh": round(price, 1),
        "charge_price_krw_per_kwh": charge_price_krw_per_kwh,
        "revenue_krw": round(revenue),
        "charge_cost_krw": round(cost),
        "net_value_krw": round(revenue - cost),
        "net_value_eok": round((revenue - cost) / 1e8, 1),
        "excluded": "용량요금·예비력 정산금 제외 (계통 서비스 대가이므로 제어 회피 가치와 분리)",
    }


DEFAULT_DISCHARGE_HOURS = (18, 19, 20, 21, 22)


def simulate_with_storage(
    hourly_curtailment_mwh: list[float],
    rated_power_mw: float,
    energy_capacity_mwh: float,
    hour_of_day: list[int] | None = None,
    round_trip_efficiency: float = 0.90,
    soc_min: float = 0.10,
    soc_max: float = 0.90,
    discharge_hours: tuple[int, ...] = DEFAULT_DISCHARGE_HOURS,
) -> EssSimulationResult:
    """③ 저장용량·충전상태·왕복효율을 반영한 현실 추정 (2026-09-26 추가).

    왜 필요한가:
      simulate_hourly_capped는 정격출력(MW)만 보고 저장용량(MWh)을 무시한다. 계획서 07장이
      정의한 방식이고 문서값(22.5MW 36.8%)을 재현하는 기준점이지만, ESS가 무한히 받는다고
      가정하므로 '이론적 상한'이다. 2023년 제주 풍력 제어는 연속 평균 4.7시간·최장 9시간
      (연 120회) 이어지고 시간당 평균 46.5MWh라, 4시간짜리 ESS는 중간에 가득 차서 더 못 받는다.
      같은 65MW 설비에서 상한 72.7% vs 저장용량 260MWh 반영 52.5% — 20.2%p 차이다.

    [공개 데이터의 한계] 한국전력거래소 「신재생연계ESS 설비용량 자료」는 PCS 용량(MW)만
    제공하고 저장용량(MWh)은 제공하지 않는다. 그래서 신재생연계 ESS(제주 풍력연계 22.5MW,
    태양광연계 14.3MW)는 duration을 가정할 수밖에 없다. 반면 장주기 BESS 중앙계약시장
    물량은 65MW/260MWh(4시간)로 둘 다 공개돼 있어 근거가 확실하다.

    [총량 계산이 타당한 범위] 신재생연계 ESS는 REC 가중치 요건상 '연계된 자기 발전기'에서만
    충전할 수 있어 발전소마다 흩어져 있다. 제주 전체 제어량에 그 합계를 대입하면 과대평가다.
    장주기 BESS는 계통에서 충전하므로 총량 기반 계산이 타당하다 — 이 함수의 대표 시나리오를
    65MW/260MWh로 잡는 이유다.

    파라미터 근거:
      energy_capacity_mwh  제주 장주기 BESS 중앙계약시장 입찰 물량 65MW/260MWh (4시간)
      round_trip_efficiency 0.90 — 상용 BESS AC 왕복효율 85~94% 범위의 중앙값
      soc_min/soc_max      0.10/0.90 — 수명 관리 관행 [가정]
      discharge_hours      18~22시 — 2023년 풍력 제어량의 99.0%가 10~17시에 몰려 있고
                           (13~14시만 42.7%) 18~22시는 0.2%(44MWh)뿐이라 방전 시간이 확보된다.
                           '항상 방전 가능' 가정과 결과가 같고(65MW/260MWh에서 둘 다 52.5%),
                           피크 19~21시로 좁혀도 51.6%라 방전 가정의 민감도는 낮다.
                           지배적인 제약은 저장용량이다.

    hour_of_day를 주지 않으면 입력이 1시부터 시작하는 연속 시계열이라고 보고 1~24를 반복한다.
    """
    n = len(hourly_curtailment_mwh)
    hod = list(hour_of_day) if hour_of_day is not None else [(i % 24) + 1 for i in range(n)]
    lo, hi = energy_capacity_mwh * soc_min, energy_capacity_mwh * soc_max
    usable = hi - lo
    k = math.sqrt(round_trip_efficiency)

    soc = lo
    hourly: list[HourlyEssResult] = []
    total_curt = total_abs = discharged = delivered = 0.0
    hours_full = 0
    for c, h in zip(hourly_curtailment_mwh, hod):
        c = max(0.0, c)
        absorbed = 0.0
        if c > 0:
            room = max(0.0, hi - soc)
            if room <= 1e-9:
                hours_full += 1
            absorbed = min(c, rated_power_mw, room / k)
            soc += absorbed * k
        elif h in discharge_hours:
            out = min(rated_power_mw, soc - lo)
            soc -= out
            discharged += out
            # 저장고에서 나온 out 중 계통에 도달하는 것은 out x 방전효율이다.
            # 왕복효율을 충·방전에 대칭 배분하므로 delivered/absorbed = round_trip_efficiency가 된다.
            delivered += out * k
        hourly.append(HourlyEssResult(hour_index=len(hourly), curtailment_mwh=c, absorbed_mwh=absorbed))
        total_curt += c
        total_abs += absorbed
    rate = (total_abs / total_curt) if total_curt > 0 else 0.0
    return EssSimulationResult(
        rated_power_mw=rated_power_mw, method="storage_constrained",
        total_curtailment_mwh=total_curt, total_absorbed_mwh=total_abs,
        absorption_rate=rate, hourly=hourly,
        energy_capacity_mwh=energy_capacity_mwh, usable_capacity_mwh=usable,
        hours_full=hours_full, annual_cycles=(discharged / usable) if usable > 0 else 0.0,
        delivered_mwh=delivered,
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
