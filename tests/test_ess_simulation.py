"""ESS 시뮬레이션의 물리적 불변식 — 데이터 없이 도는 순수 로직 테스트."""
from __future__ import annotations

import math

import pytest

from app.services.ess_simulation import (arbitrage_value_krw, simulate_hourly_capped,
                                         simulate_naive_upper_bound, simulate_with_storage)

# 제어가 10~17시에 몰린 하루 — 제주 풍력 실측 패턴을 단순화한 것
DAY = [0.0] * 9 + [40.0, 80.0, 120.0, 150.0, 140.0, 100.0, 50.0, 10.0] + [0.0] * 7
HOD = list(range(1, 25))


def test_흡수량은_제어량을_넘지_못한다():
    r = simulate_with_storage(DAY, 65, 260, HOD)
    assert r.total_absorbed_mwh <= r.total_curtailment_mwh + 1e-9
    assert 0.0 <= r.absorption_rate <= 1.0


def test_저장용량_제약은_정격출력_상한을_넘지_못한다():
    """simulate_with_storage는 simulate_hourly_capped의 부분집합이어야 한다.

    정격출력만 보는 방식은 저장용량이 무한하다고 가정하므로 항상 상한이다.
    이 관계가 깨지면 문서의 '이론적 상한' 서술이 거짓이 된다.
    """
    for mw in (10, 22.5, 65, 100):
        for hours in (1, 2, 4, 8):
            lo = simulate_with_storage(DAY, mw, mw * hours, HOD).total_absorbed_mwh
            hi = simulate_hourly_capped(DAY, mw).total_absorbed_mwh
            assert lo <= hi + 1e-9, f"{mw}MW/{hours}h: {lo} > 상한 {hi}"


def test_저장용량이_커지면_흡수량은_줄지_않는다():
    prev = -1.0
    for hours in range(1, 13):
        cur = simulate_with_storage(DAY, 65, 65 * hours, HOD).total_absorbed_mwh
        assert cur >= prev - 1e-9, f"{hours}h에서 흡수량이 감소했다"
        prev = cur


def test_정격출력이_커지면_흡수량은_줄지_않는다():
    prev = -1.0
    for mw in (5, 10, 20, 40, 65, 100, 200):
        cur = simulate_with_storage(DAY, mw, mw * 4, HOD).total_absorbed_mwh
        assert cur >= prev - 1e-9
        prev = cur


def test_방전_전달량과_흡수량의_비는_왕복효율이다():
    """충·방전에 효율을 대칭 배분하므로 delivered/absorbed == round_trip_efficiency.

    한쪽만 적용하면 sqrt(eta)가 나온다 — 환산 가치가 5% 부풀려지는 버그다.
    """
    for eta in (0.85, 0.90, 0.94):
        r = simulate_with_storage(DAY * 30, 65, 260, HOD * 30, round_trip_efficiency=eta)
        assert r.delivered_mwh is not None
        assert math.isclose(r.delivered_mwh / r.total_absorbed_mwh, eta, rel_tol=0.02)


def test_가용용량은_SoC_범위만큼이다():
    r = simulate_with_storage(DAY, 65, 260, HOD, soc_min=0.1, soc_max=0.9)
    assert math.isclose(r.usable_capacity_mwh, 260 * 0.8)


def test_제어가_없으면_흡수율은_0이고_예외가_없다():
    r = simulate_with_storage([0.0] * 24, 65, 260, HOD)
    assert r.total_absorbed_mwh == 0.0 and r.absorption_rate == 0.0
    assert r.hours_full == 0


def test_음수_제어량은_0으로_취급한다():
    a = simulate_with_storage([-5.0, 50.0] + [0.0] * 22, 65, 260, HOD)
    b = simulate_with_storage([0.0, 50.0] + [0.0] * 22, 65, 260, HOD)
    assert a.total_curtailment_mwh == b.total_curtailment_mwh


def test_hour_of_day를_생략하면_1시부터_센다():
    a = simulate_with_storage(DAY, 65, 260)
    b = simulate_with_storage(DAY, 65, 260, HOD)
    assert math.isclose(a.total_absorbed_mwh, b.total_absorbed_mwh)


def test_방전_시간이_없으면_한_번_채우고_멈춘다():
    """방전 없이는 가용용량 이상 흡수할 수 없다 — 저장용량 제약이 실제로 작동하는지 확인."""
    r = simulate_with_storage(DAY * 5, 65, 260, HOD * 5, discharge_hours=())
    assert r.total_absorbed_mwh <= r.usable_capacity_mwh / math.sqrt(0.90) + 1e-6


def test_차익거래_가치는_방전량_기준이다():
    r = simulate_with_storage(DAY, 65, 260, HOD)
    v = arbitrage_value_krw(r, discharge_price_krw_per_kwh=200.0, charge_price_krw_per_kwh=0.0)
    # revenue_krw는 원 단위로 반올림된 값이라 abs_tol=1로 비교한다
    assert math.isclose(v["revenue_krw"], r.delivered_mwh * 1000 * 200.0, abs_tol=1.0)
    assert v["net_value_krw"] == v["revenue_krw"] - v["charge_cost_krw"]
    # 흡수량(충전측)으로 계산한 값보다 작아야 한다 — 왕복효율 손실이 반영됐다는 뜻
    assert v["net_value_krw"] < r.total_absorbed_mwh * 1000 * 200.0


def test_차익거래_가치는_충전비용을_차감한다():
    r = simulate_with_storage(DAY, 65, 260, HOD)
    free = arbitrage_value_krw(r, 200.0, 0.0)["net_value_krw"]
    paid = arbitrage_value_krw(r, 200.0, 50.0)["net_value_krw"]
    assert paid < free


def test_정격출력만_쓰는_방식에는_차익거래_가치를_낼_수_없다():
    """방전을 모델링하지 않으므로 delivered_mwh가 없다 — 조용히 0을 주면 안 된다."""
    with pytest.raises(ValueError, match="delivered_mwh"):
        arbitrage_value_krw(simulate_hourly_capped(DAY, 65))


def test_단순_상한은_매시간_상한보다_크거나_같다():
    hours = sum(1 for v in DAY if v > 0)
    total = sum(DAY)
    naive = simulate_naive_upper_bound(hours, 22.5, total).total_absorbed_mwh
    capped = simulate_hourly_capped(DAY, 22.5).total_absorbed_mwh
    assert naive >= capped - 1e-9
