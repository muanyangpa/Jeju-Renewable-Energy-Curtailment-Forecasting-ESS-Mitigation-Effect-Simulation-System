"""KIM 일사량 수집 모듈 — 인증키·네트워크 없이 검증 가능한 부분만 다룬다.

시각 변환(UTC tmfc <-> KST 구간끝 시각)과 '전부 아니면 폴백' 규칙이 핵심이다.
둘 다 조용히 틀리면 일사량이 몇 시간 밀린 채로 서비스되므로 반드시 고정해 둔다.
"""
from __future__ import annotations

import datetime as dt

import pytest

from app.services import kim_forecast as K

TARGET = dt.date(2026, 9, 28)


def _line(tmef: str, value: float, name: str) -> str:
    # 실제 응답 형식: TMEF(10자리) 다음에 3개 필드, 5번째가 값, 그 뒤가 이름(단위).
    return f"{tmef} 2026092618 000 000 {value} {name}"


def _block(hf: int, direct: float, diffuse: float, acc: float) -> str:
    t = (dt.datetime(2026, 9, 26, 18) + dt.timedelta(hours=hf)).strftime("%Y%m%d%H")
    return "\n".join([
        f"#hf={hf}",
        "# 주석 줄은 무시된다",
        _line(t, direct, "SWDDIR2(W m-2)"),
        _line(t, diffuse, "SWDDIF2(W m-2)"),
        _line(t, acc, "ACSWDNB(MJ m-2)"),
        _line(t, -3.0, "U80(m s-1)"),
        _line(t, -4.0, "V80(m s-1)"),
        _line(t, 298.15, "T2(K)"),
    ])


@pytest.fixture
def sample(tmp_path):
    """거래일 24시간 + 누적차분용 앞시각 1개를 담은 가짜 응답 파일."""
    tmfc = K.default_tmfc(TARGET)
    hfs = sorted({K.hf_for(TARGET, h, tmfc) for h in range(1, 25)}
                 | {K.hf_for(TARGET, 1, tmfc) - 1})
    blocks = [_block(hf, 100.0, 50.0, 0.54 * (hf - hfs[0])) for hf in hfs]
    p = tmp_path / "kim.txt"
    p.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return str(p)


class TestTimeConversion:
    def test_default_tmfc_is_two_days_before_18utc(self):
        # 하루전시장 입찰마감이 D-1 11시 KST이므로 D-1 00 UTC는 늦다. 한 사이클 앞을 써야 한다.
        assert K.default_tmfc(TARGET) == "2026092618"

    def test_default_tmfc_is_a_model_cycle(self):
        for d in (dt.date(2026, 1, 1), dt.date(2026, 6, 15), dt.date(2027, 3, 3)):
            assert int(K.default_tmfc(d)[8:]) in (0, 6, 12, 18)

    def test_hf_covers_the_trading_day_within_48h(self):
        tmfc = K.default_tmfc(TARGET)
        hfs = [K.hf_for(TARGET, h, tmfc) for h in range(1, 25)]
        assert hfs == list(range(22, 46))          # 연속 24개, 예보범위 48h 안
        assert max(hfs) <= 48

    def test_hour_24_is_next_day_midnight(self):
        # 이 저장소 규약: h시 = 그 구간의 끝. 24시는 다음날 00시다.
        tmfc = K.default_tmfc(TARGET)
        assert K.hf_for(TARGET, 24, tmfc) == K.hf_for(TARGET, 23, tmfc) + 1
        assert K.hf_for(TARGET, 24, tmfc) == K.hf_for(
            TARGET + dt.timedelta(days=1), 0, tmfc)


class TestParsing:
    def test_unit_with_space_does_not_break_the_name(self):
        # U80(m s-1)처럼 단위에 공백이 있어 단순 split으로는 이름이 잘린다 — 실제로 겪은 문제다.
        v = K._parse(_line("2026092813", -3.0, "U80(m s-1)"))
        assert v == {"U80": -3.0}

    def test_comment_and_blank_lines_ignored(self):
        assert K._parse("# 머리말\n\n   \n") == {}

    def test_sample_blocks_split_by_hf(self, sample):
        blocks = K.load_sample(sample)
        assert len(blocks) == 25                   # 24시간 + 누적차분용 앞시각
        assert all("SWDDIR2" in v for v in blocks.values())


class TestFetchDay:
    def test_ghi_is_direct_plus_diffuse_in_mj(self, sample):
        rows = K.fetch_day(TARGET, sample_path=sample)
        assert len(rows) == 24
        assert rows[0].ghi_mj == pytest.approx((100.0 + 50.0) * 0.0036, abs=1e-6)

    def test_accumulated_difference_cross_checks_the_composition(self, sample):
        # 샘플은 시간당 0.54 MJ씩 누적하도록 만들었고 합성값도 0.54다 — 두 경로가 맞아야 한다.
        rows = K.fetch_day(TARGET, sample_path=sample)
        assert all(abs(r.ghi_mj - r.acc_mj) < 1e-6 for r in rows)

    def test_hub_height_wind_is_the_vector_magnitude(self, sample):
        rows = K.fetch_day(TARGET, sample_path=sample)
        assert rows[0].wind80 == pytest.approx(5.0, abs=0.01)   # hypot(3,4)

    def test_hours_are_1_to_24(self, sample):
        assert [r.hour for r in K.fetch_day(TARGET, sample_path=sample)] == list(range(1, 25))

    def test_temperature_converted_to_celsius(self, sample):
        assert K.fetch_day(TARGET, sample_path=sample)[0].temp_c == pytest.approx(25.0, abs=0.1)


class TestOverrideIsAllOrNothing:
    def test_full_day_returns_24_hours(self, sample):
        o = K.radiation_override(TARGET, sample_path=sample)
        assert o is not None and sorted(o) == list(range(1, 25))

    def test_partial_day_falls_back_with_a_reason(self, tmp_path):
        # 일부만 예보값이고 일부는 추정값인 상태를 만들면 안 된다 — 전부 폴백해야 한다.
        tmfc = K.default_tmfc(TARGET)
        hfs = [K.hf_for(TARGET, h, tmfc) for h in range(1, 20)]
        p = tmp_path / "part.txt"
        p.write_text("\n".join(_block(hf, 100.0, 50.0, 0.5) for hf in hfs), encoding="utf-8")
        note: list[str] = []
        assert K.radiation_override(TARGET, note=note, sample_path=str(p)) is None
        assert note and "24시간 중" in note[0]      # 사유를 삼키지 않는다

    def test_missing_file_falls_back_with_a_reason(self):
        note: list[str] = []
        assert K.radiation_override(TARGET, note=note, sample_path="/없는/경로.txt") is None
        assert note and "FileNotFoundError" in note[0]

    def test_no_key_raises_rather_than_silently_estimating(self, monkeypatch):
        monkeypatch.delenv("KMA_AUTH_KEY", raising=False)
        with pytest.raises(RuntimeError, match="KMA_AUTH_KEY"):
            K.fetch_day(TARGET)
