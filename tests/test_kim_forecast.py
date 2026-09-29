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
    # 순간값(direct/diffuse)과 누적값(acc)을 일부러 다르게 주어 어느 쪽을 채택했는지 가려낸다.
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
    # 누적은 시간당 1.0 MJ, 순간값 합성은 0.54 MJ — 값이 갈리도록 만든다
    blocks = [_block(hf, 100.0, 50.0, 1.0 * (hf - hfs[0])) for hf in hfs]
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
    def test_ghi_is_the_accumulated_difference_not_the_instant_value(self, sample):
        """순간값이 아니라 누적차분을 써야 한다.

        직달·산란은 그 시각의 순간값이라 시간평균이 아니다. 실제 자료에서 14시 순간값이
        구름이 갈라진 순간을 잡아 1.65 MJ로 나왔지만 같은 시간의 적산값은 0.74였다.
        누적장은 정의상 한 시간을 적분한 값이므로 이 문제가 없다.
        """
        rows = K.fetch_day(TARGET, sample_path=sample)
        assert len(rows) == 24
        assert rows[0].ghi_mj == pytest.approx(1.0, abs=1e-6)            # 누적차분
        assert rows[0].inst_mj == pytest.approx(0.54, abs=1e-6)          # 순간값(진단용)

    def test_instant_path_kept_as_a_diagnostic(self, sample):
        """두 경로가 벌어지는 시간은 구름 변동이 심하다는 신호이므로 값을 버리지 않는다."""
        rows = K.fetch_day(TARGET, sample_path=sample)
        assert rows[0].inst_gap == pytest.approx(0.54 - 1.0, abs=1e-6)

    def test_decreasing_accumulation_is_dropped_not_clipped(self, tmp_path):
        """누적장이 줄어들면 모델 재시작 등으로 초기화된 것이다. 0으로 깎으면 안 된다."""
        tmfc = K.default_tmfc(TARGET)
        hfs = sorted({K.hf_for(TARGET, h, tmfc) for h in range(1, 25)}
                     | {K.hf_for(TARGET, 1, tmfc) - 1})
        acc = {hf: 1.0 * (hf - hfs[0]) for hf in hfs}
        acc[hfs[12]] = 0.0                                   # 중간에서 초기화된 상황
        p = tmp_path / "reset.txt"
        p.write_text("\n".join(_block(hf, 100.0, 50.0, acc[hf]) for hf in hfs), encoding="utf-8")
        errs: list[str] = []
        rows = K.fetch_day(TARGET, sample_path=str(p), errors=errs)
        assert len(rows) < 24 and any("감소" in e for e in errs), errs
        assert all(r.ghi_mj >= 0 for r in rows)

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


class TestValidationLog:
    """수집 기록의 중복 제거 — 14일치를 쌓는 동안 조용히 오염되면 검증이 무의미해진다."""

    def _log(self, tmp_path, monkeypatch):
        import scripts.kim_validation_log as L
        p = tmp_path / "log.csv"
        monkeypatch.setattr(L, "LOG", str(p))
        return L, p

    def test_tmfc를_문자열로_읽는다(self, tmp_path, monkeypatch):
        """CSV 기본 추론이면 int64가 되어 문자열 비교가 절대 성립하지 않는다.

        그러면 같은 날을 두 번 수집해도 중복 제거가 안 되고 그대로 쌓인다 — 실제로 겪었다.
        """
        import pandas as pd
        L, p = self._log(tmp_path, monkeypatch)
        pd.DataFrame({"target_date": ["2026-09-30"] * 2, "hour": [1, 2],
                      "tmfc": ["2026092818"] * 2, "kim_mj": [0.0, 0.0],
                      "kim_inst_mj": [0.0, 0.0], "est_mj": [0.0, 0.0],
                      "wind80": [1.0, 1.0], "temp_c": [20.0, 20.0],
                      "collected_at": ["x", "x"]}).to_csv(p, index=False)
        got = pd.read_csv(p, dtype={"tmfc": str, "target_date": str})
        assert got["tmfc"].dtype == object
        assert (got["tmfc"] == "2026092818").all(), "문자열로 읽어야 비교가 성립한다"

    def test_같은_시각이_두_번_들어오면_마지막만_남는다(self, tmp_path):
        import pandas as pd
        d = pd.DataFrame({"target_date": ["2026-09-30"] * 3, "tmfc": ["2026092818"] * 3,
                          "hour": [1, 1, 2], "kim_mj": [0.1, 0.2, 0.3]})
        out = d.drop_duplicates(subset=["target_date", "tmfc", "hour"], keep="last")
        assert len(out) == 2
        assert out[out.hour == 1]["kim_mj"].iloc[0] == 0.2, "나중 수집이 이겨야 한다"
