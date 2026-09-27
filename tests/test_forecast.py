"""예보 어댑터와 청천일사량 — 데이터·모델 없이 도는 순수 로직 + 아티팩트 필요 부분 분리."""
from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.solar_radiation import (STATIONS, clear_sky_mj, cloud_from_sky, latlon_to_grid,
                                 sky_from_cloud)
from tests.conftest import MODELS_DIR


def test_격자_변환이_공식_예시와_일치한다():
    """기준점 상수에 +1을 빼먹으면 전국이 (1,1)씩 밀린다 — 실제로 한 번 그랬다."""
    assert latlon_to_grid(37.5714, 126.9658) == (60, 127)   # 서울 종로구
    assert latlon_to_grid(33.4996, 126.5312) == (53, 38)    # 제주시청


def test_전운량과_SKY_코드_변환():
    """기상청 기준: 0~5 맑음(1), 6~8 구름많음(3), 9~10 흐림(4)."""
    assert list(sky_from_cloud([0, 5, 6, 8, 9, 10])) == [1, 1, 3, 3, 4, 4]
    # 역변환은 구간 대표값 — 정보가 복원되지는 않는다
    assert list(cloud_from_sky([1, 3, 4])) == [2.5, 7.0, 9.5]
    assert np.isnan(cloud_from_sky([2])[0]), "단기예보에 없는 코드는 NaN이어야 한다"


def test_청천일사량은_밤에_0이고_정오에_최대다():
    dt = pd.Series(pd.date_range("2026-06-21 01:00", periods=24, freq="h"))
    mj = clear_sky_mj(dt, "184")
    assert mj[0] == 0.0 and mj[-1] == 0.0, "자정 전후는 0이어야 한다"
    # h시 = 그 구간의 끝이므로 13~14시에 최대가 온다(정오 남중 + 30분 보정)
    assert 12 <= int(np.argmax(mj)) + 1 <= 15, f"최대 시각이 {int(np.argmax(mj))+1}시"


def test_하지가_동지보다_일사량이_많다():
    def total(d):
        return clear_sky_mj(pd.Series(pd.date_range(f"{d} 01:00", periods=24, freq="h")), "184").sum()
    assert total("2026-06-21") > 2 * total("2026-12-21")


@pytest.mark.parametrize("station", list(STATIONS))
def test_모든_지점의_청천일사량이_물리적_범위_안이다(station):
    dt = pd.Series(pd.date_range("2026-01-01 01:00", periods=24 * 365, freq="h"))
    mj = clear_sky_mj(dt, station)
    assert mj.min() >= 0.0
    # 시간 적산 수평면 일사량은 태양상수(1367 W/m² -> 4.92 MJ/m²/h)를 넘을 수 없다
    assert mj.max() < 4.92, f"{station}: 최대 {mj.max():.2f} MJ/m²가 물리 상한을 넘는다"


def _sample(tmp_path, hours=range(1, 25), sky=1):
    items = []
    for h in hours:
        hh = f"{h % 24:02d}00"
        for cat, val in (("TMP", "10"), ("WSD", "7.2"), ("SKY", str(sky)), ("REH", "65")):
            items.append({"category": cat, "fcstDate": "20260315",
                          "fcstTime": hh, "fcstValue": val})
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"response": {"header": {"resultCode": "00"},
                                          "body": {"items": {"item": items}}}}))
    return str(p)


needs_rad = pytest.mark.skipif(
    not (MODELS_DIR and __import__("os").path.exists(
        __import__("os").path.join(MODELS_DIR, "radiation_estimator.joblib"))),
    reason="radiation_estimator가 없습니다 — train_radiation_model을 실행하세요")


@needs_rad
def test_예보로_predict_요청을_만든다(tmp_path):
    from app.services.kma_forecast import build_predict_request
    body = build_predict_request("wind", date(2026, 3, 15), sample_path=_sample(tmp_path),
                                 demand_forecast_mw=[600.0] * 24)
    assert sorted(w["hour"] for w in body["weather"]) == list(range(1, 25))
    for w in body["weather"]:
        assert {"hour", "temp", "cloud", "wind_speed", "solar_rad"} <= set(w)
        assert w["solar_rad"] >= 0
    # 밤에는 일사량이 0이어야 한다
    night = [w["solar_rad"] for w in body["weather"] if w["hour"] in (1, 2, 3, 23, 24)]
    assert all(v == 0 for v in night), f"야간 일사량 {night}"


@needs_rad
def test_예보_시간이_부족하면_거부한다(tmp_path):
    """24개가 안 되면 조용히 넘기지 말고 실패해야 한다 — 발표시각을 앞당기라는 신호다."""
    from app.services.kma_forecast import build_predict_request
    with pytest.raises(ValueError, match="24개"):
        build_predict_request("wind", date(2026, 3, 15),
                              sample_path=_sample(tmp_path, hours=range(1, 20)))


@needs_rad
def test_맑음이_흐림보다_일사량이_많다(tmp_path):
    from app.services.kma_forecast import fetch
    clear = fetch(date(2026, 3, 15), sample_path=_sample(tmp_path, sky=1))
    tmp2 = tmp_path / "b"; tmp2.mkdir()
    cloudy = fetch(date(2026, 3, 15), sample_path=_sample(tmp2, sky=4))
    cs = sum(w["solar_rad"] for w in clear)
    cc = sum(w["solar_rad"] for w in cloudy)
    assert cs > cc * 1.5, f"맑음 {cs:.2f} vs 흐림 {cc:.2f} — 하늘상태가 반영되지 않았다"


# --- 인증키 처리 (네트워크 없이) ---

def test_키가_없으면_무엇을_해야_하는지_알려주며_실패한다(monkeypatch):
    from app.services.kma_forecast import PORTALS, resolve_key
    for cfg in PORTALS.values():
        monkeypatch.delenv(cfg["env"], raising=False)
    with pytest.raises(RuntimeError) as e:
        resolve_key()
    msg = str(e.value)
    for cfg in PORTALS.values():           # 두 포털의 환경변수 이름이 안내에 모두 나와야 한다
        assert cfg["env"] in msg
    assert "export" in msg, "무엇을 하라는 지시가 없으면 안내가 아니다"


def test_포털별_인증_파라미터_이름이_다르다():
    """API허브는 authKey, 공공데이터포털은 serviceKey — 이걸 섞으면 인증이 통째로 실패한다."""
    from app.services.kma_forecast import PORTALS
    assert PORTALS["apihub"]["key_param"] == "authKey"
    assert PORTALS["data.go.kr"]["key_param"] == "serviceKey"
    # 환경변수 이름을 파라미터 이름과 맞춰 뒀다 — 어느 포털 키인지 헷갈리지 않도록
    assert PORTALS["apihub"]["env"] == "KMA_AUTH_KEY"
    assert PORTALS["data.go.kr"]["env"] == "KMA_SERVICE_KEY"


def test_둘_다_있으면_API허브를_쓴다(monkeypatch):
    """수치모델 API가 허브에만 있어 키를 하나로 통일하는 쪽이 낫다."""
    from app.services.kma_forecast import resolve_key
    monkeypatch.setenv("KMA_AUTH_KEY", "a")
    monkeypatch.setenv("KMA_SERVICE_KEY", "b")
    assert resolve_key() == ("apihub", "a")
    assert resolve_key("data.go.kr") == ("data.go.kr", "b")


def test_잘못된_포털_이름은_거부한다(monkeypatch):
    from app.services.kma_forecast import resolve_key
    monkeypatch.setenv("KMA_AUTH_KEY", "a")
    with pytest.raises(ValueError, match="portal"):
        resolve_key("kma")


# ---------------------------------------------------------------------------
# KIM 일사량 예보로의 교체 — 예보값이 오면 추정 모델을 쓰지 않아야 한다
# ---------------------------------------------------------------------------

def _kim_sample(tmp_path, target=date(2026, 3, 15), hours=range(1, 25), direct=200.0):
    """KIM 응답 원문 형식의 가짜 샘플. tests/test_kim_forecast.py의 형식과 같다."""
    from app.services import kim_forecast as K
    import datetime as _dt
    tmfc = K.default_tmfc(target)
    t0 = _dt.datetime.strptime(tmfc, "%Y%m%d%H")
    hfs = sorted({K.hf_for(target, h, tmfc) for h in hours}
                 | {K.hf_for(target, min(hours), tmfc) - 1})
    out = []
    for hf in hfs:
        t = (t0 + _dt.timedelta(hours=hf)).strftime("%Y%m%d%H")
        out.append(f"#hf={hf}")
        for val, name in ((direct, "SWDDIR2(W m-2)"), (100.0, "SWDDIF2(W m-2)"),
                          (1.08 * (hf - hfs[0]), "ACSWDNB(MJ m-2)"),
                          (-3.0, "U80(m s-1)"), (-4.0, "V80(m s-1)"), (283.15, "T2(K)")):
            out.append(f"{t} {tmfc} 000 000 {val} {name}")
    p = tmp_path / "kim.txt"
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return str(p)


@needs_rad
def test_KIM_예보값이_오면_추정을_쓰지_않는다(tmp_path):
    """단기예보에 일사량이 없어 추정하던 자리를 KIM 국지모델 예보로 바꾼 것이 이 기능의 요점이다."""
    from app.services.kma_forecast import fetch
    k = tmp_path / "k"; k.mkdir()
    src: list[str] = []
    w = fetch(date(2026, 3, 15), sample_path=_sample(tmp_path),
              kim_sample_path=_kim_sample(k), radiation_source=src)
    assert src == ["kim_l010"], src
    # 샘플은 전 시간 (200+100)*0.0036 = 1.08 MJ로 고정했다 — 추정값이면 야간이 0이 된다
    assert all(x["solar_rad"] == pytest.approx(1.08, abs=1e-3) for x in w)


@needs_rad
def test_KIM이_24시간을_못_채우면_추정으로_폴백하고_사유를_남긴다(tmp_path):
    """일부만 예보값이고 일부는 추정값인 상태를 만들지 않는다. 다만 조용히 넘기지도 않는다."""
    from app.services.kma_forecast import fetch
    k = tmp_path / "k"; k.mkdir()
    src: list[str] = []
    w = fetch(date(2026, 3, 15), sample_path=_sample(tmp_path),
              kim_sample_path=_kim_sample(k, hours=range(1, 18)), radiation_source=src)
    assert src[0].startswith("estimator") and "24시간 중 17개" in src[0], src
    night = [x["solar_rad"] for x in w if x["hour"] in (1, 2, 3, 24)]
    assert all(v == 0 for v in night), f"폴백 경로가 아니다: 야간 {night}"


@needs_rad
def test_샘플_재생에서는_네트워크를_타지_않는다(tmp_path):
    """sample_path만 주면 KIM 호출을 시도하지 않아야 한다 — 오프라인 테스트가 깨지지 않도록."""
    from app.services.kma_forecast import fetch
    src: list[str] = []
    fetch(date(2026, 3, 15), sample_path=_sample(tmp_path), radiation_source=src)
    assert src == ["estimator"], src      # 폴백 사유가 붙지 않는다 = 시도 자체를 안 했다
