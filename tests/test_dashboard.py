"""시연 대시보드 — 화면이 렌더링되는지, 그리고 설계 원칙을 지키는지 검증한다.

시연은 한 번뿐이라 "그때 열어보면 되겠지"가 통하지 않는다. AppTest로 스크립트를 실제로
실행해 예외가 없는지, 네 경로(풍력/태양광 × 수요 유무)가 모두 도는지 고정한다.

**절대 확률을 숫자로 노출하지 않는다**는 원칙도 테스트로 잠근다 — 경로마다 확률 척도가
다르고 신뢰도 곡선이 어긋나 있어 이것이 깨지면 화면이 사용자를 오도한다.
"""
from __future__ import annotations

import os

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1", reason="streamlit이 설치돼 있지 않습니다")
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "app.py")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(APP), "..", "models", "serving_config.py"))
    and not os.environ.get("MODELS_DIR") and not os.path.isdir(
        os.path.join(os.path.dirname(os.path.dirname(APP)), "models")),
    reason="models/ 아티팩트가 없습니다")


@pytest.fixture(scope="module")
def app():
    return st_testing.AppTest.from_file(APP, default_timeout=300).run()


def test_예외_없이_렌더링된다(app):
    assert not app.exception, [e.value for e in app.exception]


def test_세_탭이_모두_있다(app):
    """ESS 패널이 운영 화면과 분리돼 있어야 한다 — 한 화면에 섞으면 ESS 슬라이더가
    예측 신뢰도처럼 읽힌다(계통 실무 관점 검토 항목)."""
    assert len(app.tabs) == 3


def test_네_경로가_모두_돈다():
    """풍력/태양광 × 수요 유무. 태양광은 expected_curtailment_mwh가 null이라 분기가 다르다."""
    at = st_testing.AppTest.from_file(APP, default_timeout=300).run()
    for energy in ("wind", "solar"):
        for demand in (True, False):
            at.radio[0].set_value(energy)
            at.checkbox[0].set_value(demand)
            at.run()
            assert not at.exception, f"{energy}/수요={demand}: {[e.value for e in at.exception]}"
            assert at.metric[2].value, "사용 모델이 비어 있다"


def test_절대_확률을_숫자로_노출하지_않는다(app):
    """경로마다 확률 척도가 달라(같은 임계값에서 발화율 9배 차이) 확률을 그대로 보이면 오도한다.

    표에 curtailment_probability 컬럼이 들어가 있으면 이 원칙이 깨진 것이다.
    """
    for d in app.dataframe:
        cols = set(map(str, d.value.columns))
        assert "curtailment_probability" not in cols, f"확률 컬럼이 노출됐다: {cols}"


def test_등급은_세_밴드뿐이다(app):
    """상위 5% / 상위 20% / 그 외. 임계값 판정이 아니라 순위 밴드다."""
    import dashboard.app as mod
    assert set(mod.BAND_STYLE) == {"상위 5%", "상위 20%", "그 외"}
    assert mod.band_of(0, 24) == "상위 5%"      # 1위 = 24시간의 5%(1.2시간) 안
    assert mod.band_of(1, 24) == "상위 20%"     # 2위 = 20%(4.8시간) 안
    assert mod.band_of(10, 24) == "그 외"


def test_ESS_패널은_실측_제어량을_쓴다(app):
    """예측 기댓값을 넣으면 흡수율이 실측 36.8% 대비 80.6%로 과대평가된다(README 알려진 한계).

    화면에 그 경고가 떠 있어야 한다.
    """
    joined = " ".join(x.value for x in app.info)
    assert "실측" in joined and "과대평가" in joined, joined[:200]
