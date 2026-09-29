"""서빙 모델 선택의 단일 출처 — 학습과 추론이 같은 표를 본다.

[왜 별도 파일인가]
/predict는 (발전원, 수요 유무, 태양광 기상값 유무)로 아티팩트 이름을 조립한다. 여기에
'모델군'(RandomForest / 로지스틱 회귀)과 '경로별 피처 제외'가 더해지면서, 학습 쪽과 서빙 쪽이
같은 규칙을 두 번 적어야 하는 상태가 됐다. 한쪽만 고치면 조용히 어긋나므로 표를 한 곳에 둔다.
serving이 training을 import하지 않도록 이 파일은 sklearn에 의존하지 않는다.

[모델군을 경로별로 고른 근거 — 2026-09-27]
지금까지 모든 분류기가 RandomForest 기본 설정이었고, 더 단순한 모델과 비교한 적이 없었다.
같은 분할·같은 sigmoid 보정으로 로지스틱 회귀를 붙여 일(day) 블록 짝지은 부트스트랩으로
비교한 결과가 models/model_family_comparison.csv다. 이 문제는 본질적으로 단조이기 때문
(순열 중요도에서 침투율 계열 하나가 지배)에 선형 모델이 유리하다. 특히 태양광은 학습 양성이
101건뿐이라 RandomForest의 용량이 과하다.
"""
from __future__ import annotations

# 확률 보정 방식. sigmoid는 단조 변환이라 순위 지표를 보존한다(README '분류기 — 확률 보정').
SERVED_METHOD = "sigmoid"

# 경로 키 = 아티팩트 이름의 중간 부분. (발전원, 수요 유무, 교차피처 여부)로 만든다.
PATH_KEYS = ("solar", "solar_demand", "wind", "wind_demand", "wind_demand_crossp")

# 경로별로 서빙하는 모델군. train_classifier가 두 군을 모두 학습·비교해 저장하고,
# 여기 적힌 쪽만 /predict가 로드한다. 근거는 model_family_comparison.csv.
# 판정은 일 블록 짝지은 부트스트랩 95% CI가 0을 배제하는지로 한다. '차이 없음'이면 바꾸지
# 않는다(RandomForest가 기존 서빙이므로 기본값).
#
# 실측 발전량 입력 기준 (models/model_family_comparison.csv):
#   경로                  ΔAUC(RF−LR)            ΔPR-AUC(RF−LR)         판정
#   solar                 [-0.0013, +0.0090]     [-0.0224, +0.2129]     차이 없음
#   solar_demand          [-0.0027, +0.0045]     [-0.1397, -0.0059]     LR 우세
#   wind                  [+0.0181, +0.0404]     [+0.0550, +0.1493]     RF 우세
#   wind_demand           [-0.0102, +0.0001]     [-0.0241, +0.0578]     차이 없음
#   wind_demand_crossp    [-0.0091, -0.0018]     [-0.0346, -0.0009]     LR 우세
#
# 배포는 **서비스 경로**(컨버터 예측 입력) 기준으로 정한다 — 실제로 그 입력이 들어오기 때문이다.
#   wind_demand_crossp : ΔAUC [-0.0157,-0.0059] LR 우세 / ΔPR-AUC [-0.0367,-0.0095] LR 우세
#                        / Δtop5 [-0.0185,+0.0196] 차이 없음   -> **LR 채택**
#   solar_demand       : ΔPR-AUC [-0.1111,-0.0075] LR 우세인데 Δtop5 [+0.0000,+0.1294]로
#                        **RF가 나을 가능성**이 있다(하한이 0에 접함). top5_capture는 제품이
#                        실제로 쓰는 지표다 — 대시보드 '높음' 밴드가 상위 5%로 정의돼 있다.
#                        지표가 엇갈릴 때 제품이 쓰는 쪽을 따라 **RF 유지**.
#
# 패턴이 설명된다. **침투율 계열 피처가 있는 경로에서 선형이 유리하고, 없는 경로에서 트리가
# 유리하다.** 침투율이 들어오면 '침투율이 높으면 제어가 난다'는 단조 관계가 지배하므로(순열
# 중요도 +0.49~+0.55) 선형 모델이 그 구조에 맞고 표본 대비 분산도 작다. 반대로 수요가 없는
# 풍력 경로는 이용률 단독으로 단조가 아니어서(풍력 이용률 단독 AUC 0.476 — 무작위보다 나쁘다)
# 트리의 분할이 필요하다. 모델군을 경로별로 두는 것이 우연이 아니라는 뜻이다.
SERVED_FAMILY: dict[str, str] = {
    "solar": "rf",
    "solar_demand": "rf",   # PR-AUC는 LR 우세지만 top5(제품 지표)는 RF — 위 주석 참고
    "wind": "rf",
    "wind_demand": "rf",
    "wind_demand_crossp": "lr",
}

# 경로별로 제외하는 피처.
#
# wind_demand_crossp에서 capacity_factor·penetration을 빼는 이유: 순열 중요도가 음수였다
# (쓸모없거나 해롭다). total_penetration = (풍력+태양광)/수요가 penetration = 풍력/수요를 이미
# 포함하고, 풍력 자신의 이용률은 제어를 거의 설명하지 못한다(풍력 이용률 단독 AUC 0.476 —
# 무작위보다 나쁘다). 다른 경로에서는 빼면 발전량 정보가 아예 사라지므로 빼지 않는다.
#
# ⚠ 두 피처의 근거 강도가 다르다. 사전 구간(보정 2022 하반기)과 사후 구간(2023 테스트)에서
# 각각 재보니:
#     capacity_factor   사전 -0.0218 / 사후 -0.0327   -> 양쪽에서 음수, 사전 증거 있음
#     penetration       사전 +0.0192 / 사후 -0.0299   -> **사후에만 음수**
# 즉 penetration 제외는 2023 테스트셋 정보로 고른 것이다. 이 저장소는 임계값 선정에서
# 그것을 피하려고 중첩 분할까지 만들었는데, 피처 선택에는 같은 규율을 적용하지 않았다.
# 되돌리지 않고 **표기**하기로 했다 — 제외 후 성능이 실제로 더 좋고(PR-AUC 0.7902 -> 0.8145),
# 되돌린다고 해서 2023을 이미 본 사실이 사라지지 않는다. 다만 이 한 건은 "사전 증거 없음"으로
# 읽어야 하고, 독립 연도(2024 이후 라벨)가 생기면 가장 먼저 재확인할 항목이다.
FEATURE_DROP: dict[str, tuple[str, ...]] = {
    "wind_demand_crossp": ("capacity_factor", "penetration"),
}


# 경로별·모델군별 하이퍼파라미터 오버라이드: HYPERPARAMS[경로][모델군]. 없으면 기본값.
# 모델군별로 나눠야 한다 — RandomForest의 max_depth를 로지스틱 회귀에 주면 터진다.
#
# [2026-09-27] 지금까지 주 분류기의 하이퍼파라미터는 탐색 기록 없는 고정값이었다.
# validate_blockcv.py가 2023을 건드리지 않고(2023 이전 rolling-origin 폴드만) 격자를 탐색하고,
# 후보를 2023 홀드아웃으로 '확인'한다(선택이 아니라 확인이다).
#
#   경로                  2023 이전 최적              사전 이득    2023 확인
#   wind                  max_depth 6/n_est 400      +0.0039     -> 이득이 무의미해 현행 유지
#   wind_demand           max_depth 4                +0.0214     -> 세 지표 모두 유의하게 개선, 채택
#   wind_demand_crossp    C=0.01                     +0.0360     -> 2023에서 차이 없음, 현행 유지
#   solar / solar_demand  탐색 불가                   —           -> 아래 주석 참고
#
# 세 경로가 모두 '더 강한 정규화'를 선호한 것은 표본이 작다는 진단과 일치한다. 다만 2023 이전
# 폴드가 2개뿐이라 근거가 약하고, 실제로 2023 확인에서 셋 중 둘이 기각됐다 — 사전 이득만 보고
# 갈아타면 안 된다는 사례다.
#
# ⚠ 태양광은 하이퍼파라미터를 탐색할 수 없다. 라벨이 2021-10에 시작하고 블록별 양성이 3~20건뿐
# 이어서 2023 이전에 성립하는 rolling-origin 폴드가 0개다. 현행값은 **탐색되지 않은 기본값**이며,
# 테스트 구간을 건드리지 않고 고를 방법이 현재 데이터로는 없다. 라벨을 2024-04까지 늘리면
# (README '무슨 일이 있었나') 폴드를 하나 더 만들 수 있다.
# ⚠ 모델군 비교(SERVED_FAMILY)는 각 군의 '기본값' 하이퍼파라미터로 수행했다. 여기에 오버라이드를
# 넣으면 그 경로의 비교가 한쪽만 튜닝된 상태가 되므로, 재학습 시 model_family_comparison.csv를
# 다시 읽고 판정이 바뀌지 않았는지 확인할 것. wind_demand는 rf를 튜닝해 rf가 더 유리해졌고
# (이미 rf 서빙이라 결정은 그대로), lr은 탐색하지 않았다.
HYPERPARAMS: dict[str, dict[str, dict]] = {
    "wind_demand": {"rf": {"max_depth": 4}},
}


def hyperparams(key: str, family: str) -> dict:
    return HYPERPARAMS.get(key, {}).get(family, {})


def path_key(energy_type: str, use_demand: bool, use_cross: bool) -> str:
    return energy_type + ("_demand" if use_demand else "") + ("_crossp" if use_cross else "")


def artifact_name(energy_type: str, use_demand: bool, use_cross: bool,
                  family: str | None = None, calibrated: bool = True) -> str:
    """서빙 아티팩트 이름. family를 주면 그 모델군, 생략하면 SERVED_FAMILY를 따른다.

    RandomForest는 접미사가 없다 — 기존 이름을 그대로 유지해 하위 호환을 지킨다.
    """
    key = path_key(energy_type, use_demand, use_cross)
    fam = family or SERVED_FAMILY.get(key, "rf")
    name = f"classifier_{key}" + ("" if fam == "rf" else f"_{fam}")
    return name + (f"_calibrated_{SERVED_METHOD}" if calibrated else "")


def features_for(key: str, features: list[str]) -> list[str]:
    drop = FEATURE_DROP.get(key, ())
    return [f for f in features if f not in drop]


# ---------------------------------------------------------------------------
# 밴드 바닥값 — '오늘은 위험 시간 없음'을 표시할 수 있게 하는 상수
# ---------------------------------------------------------------------------
# 대시보드는 하루 24시간 안에서 순위를 매겨 등급을 만든다. 순위는 정의상 척도를 지우므로
# 제어가 전혀 없는 날에도 1등이 존재하고 그것이 '매우 높음'으로 표시된다 — 2023년 테스트에서
# 무제어일 248일 전부에 경보가 떴다. 바닥값은 그 아래면 아무것도 표시하지 않게 해 이를 막는다.
#
# 선정 기준: 조용해진 날 중 실제 제어일이 2%를 넘지 않는 가장 높은 값.
# 제어일을 놓치는 비용(흡수 못 한 MWh)이 헛걸음 비용보다 크므로 재현율 쪽으로 기울였다.
# 선정 구간은 학습+보정이고 테스트 구간은 보지 않는다. 근거: app/training/select_band_floor.py
# 산출물: models/band_floors.csv
#
# 경로마다 값이 다른 이유는 확률 척도가 경로마다 다르기 때문이다(같은 임계값에서 발화율 9배 차이).
BAND_FLOOR: dict[str, float] = {
    "solar": 0.070,
    "solar_demand": 0.060,
    "wind": 0.035,                 # 신뢰 불가 — 아래 BAND_FLOOR_RELIABLE 참고
    "wind_demand": 0.095,
    "wind_demand_crossp": 0.320,
}

# wind 단독 경로는 PR-AUC 0.391로 애초에 '제어일 판별'을 못 한다. 놓침 2% 제약을 지키는
# 바닥값이 0.035에 머물러 조용한 날이 730일 중 2일뿐이다(주경로는 318일). 억지로 올리면
# 제어일을 통째로 놓친다(0.10에서 9.0%, 0.30에서 16.2%). 그래서 이 경로에서는 바닥값을
# 적용하지 않고 기존 방식을 유지하며, 화면에 한계를 표시한다.
BAND_FLOOR_RELIABLE: dict[str, bool] = {
    "solar": True, "solar_demand": True, "wind": False,
    "wind_demand": True, "wind_demand_crossp": True,
}


def band_floor(key: str) -> tuple[float, bool]:
    """(바닥값, 신뢰 가능 여부). 신뢰 불가면 호출부가 바닥값을 적용하지 말아야 한다."""
    return BAND_FLOOR.get(key, 0.0), BAND_FLOOR_RELIABLE.get(key, False)
