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
# wind_demand_crossp에서 capacity_factor·penetration을 빼는 이유: 순열 중요도가 각각
# -0.0035, -0.0051로 **음수**다(쓸모없거나 해롭다). total_penetration = (풍력+태양광)/수요가
# penetration = 풍력/수요를 이미 포함하고, 풍력 자신의 이용률은 제어를 거의 설명하지 못한다
# (풍력 이용률 단독 AUC 0.476 — 무작위보다 나쁘다). 다른 경로에서는 빼면 발전량 정보가
# 아예 사라지므로 빼지 않는다.
FEATURE_DROP: dict[str, tuple[str, ...]] = {
    "wind_demand_crossp": ("capacity_factor", "penetration"),
}


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
