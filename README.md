# 출력제어 예측 AI 서버

계획서(출력제어예측_프로젝트계획서.docx) 02장·04장·06장·07장·09장에서 정의한 AI 서버 구현체.
Backend(Java/Spring Boot)가 REST로 호출하는 내부 서비스다.

## 실행

```bash
pip install -r requirements.txt

# 0) 원본 공공데이터 CSV 위치 지정 (기본값: ai_server/data/raw)
export DATA_DIR=/path/to/원본CSV폴더

# 1) 모델 학습 (최초 1회, 또는 데이터 갱신 시)
python -m app.training.train_converter
python -m app.training.train_classifier
python -m app.training.train_curtailment_regressor   # 풍력 전용

# 2) 검증 (서비스와 무관, 결과는 models/*.csv)
python -m app.training.evaluate_pipeline             # 날씨→컨버터→분류기 서비스 경로 성능
python -m app.training.validate_demand_blockcv       # 풍력 수요피처 6블록 교차검증

# 3) 점검 후 서버 기동
python smoke_test.py
uvicorn app.main:app --reload --port 8000
```

## 2026-09-23 수정 이력 (재학습 필수)

| 구분 | 수정 전 | 수정 후 |
|---|---|---|
| 학습 데이터 구성 | 출력제어 이력과 inner join → **제어 발생일만** 학습·평가 (2023 테스트: 태양광 64일, 풍력 117일) | 발전량 실적 전체 달력에 left join, 이력에 없는 시간 = 제어 없음 (`data_prep.build_labeled_hourly`) |
| 상위5%포착률 | 양성 비율이 20%에 가까워 이론상 최대값(약 0.25)에 막힘 → 수요피처 비교 불가 | `top5_ceiling`·`top5_precision`·`pr_auc`·`brier` 함께 보고, 상한 도달 시 경고 (`app/metrics.py`) |
| 24시 정렬 | D일 24시가 D일 00:00에 저장(하루 어긋남) → ASOS와 1시간/일 오정렬 | D+1일 00:00으로 저장 |
| 누락 기상값 | `None`을 조용히 0으로 대체 (풍속 null → '제어 없음' 오예측) | 422 `MISSING_REQUIRED_FIELD` |
| 풍력 + 수요예측 생략 | 제어량 회귀모델이 수요 피처를 찾지 못해 **500 에러** | 수요 미포함 분류기로 전환, 제어량은 null, `model_used`·`note`로 안내 |
| 월 피처(서빙) | 24시간 모두 `target_date.month` | 각 시간의 실제 시각 기준(학습과 동일) |
| 서비스 경로 성능 | 측정 안 함 | `evaluate_pipeline.py` 신규 |
| 재현성 | 데이터 경로 하드코딩, 버전 미고정, 파일 순서 의존 | `DATA_DIR` 환경변수, scikit-learn 고정, 파일명 정렬, 아티팩트에 메타데이터(학습 구간·버전) 저장 |

> ✅ **2026-09-23 재학습 완료.** 아래 '모델 성능'·'알려진 한계'의 수치는 모두 수정 후 코드로
> 재학습·재검증한 값이다(`models/*_metrics.csv`, `pipeline_eval_metrics.csv`,
> `demand_blockcv_summary.csv` 기준). 학습 환경: Python 3.12 / scikit-learn 1.7.2.
> 2023년 테스트 구간 n=8,760(전체 달력)으로 확인돼 left-join 수정이 정상 반영됐다.

## API 계약 (Backend 팀 공유용)

스키마 정의의 단일 출처는 `app/schemas.py`다 — 아래는 그 요약.

### POST /predict

**요청**
```json
{
  "energy_type": "solar" | "wind",
  "region": "남원읍",
  "target_date": "2026-09-22",
  "weather": [ {"hour": 1, "solar_rad": 0.0, "temp": 18.2, "cloud": 4.0, "wind_speed": 5.1}, ... 24개 ... ],
  "demand_forecast_mw": [600.1, ...] // 24개, 풍력 전용(선택)
}
```

**검증 책임 분담 (09장 리스크 대응 그대로)**
- Backend: `weather` 필드 존재 여부·JSON 타입만 확인
- AI 서버(여기): `weather`가 정확히 24개인지, 1~24시가 중복·누락 없이 있는지 최종 검증
  - 위반 시 `422 {"error_code": "INVALID_HOUR_SET", "message": "..."}`
- AI 서버(여기): 발전원별 필수 기상값이 24시간 모두 있는지 검증
  - 태양광: `solar_rad`, `temp`, `cloud` / 풍력: `wind_speed`
  - 위반 시 `422 {"error_code": "MISSING_REQUIRED_FIELD", "message": "..."}`
- Backend는 이 에러코드들을 그대로 릴레이하면 됨 (별도 매핑 불필요)

**응답**
```json
{
  "energy_type": "solar",
  "region": "남원읍",
  "target_date": "2026-09-22",
  "hourly": [
    {"hour": 1, "generation_forecast_mwh": 0.0, "curtailment_probability": 0.02, "expected_curtailment_mwh": null},
    ...
  ],
  "model_used": "classifier_solar",  // 실제 사용된 분류모델
  "note": "태양광은 ... expected_curtailment_mwh를 제공하지 않습니다 ..."  // 태양광, 또는 풍력에서 수요예측 생략 시
}
```

풍력에서 `demand_forecast_mw`를 생략하면 수요 미포함 모델(`classifier_wind`)로 예측하고
`expected_curtailment_mwh`는 null이다(제어량 회귀모델이 수요 피처를 필요로 함).

**중요**: `expected_curtailment_mwh`는 **풍력만** 값이 채워진다. 태양광은 항상 `null`이다 —
전력거래소가 태양광 출력제어량(MWh)을 공식적으로 산정하지 않기 때문(계획서 07장, data.go.kr 메타데이터로
확인됨). 프론트엔드/대시보드에서 태양광에 대해 이 필드를 표시할 때는 반드시 `note` 문구를 함께 노출할 것.

### POST /ess/simulate

04장 02번(기본 ESS 시뮬레이션)과 03번(용량 조정 슬라이더)이 공유하는 엔드포인트.
슬라이더는 `rated_power_mw`만 바꿔가며 이 엔드포인트를 반복 호출하면 된다.

```json
// 요청
{"hourly_curtailment_mwh": [10, 30, 5, 0, 40], "rated_power_mw": 22.5, "method": "hourly_capped"}
// 응답
{"rated_power_mw": 22.5, "method": "hourly_capped", "total_curtailment_mwh": 85.0,
 "total_absorbed_mwh": 60.0, "absorption_rate": 0.7059}
```

`method`는 07장의 두 계산 방식과 1:1 대응:
- `hourly_capped` = ②(정식 채택) = `sum(min(hourly, rated_power_mw))`
- `naive_upper_bound` = ①(단순 상한) = `rated_power_mw * curtailed_hours`

## 모델 성능 (2026-09-23 재학습 기준)

이 수치는 계획서 05장·07장에 정리된 **연구 검증 결과**(다중 지역·다중 방법론 교차검증)와는
학습/평가 조건이 다르다 — 단일 2023년 전체 테스트 구간으로 단순화한 **1차 프로덕션 베이스라인**이다.
풍력 컨버터는 I장 검증 결과를 반영해 3지점(제주184·고산185·성산188) 평균 관측치로 재학습했고,
태양광 컨버터는 일사량 기준 단일 지점(H장 검증)을 그대로 사용한다. 배포 전 재검토가 필요하다.

### 컨버터 (날씨 → 발전량)

| 모델 | corr | NMAE | n_test |
|---|---|---|---|
| 컨버터(태양광, 184) | 0.967 | 29.6% | 8,743 |
| 컨버터(풍력, 184+185+188 평균) | 0.822 | 42.0% | 8,760 |

풍력 단일지점(184) 베이스라인은 같은 조건에서 corr 0.625 / NMAE 63.1% — 3지점 평균이
corr +0.20, NMAE −21.1%p로 I장의 "위치오차 상쇄" 효과를 재현한다.

### 분류기 (출력제어 발생 여부) — 2023년 테스트, 실측 발전량 입력

| 모델 | AUC | PR-AUC | Brier | top5 포착률 (상한) | prec@5% | n_test / 양성 |
|---|---|---|---|---|---|---|
| 분류기(태양광) | 0.979 | 0.491 | 0.034 | 0.688 (1.000) | 0.443 | 8,760 / 282 (3.2%) |
| 분류기(풍력, 수요 미포함) | 0.947 | 0.498 | 0.092 | 0.425 (0.778) | 0.546 | 8,760 / 563 (6.4%) |
| 분류기(풍력, 수요 포함·정식) | 0.974 | 0.731 | 0.034 | 0.561 (0.778) | 0.722 | 8,760 / 563 (6.4%) |

양성 비율이 3~6%로 정상화되면서 top5 포착률이 더 이상 상한(수정 전 약 0.25)에 막히지 않는다 —
태양광은 상한 1.000 대비 0.688, 풍력은 상한 0.778 대비 0.561이다.

### 서비스 경로 평가 (`evaluate_pipeline.py`) — 실측 발전량 대신 컨버터 예측값 입력

| 모델 | PR-AUC 실측 → 컨버터 | top5 실측 → 컨버터 | AUC 실측 → 컨버터 |
|---|---|---|---|
| 분류기(태양광) | 0.491 → 0.739 (+50.4%) | 0.688 → 0.844 (+22.7%) | 0.979 → 0.987 |
| 분류기(풍력, 수요 미포함) | 0.498 → 0.484 (−2.8%) | 0.425 → 0.423 (−0.4%) | 0.947 → 0.945 |
| 분류기(풍력, 수요 포함·정식) | 0.731 → 0.766 (+4.7%) | 0.561 → 0.602 (+7.3%) | 0.974 → 0.975 |

즉 **실제 서빙 경로(날씨→컨버터→분류기)의 성능이 실측 발전량을 넣은 경우보다 나쁘지 않다.**
컨버터 예측값이 실측보다 평활(smooth)해서 분류기 입력으로는 오히려 노이즈가 적기 때문으로 보인다.

### 풍력 수요피처 6블록 교차검증 (`validate_demand_blockcv.py`, 21.01~24.01)

| 지표(양성 가중평균) | 수요 미포함 | 수요 포함 | 변화 |
|---|---|---|---|
| top5 포착률 | 0.3767 | 0.4978 | **+12.1%p** |
| PR-AUC | 0.4295 | 0.6603 | +23.1%p |
| AUC | 0.9292 | 0.9664 | +3.7%p |
| Brier | 0.1102 | 0.0706 | 개선 |

**6개 블록 전부에서 수요피처 포함 모델이 우세**(PR-AUC 개선 6 / 악화 0). J장의 "정식 채택" 결론이
프로덕션 데이터에서 재확인됐다.

### 제어량 회귀 (풍력, 2023년)

| 지표 | 값 |
|---|---|
| 실측 총합 | 26,197 MWh |
| 예측 총합 | 4,803 MWh (**−81.7%**) |
| 제어시간(563h) 예측합 | 2,791 MWh |
| 비제어시간 오예측합 | 2,012 MWh |
| 제어시간 MAE | 42.8 MWh |

**알려진 한계 (정직하게 기록)**
- 풍력 컨버터(3지점 평균) NMAE 42.0%는 I장의 최종 수치(NMAE 12~15.6%)에 아직 못 미친다 —
  I장은 예보 아카이브의 리드타임별 앙상블까지 반영한 결과이고, 여기서는 ASOS 실측 관측치의
  단순 회귀만 사용하기 때문. 개선 방향(다지점 평균)은 검증됐으나 절대 수치는 추가 작업 필요.
- 풍력 수요피처는 단일 2023년 분할(top5 0.425→0.561)과 6블록 교차검증(0.3767→0.4978, +12.1%p)
  **양쪽 모두에서 일관되게 개선**을 보인다. 수정 전 코드에서 나타났던 "단일 분할 역전"과
  "블록CV 개선폭이 J장의 +11.6%p보다 작음(+1.3%p)"은 둘 다 상한에 막힌 지표가 만든 착시였고,
  라벨 구성을 고친 지금은 J장이 보고한 개선폭(+11.6%p)과 사실상 같은 크기(+12.1%p)가 나온다.
  09장 리스크에서 해제.
- **제어량 회귀모델의 과소추정이 −38%에서 −81.7%로 크게 악화됐다.** 원인은 명확하다 —
  라벨 수정으로 학습 데이터에 제어량 0인 시간(전체의 약 94%)이 모두 포함되면서 RandomForest가
  예측을 0 쪽으로 강하게 끌어당긴다(shrinkage). 수정 전 −38%는 "제어 발생일만" 학습한 값이라
  애초에 비교 대상이 아니었다. **`/ess/simulate`에 이 값을 그대로 쓰면 ESS 흡수량이 심하게
  과소평가된다.** 대시보드 노출 전 대응이 필요하다 — 제어 발생 시간만 대상으로 하는 2단계
  (분류 → 조건부 회귀) 구성, 또는 로그변환·분위수 회귀 검토를 권장한다. 현재 09장 최우선 리스크.
- 태양광 제어량(MWh)은 전력거래소가 공식 산정하지 않으므로 이 표에 없다(07장) — API 응답에서도
  `expected_curtailment_mwh`는 항상 null이다.

## 폴더 구조

```
ai_server/
  app/
    data_prep.py          # 원본 CSV 로딩/전처리 공통 유틸
    schemas.py             # API 계약 (Backend와 공유하는 단일 출처)
    main.py                 # FastAPI 앱, 엔드포인트, 에러코드 매핑
    metrics.py              # 분류 평가 지표 공통 (top5 + 상한·정밀도·PR-AUC·Brier)
    model_io.py             # 모델 저장/로드 + 메타데이터·버전 확인
    services/
      predict.py            # 컨버터+분류기+회귀 추론 파이프라인
      ess_simulation.py      # ESS 흡수율 계산 (07장 두 방식)
    training/
      train_converter.py
      train_classifier.py
      train_curtailment_regressor.py
      validate_demand_blockcv.py   # J장 방식 6블록 교차검증 (풍력 수요피처 검증용, 서비스에 미사용)
      evaluate_pipeline.py         # 날씨→컨버터→분류기 서비스 경로 평가 (서비스에 미사용)
      solar_potential_curtailment.py  # 태양광 조건부 잠재발전량 모델 실험 (07장 방법C, 서비스에 미사용)
  models/                  # 학습된 .joblib 아티팩트 + *_metrics.csv (demand_blockcv_results.csv 포함)
  smoke_test.py             # FastAPI TestClient 기반 엔드투엔드 점검
```
