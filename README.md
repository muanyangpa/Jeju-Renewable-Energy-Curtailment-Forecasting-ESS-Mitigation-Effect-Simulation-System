# 출력제어 예측 AI 서버

계획서(출력제어예측_프로젝트계획서.docx) 02장·04장·06장·07장·09장에서 정의한 AI 서버 구현체.
Backend(Java/Spring Boot)가 REST로 호출하는 내부 서비스다.

## 실행

```bash
pip install -r requirements.txt

# 1) 모델 학습 (최초 1회, 또는 데이터 갱신 시)
python -m app.training.train_converter
python -m app.training.train_classifier
python -m app.training.train_curtailment_regressor   # 풍력 전용

# 2) 서버 기동
uvicorn app.main:app --reload --port 8001
```

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
  - Backend는 이 에러코드를 그대로 릴레이하면 됨 (별도 매핑 불필요)

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
  "note": "태양광은 ... expected_curtailment_mwh를 제공하지 않습니다 ..."  // 태양광 응답에만 존재
}
```

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

## 모델 성능 (이번 학습 실행 기준, models/*_metrics.csv 참고)

이 수치는 계획서 05장·07장에 정리된 **연구 검증 결과**(다중 지역·다중 방법론 교차검증)와는
학습/평가 조건이 다르다 — 단일 2023년 전체 테스트 구간으로 단순화한 **1차 프로덕션 베이스라인**이다.
풍력 컨버터는 I장 검증 결과를 반영해 3지점(제주184·고산185·성산188) 평균 관측치로 재학습했고,
태양광 컨버터는 일사량 기준 단일 지점(H장 검증)을 그대로 사용한다. 배포 전 재검토가 필요하다.

| 모델 | 지표 | 값 |
|---|---|---|
| 컨버터(태양광) | corr / NMAE | 0.967 / 29.6% |
| 컨버터(풍력, 3지점 평균) | corr / NMAE | 0.807 / 43.4% (단일지점 대비 0.616→0.807, 63.7%→43.4% 개선) |
| 분류기(태양광) | AUC / top5 | 0.977 / 0.255 |
| 분류기(풍력, 수요 미포함) | AUC / top5 | 0.961 / 0.236 |
| 분류기(풍력, 수요 포함·정식) | AUC / top5 | 0.970 / 0.233 |
| 제어량 회귀(풍력) | 2023 총합 오차 | 실측 26,197MWh → 예측 16,237MWh (-38%) |

**알려진 한계 (정직하게 기록)**
- 풍력 컨버터를 3지점(제주184·고산185·성산188) 평균 관측치로 재학습해 NMAE를 63.7%→43.4%로
  개선했다(corr 0.616→0.807) — I장이 검증한 "위치오차 상쇄" 효과가 프로덕션 코드에서도 방향성 있게
  재현됨. 다만 I장의 최종 수치(NMAE 12~15.6%)에는 아직 못 미친다 — I장은 예보 아카이브의
  리드타임별 앙상블까지 반영한 결과이고, 여기서는 ASOS 실측 관측치의 단순 회귀만 사용하기 때문.
  추가 개선 여지는 있으나 기본 방향은 검증됨.
- 풍력 수요피처: 2023년 단일 분할 테스트에서는 포함 모델의 top5(0.233)가 미포함(0.236)보다
  근소하게 낮게 나와 J장의 "정식 채택" 결론과 모순되는 것처럼 보였다. 이를 확인하기 위해
  `app/training/validate_demand_blockcv.py`로 J장과 동일한 leave-block-out 6블록 교차검증
  (21.01~24.01, 6개월 단위)을 프로덕션 데이터로 재현한 결과 — **6개 블록 전부에서 수요피처 포함
  모델이 미포함 모델과 같거나 더 높은 top5/AUC를 기록**했고(단 한 블록도 악화 없음), 양성 건수
  가중평균 top5는 0.2407→0.2437(+1.3%p), AUC는 0.9627→0.9709로 개선됐다. 즉 단일 2023년
  분할에서 나타난 역전은 표본 노이즈였고, 블록 교차검증으로 보면 수요피처는 일관되게 도움이 된다 —
  J장의 "정식 채택" 결론이 프로덕션 코드/데이터에서도 재확인됐다. 다만 개선폭(top5 +1.3%p)은
  J장이 원래 보고한 폭(0.377→0.493, +11.6%p)보다 작다 — 원인은 미확정이나, J장 분석이 예보
  아카이브 기반 피처·다른 지역 단위 수요 데이터를 사용한 반면 여기서는 단순 실측 수요(도 전체)만
  썼기 때문일 가능성이 높다. 결론(채택 여부)은 재확인됐으므로 09장 리스크로 남기지 않되, 개선폭
  차이는 향후 피처 정교화 과제로 기록해둔다.
- 제어량 회귀모델은 총합을 38% 과소추정한다 — RandomForest가 극단적으로 큰 제어량 시간대를
  평균으로 당기는(shrinkage) 전형적인 현상. `/ess/simulate`의 결과값은 이 오차를 그대로 물려받으므로,
  대시보드에 노출할 때 "추정치이며 과소평가 경향이 있다"는 안내를 함께 표시하는 것을 권장한다.

## 폴더 구조

```
ai_server/
  app/
    data_prep.py          # 원본 CSV 로딩/전처리 공통 유틸
    schemas.py             # API 계약 (Backend와 공유하는 단일 출처)
    main.py                 # FastAPI 앱, 엔드포인트, 에러코드 매핑
    services/
      predict.py            # 컨버터+분류기+회귀 추론 파이프라인
      ess_simulation.py      # ESS 흡수율 계산 (07장 두 방식)
    training/
      train_converter.py
      train_classifier.py
      train_curtailment_regressor.py
      validate_demand_blockcv.py   # J장 방식 6블록 교차검증 (풍력 수요피처 검증용, 서비스에 미사용)
      solar_potential_curtailment.py  # 태양광 조건부 잠재발전량 모델 실험 (07장 방법C, 서비스에 미사용)
  models/                  # 학습된 .joblib 아티팩트 + *_metrics.csv (demand_blockcv_results.csv 포함)
  smoke_test.py             # FastAPI TestClient 기반 엔드투엔드 점검
```
