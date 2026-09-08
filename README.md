# 제주 출력제어 예측 프로젝트 — Backend (Spring Boot)

담당: 이성헌 | 스택: Java 21, Spring Boot 3.3, PostgreSQL
AMI 프로젝트 뼈대(ami-backend)를 재사용해 새 도메인에 맞게 재구성함.

## 폴더 구조

```
domain/       Region, EnergySource(SOLAR/WIND), PowerGeneration,
              CurtailmentPrediction, EssSimulationResult
repository/   각 Entity의 JPA Repository
controller/   RegionController, CurtailmentPredictionController,
              EssSimulationController — 외부 노출 API
service/      AiClientService(Python 호출), EssSimulationService(흡수율 계산)
dto/          PredictionRequest/Response — Python AI 서버와 주고받을 통신규격 초안
```

## 계획서 대응 관계

- **04장 01 (출력제어 확률 예측 엔진)** → `CurtailmentPredictionController`,
  `AiClientService`
- **04장 02·03 (ESS 충방전 시뮬레이션 / 용량 조정 시뮬레이터)** →
  `EssSimulationService.simulate()` — 07장의 `min(그 시각 제어량, ESS 정격출력)`
  계산을 그대로 구현. 대시보드 슬라이더가 `essCapacityMw`만 바꿔서 반복 호출하면
  별도 모델 없이 즉시 재계산됨(04장 03 요구사항).

## ⚠️ 카톡으로 논의 후 확정해야 할 것 (통신규격)

`dto/PredictionRequest.java`, `dto/PredictionResponse.java`는 초안입니다.
지호님과 논의해서 아래를 확정한 뒤 수정 필요:

1. **`forecastGenerationMwh`**: 기상청 예보 기반 발전량 예측치를 Backend가
   미리 받아서 보낼지, AI 서버가 자체적으로 계산할지
2. **`demandMwh`**: 풍력만 쓰는 필드인데, 태양광 요청 시 이 필드를 아예
   생략할지 null로 보낼지
3. **시간·월 sin/cos 변환**: 계획서 04장에 모델 입력이 "시간(sin/cos)·
   월(sin/cos)"로 되어 있는데, 이 변환을 Backend에서 미리 해서 보낼지
   AI 서버가 `targetHour`만 받아서 자체 계산할지
4. **`excessGenerationMwh`(초과발전량)**: AI 서버가 분류 확률과 함께 이 값도
   같이 내려줄지, 아니면 Backend가 발전량-수요 차이로 별도 계산할지

## 실행 방법

1. PostgreSQL에 DB 생성
   ```sql
   CREATE DATABASE curtailment_db;
   ```
2. `application.yml`에서 비밀번호 설정
3. IntelliJ에서 실행 (JDK 21 필요 — AMI 때와 동일하게 Project Structure에서 확인)

## 지금 상태

- [x] Entity/Repository/Controller 뼈대
- [x] ESS 흡수율 계산 로직(07장 공식 그대로 구현)
- [ ] 통신규격 확정 (위 4가지 항목 — 카톡 논의 후 dto 수정)
- [ ] 로컬 실행 테스트 (AMI 때처럼 PostgreSQL 연결 후 API 테스트 필요)
