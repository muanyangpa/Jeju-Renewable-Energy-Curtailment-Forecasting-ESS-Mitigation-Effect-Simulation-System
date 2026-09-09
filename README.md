# 제주 출력제어 예측 프로젝트 — Backend (Spring Boot)

담당: 이성헌 | 스택: Java 21, Spring Boot 3.3, PostgreSQL
통신규격 확정서 v1.0 (2026.09.08, 지호 작성) 반영 완료.

## 폴더 구조

```
domain/       Region, EnergySource(SOLAR/WIND), PowerGeneration,
              CurtailmentPrediction(필드명 정정: curtailmentMwh),
              EssSimulationResult
repository/   각 Entity의 JPA Repository
controller/   RegionController, CurtailmentPredictionController(/daily, /hourly),
              EssSimulationController(/api/v1/ess/simulate), HealthController
service/      AiClientService(배치/단건/헬스체크 호출), EssSimulationService
dto/          PredictionRequest/Response, WeatherHour, HourlyPrediction,
              HourlyPredictionRequest/Response, ErrorResponse
config/       SecurityConfig, WebClientConfig, GlobalExceptionHandler(AI 서버 에러 전달)
```

## 확정된 통신규격 반영 내역

- `excessGenerationMwh` → **`curtailmentMwh`**로 필드명 정정 (개념이 "초과발전량"이
  아니라 "출력제어량"이었다는 지적 반영, Entity 포함)
- `forecastGenerationMwh`, sin/cos 변환 전부 **AI 서버가 계산** — Backend는
  기상 원본(`WeatherHour`)과 조회 조건만 전달
- 배치 엔드포인트 `/api/v1/predictions/daily`가 주 경로 (대시보드 기본 화면),
  단건 `/api/v1/predictions/hourly`는 재조회·디버깅용으로 병행
- `curtailmentMwhAvailable` 플래그 추가 — SOLAR는 항상 false, Frontend가 이 값
  보고 ESS 시뮬레이션 화면을 비활성 처리해야 함
- `modelVersion` 필드 추가 — 발표 시 재현성 근거로 사용
- `/api/v1/ess/simulate` — ESS 슬라이더 what-if 계산 (07장 min() 공식 그대로,
  curtailmentMwh 정의를 CurtailmentPrediction과 동일하게 맞춤)
- AI 서버 에러(400/503)를 `GlobalExceptionHandler`가 그대로 전달(pass-through)

## 아직 지호님과 확인이 필요한 것

- `weather` 배열 24개 검증(개수·중복 등, `INVALID_HOUR_SET`)을 AI 서버가
  전담하는지, Backend가 미리 걸러줘야 하는지 — 현재는 AI 서버 전담으로 가정하고
  Backend는 그대로 전달만 함
- `curtailmentMwhAvailable=false`일 때 Frontend UX(화면 숨김 vs 비활성 표시)
- ESS 슬라이더(`essCapacityMw`) 범위 — Frontend 작업 시 결정 필요

## 실행 방법

1. PostgreSQL에 DB 생성
   ```sql
   CREATE DATABASE curtailment_db;
   ```
2. `application.yml`에서 비밀번호 설정
3. IntelliJ에서 실행 (JDK 21 필요)
4. 지호님이 AI 서버 스텁(고정값 반환)을 올려주면, `ai-server.base-url`
   (기본 `http://localhost:8000`)에 맞춰 연동 테스트 가능

## 지금 상태

- [x] Entity/Repository/Controller/DTO — 확정 스펙(v1.0) 전체 반영
- [x] ESS 흡수율 계산 로직(07장 공식)
- [x] AI 서버 에러 pass-through 처리
- [ ] AI 서버 스텁 연동 테스트 (지호님 스텁 배포 대기)
- [ ] 로컬 실행 테스트 (PostgreSQL 연결 후 API 테스트)
