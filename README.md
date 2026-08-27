# AMI 이상탐지 프로젝트 — Backend (Spring Boot)

담당: 이성헌 | 스택: Java 21, Spring Boot 3.3, PostgreSQL

## 폴더 구조

```
src/main/java/com/ami/anomaly/
├── domain/          # Entity (Meter, AnomalyResult, ViolationType)
├── repository/      # JPA Repository
├── controller/      # 외부 노출 REST API (AnomalyController, MeterController)
├── service/         # 비즈니스 로직 (AiClientService - Python 서버 호출)
├── config/          # 설정 (SecurityConfig, WebClientConfig)
└── dto/             # Python AI 서버와 주고받는 요청/응답 객체
```

## 실행 방법

1. **로컬에 PostgreSQL 설치 후 DB 생성**
   ```sql
   CREATE DATABASE ami_db;
   ```

2. `src/main/resources/application.yml`에서 DB 계정 정보 본인 환경에 맞게 수정

3. IntelliJ / VS Code + Java Extension Pack으로 프로젝트 열기
   (또는 터미널에서 `mvn spring-boot:run`, Maven 설치 필요)

4. 실행 후 `http://localhost:8080/api/meters` 로 확인

## 지금 당장 할 수 있는 것 (AI 담당자 확정 전)

- [x] 프로젝트 뼈대 (Entity, Repository, Controller, Security)
- [ ] Meter, AnomalyResult 저장/조회 API 테스트
- [ ] Postman 등으로 API 동작 확인
- [ ] Swagger(springdoc-openapi) 추가해서 API 문서 자동화 (선택)

## IoT 파트 — AMI 데이터 수신 & 시뮬레이터

실제 스마트 계량기가 없으므로, `simulator/simulate_ami.py`가 계량기 역할을 대신해서
Backend로 데이터를 흘려보낸다.

### 흐름
```
[시뮬레이터(계량기 역할)] --POST--> [MeterReadingController] --저장--> [DB]
                                                              --조회(최근 96개)--> [AI 판정 요청 시 사용]
```

### 사용 순서
1. Backend 실행 후, `POST /api/meters`로 테스트 세대 하나 먼저 등록
   ```json
   { "consumerNo": "1001", "contractPowerKw": 5.0, "idleRequested": false, "reservePowerContract": false }
   ```
2. 시뮬레이터 의존성 설치
   ```
   cd simulator
   pip install -r requirements.txt
   ```
3. 가상 데이터 전송 시작 (2초 간격, 5% 확률로 이상치 주입)
   ```
   python simulate_ami.py --mode synthetic --consumer 1001 --interval 2
   ```
4. 저장된 데이터 확인
   ```
   GET http://localhost:8080/api/meters/1001/readings/recent
   ```

### 실제 데이터셋(SGCC 등)으로 돌리고 싶을 때
`--mode csv --csv 파일경로`로 실행. 단, SGCC 원본은 날짜가 컬럼으로 나열된 형태라
`recorded_at,usage_kwh` 두 컬럼 포맷으로 변환하는 전처리가 먼저 필요함 (AI 담당자 합류 후 협의).



`dto/AiPredictRequest.java`, `dto/AiPredictResponse.java`가 Spring Boot ↔ Python 간
통신 규격 **초안**입니다. 실제 필드명/형식은 AI 담당자와 반드시 합의 후 수정하세요.

- `loadSeries`: 시계열 데이터 포맷 (몇 포인트? 15분 단위 96개?)
- `violationType`: AI가 내려주는 문자열이 `ViolationType` enum 값과 정확히 일치해야 함
- `explanation`: SHAP 결과를 어떤 형식(JSON? 텍스트?)으로 줄지

`service/AiClientService.java`의 `/predict` 경로도 Python 쪽 FastAPI 라우트와
맞춰서 수정 필요.

## 보안 설정 관련

`SecurityConfig`는 현재 `permitAll()`로 전부 열어둔 데모 상태입니다.
팀/발표 일정에 여유가 있으면 JWT 기반 인증 정도는 추가해서
"보안에 강한 Java"라는 제안 취지를 실제로 보여주는 게 좋습니다.
