# AMI 이상탐지 프로젝트 — Frontend (Streamlit)

1주차 목표(뼈대: 화면 전환 + API 연결 + 목록/상세 기본 구조) 기준으로 만든 뼈대입니다.

## 폴더 구조

```
ami-frontend/
├── app.py                    # 화면 1: 이상 의심 세대 목록(큐) — 진입점
├── pages/
│   └── 1_세대_상세.py         # 화면 2: 세대 상세 (1순위 — SHAP 패널, 체크리스트 포함)
├── api_client.py              # Backend(Spring Boot) 호출 전담 모듈
├── models.py                  # 위반유형 표시(라벨/색상) 공통 데이터 모델
└── requirements.txt
```

## 실행 방법

1. Backend(Spring Boot)가 먼저 `localhost:8080`에서 실행 중이어야 합니다.

2. 가상환경 생성 후 의존성 설치
   ```
   cd ami-frontend
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

3. 실행
   ```
   streamlit run app.py
   ```
   브라우저가 자동으로 `localhost:8501`을 엽니다.

4. Backend에 테스트 데이터가 없으면 목록이 비어 있습니다. Backend README의
   curl/PowerShell 명령으로 세대 1건 이상 등록 후 새로고침하세요.

## 지금 상태 (1주차 완료 기준)

- [x] 화면 전환 (목록 → 상세, `st.switch_page` 사용)
- [x] Spring Boot API 연결 (`api_client.py`)
- [x] 공통 데이터 모델 (위반유형 라벨/색상, `models.py`)
- [x] 목록 기본 테이블 (필터·정렬 포함)
- [x] 상세 페이지 기본 구조 (시계열 그래프, SHAP 패널, 체크리스트)

## 다음 단계에서 확정 필요 (AI 담당자와 협의)

`pages/1_세대_상세.py` 안에 TODO로 표시해둔 두 곳:

1. **이상 구간 하이라이트**: 지금은 예시 마커만 표시. AI 응답에 실제
   이상 시점(날짜 인덱스) 정보가 포함되면, 그 구간을 그래프에 정확히 표시하도록 교체.
2. **SHAP 설명(explanation) 포맷**: 지금은 문자열을 그대로 표시. AI 서버가
   JSON 구조로 줄지, 완성된 문장으로 줄지에 따라 파싱 로직이 달라짐.

이 두 가지는 Backend-AI 통신규격(IF-02)이 최종 확정되면 함께 정리하면 됩니다.

## 2주차 이후 계획 (참고)

- 2주차: 목록+상세 완성 (MVP)
- 3주차: 통계 요약 화면, Folium 지도 시각화 추가
- 4주차: 디자인 통합, 실제 AI 결과 연결
- 5주차: 배포, 발표 리허설
