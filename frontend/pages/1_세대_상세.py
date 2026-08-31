"""
화면 2 — 세대 상세 뷰 (1순위: 프로젝트의 핵심을 보여주는 화면)
시계열 그래프 + 이상구간 하이라이트, SHAP 설명 패널, 판정 지원 체크리스트.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from api_client import get_anomalies, get_recent_readings
from models import label_for, color_for

st.set_page_config(page_title="세대 상세", layout="wide")

if "selected_meter_id" not in st.session_state:
    st.warning("목록에서 세대를 먼저 선택해주세요.")
    if st.button("← 목록으로 돌아가기"):
        st.switch_page("app.py")
    st.stop()

meter_id = st.session_state["selected_meter_id"]
consumer_no = st.session_state["selected_consumer_no"]

st.title(f"🔍 세대 상세 — {consumer_no}")
if st.button("← 목록으로 돌아가기"):
    st.switch_page("app.py")

st.divider()

# --- 데이터 로드 ---
anomalies, err1 = get_anomalies(meter_id)
readings, err2 = get_recent_readings(consumer_no)

if err1 or err2:
    st.error(f"⚠️ {err1 or err2}")
    st.stop()

latest = anomalies[-1] if anomalies else None

col_left, col_right = st.columns([2, 1])

# --- 좌측: 시계열 그래프 ---
with col_left:
    st.subheader("전력사용량 시계열")

    if readings:
        df = pd.DataFrame(readings)
        df["recordedAt"] = pd.to_datetime(df["recordedAt"])
        df = df.sort_values("recordedAt")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["recordedAt"], y=df["usageKwh"],
            mode="lines+markers", name="사용량(kWh)",
            line=dict(color="#3498DB"),
        ))

        # TODO: AI가 지목한 실제 이상 구간으로 하이라이트 교체 필요
        # (현재는 SHAP 연동 전이라 플레이스홀더 — explanation 파싱 로직은
        #  AI 담당자와 응답 포맷 확정 후 구현)
        if latest:
            fig.add_annotation(
                x=df["recordedAt"].iloc[-1], y=df["usageKwh"].max(),
                text="⚠️ 이상 구간 (예시)", showarrow=True, arrowcolor="red",
            )

        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("측정 데이터가 없습니다. 시뮬레이터로 데이터를 먼저 전송해주세요.")

# --- 우측: 판정 정보 + SHAP 패널 ---
with col_right:
    st.subheader("AI 판정 정보")

    if latest:
        v_type = latest["violationType"]
        st.metric("이상점수", f"{latest['anomalyScore']:.3f}")
        st.markdown(
            f"<span style='background-color:{color_for(v_type)}; "
            f"color:white; padding:4px 10px; border-radius:6px;'>"
            f"{label_for(v_type)}</span>",
            unsafe_allow_html=True,
        )

        st.markdown("**SHAP 설명 (근거)**")
        # TODO: SHAP 후처리 번역 문구는 AI 서버 응답(explanation 필드)을
        # 그대로 표시. AI 담당자와 explanation 포맷(JSON/텍스트) 확정 필요.
        st.info(latest.get("explanation") or "설명 데이터가 아직 없습니다.")
        st.caption("※ 이 설명은 SHAP 값이 보여주는 사실을 문장으로 재표현한 것으로, "
                   "판정의 추가 근거가 아닌 가독성을 위한 표현입니다.")
    else:
        st.info("아직 AI 판정 결과가 없습니다.")

    st.divider()

    # --- 판정 지원 체크리스트 (FR-06) ---
    st.subheader("판정 지원 체크리스트")
    st.caption("예비전원무단사용·휴지기간사용은 AI 단독 판정이 불가하여 "
               "행정 데이터 대조 여부를 사람이 직접 확인합니다.")
    st.checkbox("휴지신고 여부 확인함")
    st.checkbox("예비전력 계약 여부 확인함")
    st.checkbox("현장조사 필요 판단")

    st.divider()
    status = st.selectbox("처리 상태", ["미판정", "검토중", "현장조사 요청", "오탐 처리", "확정"])
    if st.button("상태 저장", type="primary"):
        st.success(f"'{status}' 상태로 저장했습니다. (Backend 연동 예정)")
