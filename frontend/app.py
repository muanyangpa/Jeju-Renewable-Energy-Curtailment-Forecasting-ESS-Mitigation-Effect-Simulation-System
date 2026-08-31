"""
화면 1 — 이상 의심 세대 목록(큐)
메인 화면. 위반유형·정렬 필터로 의심 세대를 훑어보고, 선택하면 상세 화면으로 이동.
"""

import streamlit as st
import pandas as pd

from api_client import get_meters, get_anomalies
from models import VIOLATION_TYPE_DISPLAY, label_for

st.set_page_config(page_title="AMI 이상탐지 관제", layout="wide")

st.title("⚡ AMI 이상탐지 관제")
st.caption("AI가 1차로 걸러낸 이상 의심 세대 목록 — 판정 권한은 사람에게 있습니다.")

# --- 데이터 로드 ---
meters, err = get_meters()
if err:
    st.error(f"⚠️ {err}")
    st.info("Spring Boot Backend(localhost:8080)가 실행 중인지 확인해주세요.")
    st.stop()

if not meters:
    st.warning("등록된 세대가 없습니다. Backend에 테스트 데이터를 먼저 등록해주세요.")
    st.stop()

# 세대별 최신 이상탐지 결과 취합
rows = []
for meter in meters:
    anomalies, _ = get_anomalies(meter["id"])
    latest = anomalies[-1] if anomalies else None
    rows.append({
        "id": meter["id"],
        "세대ID": meter["consumerNo"],
        "위반유형": label_for(latest["violationType"]) if latest else "미판정",
        "이상점수": round(latest["anomalyScore"], 3) if latest else None,
        "상태": "검토중" if latest else "데이터 없음",
    })

df = pd.DataFrame(rows)

# --- 필터 ---
col1, col2 = st.columns([3, 1])
with col1:
    type_options = ["전체"] + [v["label"] for v in VIOLATION_TYPE_DISPLAY.values()]
    selected_type = st.radio("위반유형 필터", type_options, horizontal=True)
with col2:
    sort_desc = st.checkbox("이상점수 높은 순", value=True)

filtered = df if selected_type == "전체" else df[df["위반유형"] == selected_type]
if sort_desc:
    filtered = filtered.sort_values("이상점수", ascending=False, na_position="last")

st.divider()

# --- 목록 테이블 ---
st.subheader(f"의심 세대 목록 ({len(filtered)}건)")

for _, row in filtered.iterrows():
    c1, c2, c3, c4, c5 = st.columns([2, 2, 1.5, 1.5, 1])
    c1.write(f"**{row['세대ID']}**")
    c2.write(row["위반유형"])
    c3.write(row["이상점수"] if row["이상점수"] is not None else "-")
    c4.write(row["상태"])
    if c5.button("상세보기", key=f"detail_{row['id']}"):
        st.session_state["selected_meter_id"] = row["id"]
        st.session_state["selected_consumer_no"] = row["세대ID"]
        st.switch_page("pages/1_세대_상세.py")
