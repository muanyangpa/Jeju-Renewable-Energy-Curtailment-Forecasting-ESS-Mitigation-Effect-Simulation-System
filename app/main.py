"""
AI 서버 (Python/FastAPI) — Backend(Java/Spring Boot)가 REST로 호출하는 내부 서비스.
계획서 02장·06장·09장에서 정의한 역할: ②(날씨->발전량)·③(출력제어 확률예측) 단계를 담당하고,
④(ESS 충방전 결정)에 필요한 계산도 /ess/simulate로 제공한다.

실행: uvicorn app.main:app --reload --port 8001
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import EssSimulateRequest, EssSimulateResponse, PredictRequest, PredictResponse
from app.services import ess_simulation
from app.services.predict import predict as run_predict

app = FastAPI(
    title="출력제어 예측 AI 서버",
    description="제주 재생에너지 출력제어 예측 및 ESS 완화효과 시뮬레이션 — 내부 AI 서버 (Backend 전용)",
    version="0.1.0",
)


def _error_code_from_message(msg: str) -> tuple[str, str]:
    if msg.startswith("INVALID_HOUR_SET:"):
        return "INVALID_HOUR_SET", msg.split(":", 1)[1]
    return "VALIDATION_ERROR", msg


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # pydantic 검증 실패 시 도메인 에러코드로 변환해 반환한다 (09장 검증 책임 분담: AI 서버가 최종 담당)
    first = exc.errors()[0]
    raw_msg = str(first.get("msg", ""))
    code, msg = _error_code_from_message(raw_msg.replace("Value error, ", ""))
    return JSONResponse(status_code=422, content={"error_code": code, "message": msg})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """weather 배열 24개 검증은 여기(AI 서버)에서 최종 수행 — 09장 참고.
    Backend는 필드 존재·타입 같은 기본 형식만 사전 확인하고, 이 규칙 위반 시
    422 + {"error_code": "INVALID_HOUR_SET", ...}를 그대로 릴레이하면 된다.
    """
    return run_predict(req)


@app.post("/ess/simulate", response_model=EssSimulateResponse)
def ess_simulate(req: EssSimulateRequest):
    """04장 02번(기본 시뮬레이션)·03번(용량 조정 슬라이더)이 공유하는 단일 엔드포인트 —
    rated_power_mw만 바꿔서 여러 번 호출하면 슬라이더 UI가 된다."""
    if req.method == "naive_upper_bound":
        curtailed_hours = sum(1 for v in req.hourly_curtailment_mwh if v > 0)
        total = sum(req.hourly_curtailment_mwh)
        result = ess_simulation.simulate_naive_upper_bound(curtailed_hours, req.rated_power_mw, total)
    else:
        result = ess_simulation.simulate_hourly_capped(req.hourly_curtailment_mwh, req.rated_power_mw)
    d = result.to_dict()
    return EssSimulateResponse(
        rated_power_mw=d["rated_power_mw"], method=d["method"],
        total_curtailment_mwh=d["total_curtailment_mwh"], total_absorbed_mwh=d["total_absorbed_mwh"],
        absorption_rate=d["absorption_rate"],
    )
