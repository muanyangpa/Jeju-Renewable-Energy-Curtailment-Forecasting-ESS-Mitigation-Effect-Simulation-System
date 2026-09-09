package com.ami.curtailment.dto;

import lombok.Getter;

/**
 * 통신규격 확정서 v1.0 04장 에러 응답 형식.
 * AI 서버가 이 형식으로 에러를 내려주면 Backend가 그대로 전달(pass-through).
 */
@Getter
public class ErrorResponse {
    private final String errorCode;
    private final String message;
    private final Integer hour; // nullable - hour와 무관한 에러(UNSUPPORTED_REGION 등)는 null

    public ErrorResponse(String errorCode, String message, Integer hour) {
        this.errorCode = errorCode;
        this.message = message;
        this.hour = hour;
    }
}
