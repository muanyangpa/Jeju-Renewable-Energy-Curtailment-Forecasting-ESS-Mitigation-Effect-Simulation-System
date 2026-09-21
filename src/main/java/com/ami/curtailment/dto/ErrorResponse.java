package com.ami.curtailment.dto;

import lombok.Getter;

/**
 * AI 서버 에러 응답 형식 (예: 422 INVALID_HOUR_SET). Backend는 별도 매핑 없이 그대로 릴레이.
 */
@Getter
public class ErrorResponse {
    private final String error_code;
    private final String message;

    public ErrorResponse(String error_code, String message) {
        this.error_code = error_code;
        this.message = message;
    }
}
