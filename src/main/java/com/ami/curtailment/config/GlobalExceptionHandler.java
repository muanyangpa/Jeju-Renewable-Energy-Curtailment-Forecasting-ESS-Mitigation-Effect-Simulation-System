package com.ami.curtailment.config;

import com.ami.curtailment.dto.ErrorResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.reactive.function.client.WebClientResponseException;

/**
 * AI 서버가 에러(422 INVALID_HOUR_SET 등)를 반환하면
 * Backend가 가공하지 않고 그대로 클라이언트(Frontend)에 전달한다.
 * AI 서버 README 검증 책임 분담: "Backend는 이 에러코드를 그대로 릴레이하면 됨 (별도 매핑 불필요)".
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(WebClientResponseException.class)
    public ResponseEntity<String> handleAiServerError(WebClientResponseException e) {
        // AI 서버가 준 원본 에러 본문({error_code, message} 형식)을 그대로 통과시킴
        return ResponseEntity.status(e.getStatusCode()).body(e.getResponseBodyAsString());
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleBadRequest(IllegalArgumentException e) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                .body(new ErrorResponse("INVALID_REQUEST", e.getMessage()));
    }
}
