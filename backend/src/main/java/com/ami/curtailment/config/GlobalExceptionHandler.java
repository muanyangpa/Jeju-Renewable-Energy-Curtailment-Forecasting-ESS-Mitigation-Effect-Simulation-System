package com.ami.curtailment.config;

import com.ami.curtailment.dto.ErrorResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.reactive.function.client.WebClientResponseException;

/**
 * AI 서버가 에러를 반환하면 Backend가 가공을 최소화해서 클라이언트(Frontend)에 전달한다.
 *
 * 통신규격 확정서 v1.1 03장 반영: 4xx와 500의 응답 모양이 다르다.
 * - 4xx({@code error_code, message} JSON): 그대로 릴레이 (본래 로직 유지)
 * - 500(평문 "Internal Server Error"): JSON이 아니라서 그대로 넘기면 Frontend가 파싱 실패할
 *   수 있음 - ErrorResponse 형식으로 감싸서 내려줌. v1.2에서 AI 서버가 500도 JSON으로
 *   바꾸면 이 분기는 제거 가능.
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(WebClientResponseException.class)
    public ResponseEntity<?> handleAiServerError(WebClientResponseException e) {
        if (e.getStatusCode().is5xxServerError()) {
            return ResponseEntity.status(e.getStatusCode())
                    .body(new ErrorResponse("AI_SERVER_ERROR",
                            "AI 서버 내부 오류: " + e.getResponseBodyAsString()));
        }
        // 4xx: AI 서버가 준 원본 에러 본문({error_code, message} 형식)을 그대로 통과시킴
        return ResponseEntity.status(e.getStatusCode()).body(e.getResponseBodyAsString());
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleBadRequest(IllegalArgumentException e) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                .body(new ErrorResponse("INVALID_REQUEST", e.getMessage()));
    }
}
