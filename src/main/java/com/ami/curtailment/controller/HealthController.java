package com.ami.curtailment.controller;

import com.ami.curtailment.service.AiClientService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * Backend 자체 헬스체크 + AI 서버 연결 상태 확인.
 * 통신규격 확정서 v1.0 04장: GET /api/v1/health를 Backend 헬스체크에서 호출.
 */
@RestController
@RequestMapping("/api/v1/health")
@RequiredArgsConstructor
public class HealthController {

    private final AiClientService aiClientService;

    @GetMapping
    public Map<String, Object> health() {
        boolean aiServerUp = aiClientService.checkHealth();
        return Map.of(
                "status", "UP",
                "aiServer", aiServerUp ? "UP" : "DOWN"
        );
    }
}
