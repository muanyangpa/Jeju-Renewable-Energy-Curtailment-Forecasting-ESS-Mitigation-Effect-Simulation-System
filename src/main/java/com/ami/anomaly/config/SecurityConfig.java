package com.ami.anomaly.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;

/**
 * 기본 Spring Security 설정.
 * 데모 단계에서는 permitAll로 열어두고, 팀 확정 후 JWT/세션 방식으로 확장 예정.
 */
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf(csrf -> csrf.disable()) // REST API + 별도 프론트(Streamlit)라 세션 기반 CSRF 비활성화
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/**").permitAll() // TODO: 팀 확정 후 인증 적용
                .anyRequest().authenticated()
            );
        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
