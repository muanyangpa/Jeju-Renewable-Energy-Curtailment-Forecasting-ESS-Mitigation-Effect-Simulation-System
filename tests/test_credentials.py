"""인증키 조회 — 순서, 그리고 '키가 새지 않는다'는 성질을 고정한다.

키는 로그·예외 메시지·저장소 어디에도 남으면 안 된다. 이 파일이 그 계약을 지킨다.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from app.services import credentials as C

SECRET = "SECRET-TEST-VALUE-1234"


def test_환경변수가_키체인을_이긴다(monkeypatch):
    """일회성 override가 가능해야 한다 — 다른 키로 한 번만 시험할 때 쓴다."""
    monkeypatch.setenv("TEST_KEY", SECRET)
    monkeypatch.setattr(C, "from_keychain", lambda *a, **k: "KEYCHAIN-VALUE")
    assert C.resolve_secret("TEST_KEY") == SECRET


def test_환경변수가_없으면_키체인을_본다(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    monkeypatch.setattr(C, "from_keychain", lambda *a, **k: SECRET)
    assert C.resolve_secret("TEST_KEY") == SECRET


def test_빈_환경변수는_없는_것으로_친다(monkeypatch):
    """export TEST_KEY= 처럼 비워둔 경우 키체인으로 넘어가야 한다."""
    monkeypatch.setenv("TEST_KEY", "   ")
    monkeypatch.setattr(C, "from_keychain", lambda *a, **k: SECRET)
    assert C.resolve_secret("TEST_KEY") == SECRET


def test_키체인_서비스_이름은_기본이_환경변수명(monkeypatch):
    """두 이름이 갈리면 어디에 무엇을 넣었는지 추적이 안 된다."""
    monkeypatch.delenv("TEST_KEY", raising=False)
    seen = []
    monkeypatch.setattr(C, "from_keychain", lambda s, **k: seen.append(s) or None)
    C.resolve_secret("TEST_KEY")
    assert seen == ["TEST_KEY"]


class TestKeychainCall:
    def test_셸_없이_리스트_인자로_부른다(self, monkeypatch):
        """서비스 이름에 셸 메타문자가 들어가도 안전해야 한다."""
        calls = []

        def fake(cmd, **kw):
            calls.append((cmd, kw))
            return subprocess.CompletedProcess(cmd, 0, stdout=SECRET + "\n", stderr="")
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(C.shutil, "which", lambda _: "/usr/bin/security")
        monkeypatch.setattr(C.subprocess, "run", fake)
        assert C.from_keychain("A; rm -rf /") == SECRET
        cmd, kw = calls[0]
        assert isinstance(cmd, list) and "A; rm -rf /" in cmd
        assert kw.get("shell") is not True and kw.get("timeout")

    def test_맥이_아니면_건너뛴다(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert C.from_keychain("X") is None

    @pytest.mark.parametrize("exc", [subprocess.TimeoutExpired("security", 1), OSError("boom")])
    def test_실패는_예외가_아니라_None(self, monkeypatch, exc):
        """키체인이 잠겨 프롬프트가 뜨면 자동화가 매달릴 수 있다 — 타임아웃 후 조용히 포기한다."""
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(C.shutil, "which", lambda _: "/usr/bin/security")
        def raiser(*a, **k):
            raise exc
        monkeypatch.setattr(C.subprocess, "run", raiser)
        assert C.from_keychain("X") is None

    def test_미등록이면_None(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(C.shutil, "which", lambda _: "/usr/bin/security")
        monkeypatch.setattr(C.subprocess, "run",
                            lambda c, **k: subprocess.CompletedProcess(c, 44, stdout="", stderr="not found"))
        assert C.from_keychain("X") is None


def test_안내문에_키가_들어가지_않는다(monkeypatch):
    """안내문은 키가 없을 때 뜨므로 값이 섞이면 안 된다."""
    monkeypatch.setenv("KMA_AUTH_KEY", SECRET)
    assert SECRET not in C.setup_hint("KMA_AUTH_KEY")


def test_실패_예외에_키가_새지_않는다(monkeypatch):
    """키가 잘못돼 실패할 때 메시지에 값이 찍히면 로그에 영구히 남는다."""
    from app.services import kim_forecast as K
    monkeypatch.delenv("KMA_AUTH_KEY", raising=False)
    monkeypatch.setattr(C, "from_keychain", lambda *a, **k: None)
    with pytest.raises(RuntimeError) as e:
        K.fetch_day(__import__("datetime").date(2026, 9, 30))
    assert SECRET not in str(e.value)
    assert "키체인" in str(e.value)
