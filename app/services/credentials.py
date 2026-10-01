"""인증키 조회 — 환경변수 다음에 macOS 키체인.

[왜 키체인인가]
매일 자동으로 도는 스크립트(scripts/kim_validation_log.py)가 키를 필요로 한다. 선택지는 셋이다.

  ~/.zshrc에 export   평문으로 남는다 + cron/launchd는 로그인 셸을 거치지 않아 안 보인다
  .env 파일           평문으로 남는다 (gitignore돼 있어 커밋 사고는 막혀 있다)
  키체인              암호화 저장 + launchd에서도 읽힌다   <= 채택

자동화하면 키가 무인으로 매일 쓰이고, 그때 평문 파일에 있으면 백업·동기화·화면공유로
새어나갈 경로가 늘어난다. 시연 준비로 화면을 공유할 일이 있으면 특히 그렇다.

[이 모듈이 지키는 것]
- 키를 **반환만** 하고 로그·예외 메시지·저장소 어디에도 남기지 않는다.
- `security`를 셸 없이(list 인자) 부른다 — 서비스 이름에 셸 메타문자가 들어가도 안전하다.
- macOS가 아니거나 `security`가 없으면 조용히 건너뛴다(예외를 올리지 않는다).
- 타임아웃을 둔다. 키체인이 잠겨 GUI 프롬프트가 뜨면 자동화가 무한정 매달릴 수 있다.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

KEYCHAIN_TIMEOUT_SEC = 10.0


def from_keychain(service: str, timeout: float = KEYCHAIN_TIMEOUT_SEC) -> str | None:
    """macOS 로그인 키체인에서 generic password를 읽는다. 없거나 못 읽으면 None.

    실패를 예외로 올리지 않는다 — 호출부는 '환경변수도 키체인도 없다'는 하나의 안내만
    보여주면 되고, 키체인 미설정은 정상 상태이기 때문이다.
    """
    if sys.platform != "darwin" or not shutil.which("security"):
        return None
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    v = r.stdout.strip()
    return v or None


def resolve_secret(env_name: str, keychain_service: str | None = None) -> str | None:
    """환경변수 -> 키체인 순으로 찾는다. 환경변수가 이긴다(일회성 override를 위해).

    키체인 서비스 이름은 기본적으로 환경변수 이름과 같게 둔다 — 두 이름이 갈리면
    어디에 무엇을 넣었는지 추적이 안 된다.
    """
    v = os.environ.get(env_name)
    if v and v.strip():
        return v.strip()
    return from_keychain(keychain_service or env_name)


def setup_hint(env_name: str) -> str:
    """키를 찾지 못했을 때 보여줄 안내. **키 값 자체는 절대 넣지 않는다.**"""
    return (
        f"  1) 키체인에 한 번만 등록 (권장, 평문으로 남지 않음)\n"
        f"     security add-generic-password -a \"$USER\" -s {env_name} \\\n"
        f"       -T /usr/bin/security -U -w\n"
        f"     -> 실행하면 비밀번호를 물어봅니다. 거기에 발급받은 키를 붙여넣으세요.\n"
        f"        (-w 뒤에 키를 직접 쓰면 셸 히스토리에 남습니다 — 쓰지 마세요)\n"
        f"        -T /usr/bin/security 는 자동 실행 때 접근 허용 창이 뜨지 않게 합니다.\n"
        f"  2) 또는 이번 셸에서만\n"
        f"     export {env_name}=발급받은키")
