"""요청 처리 컨텍스트와 로그인 검사 의존성."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import Depends, HTTPException, Request

from src.core.config import AppConfig
from src.web.auth import SESSION_COOKIE, AccountStore, SessionSigner

if TYPE_CHECKING:
    from pathlib import Path

    from src.web.jobs import JobStore
    from src.web.system_probe import SystemInfo
    from src.web.worker import JobWorker

# 로그인 실패 응답을 늦춰 비밀번호 대입을 느리게 한다
FAILED_LOGIN_DELAY_S = 1.0


@dataclass(frozen=True)
class WebContext:
    """요청 처리에 필요한 객체."""

    data_dir: Path
    config_path: Path
    store: JobStore
    worker: JobWorker
    accounts: AccountStore
    signer: SessionSigner
    log_file: Path
    system: SystemInfo
    started_at: float
    failed_login_delay_s: float = FAILED_LOGIN_DELAY_S

    def config(self, overrides: dict[str, Any] | None = None) -> AppConfig:
        """지금 설정 파일 값 (환경 변수 오버라이드 포함). 화면에서 저장하면 다음 요청부터 반영된다."""
        return AppConfig.load(config_path=self.config_path, cli_overrides=overrides)


def get_context(request: Request) -> WebContext:
    """앱에 붙인 컨텍스트 (FastAPI 의존성)."""
    return cast("WebContext", request.app.state.context)


Context = Annotated[WebContext, Depends(get_context)]


def require_session(request: Request, context: Context) -> None:
    """로그인한 요청만 통과시킨다 (FastAPI 의존성).

    Raises:
        HTTPException: 401, 세션 쿠키가 없거나 만료·변조됐다.
    """
    if not context.signer.verify(request.cookies.get(SESSION_COOKIE)):
        raise HTTPException(status_code=401, detail="not_authenticated")
