"""웹 서버: FastAPI 앱 구성과 ``serve`` 실행.

화면(``web/dist``)과 API 를 한 포트에서 낸다. 작업 워커 스레드가 같은 프로세스에서 돌며, 작업 자체는
작업마다 자식 프로세스에서 실행한다.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.core.config import AppConfig
from src.core.logging import setup_logging
from src.web import api
from src.web.auth import SESSION_COOKIE, AccountError, AccountStore, SessionSigner
from src.web.context import Context, WebContext, require_session
from src.web.jobs import JobStore
from src.web.logs import log_path, rotate_if_large
from src.web.secret_store import SecretStore
from src.web.settings_io import ensure_config_file
from src.web.system_probe import SystemInfo
from src.web.worker import JobWorker

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

log = structlog.get_logger(__name__)

# 다른 사이트의 폼 제출(text/plain·form)을 막는다. SameSite=Strict 쿠키와 함께 쓴다
_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_JSON_TYPE = "application/json"
# 업로드는 본문 그대로 받는다. 이 형식도 다른 사이트의 폼으로는 보낼 수 없다 (CORS 사전 요청 대상)
_UPLOAD_TYPE = "application/octet-stream"
_UPLOAD_PATH = "/api/uploads"
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


class LoginRequest(BaseModel):
    """로그인·처음 설정 요청."""

    username: str
    password: str


class SessionState(BaseModel):
    """로그인 상태. setup_required 면 관리 계정이 아직 없다 (처음 설정 화면)."""

    authenticated: bool
    username: str | None = None
    setup_required: bool = False


session_router = APIRouter(prefix="/api/session")


@session_router.post("")
async def login(body: LoginRequest, response: Response, context: Context) -> SessionState:
    """사용자 이름·비밀번호를 확인하고 세션 쿠키를 준다."""
    if not context.accounts.exists():
        raise HTTPException(status_code=409, detail="setup_required")
    if not await asyncio.to_thread(context.accounts.check, body.username, body.password):
        log.warning("웹 로그인 실패", username=body.username)
        await asyncio.sleep(context.failed_login_delay_s)
        raise HTTPException(status_code=401, detail="invalid_credentials")
    _set_session(response, context)
    return SessionState(authenticated=True, username=context.accounts.username())


@session_router.post("/setup")
def setup(body: LoginRequest, response: Response, context: Context) -> SessionState:
    """처음 설정: 관리 계정이 없을 때만 만든다. 있으면 409."""
    try:
        context.accounts.create(body.username, body.password)
    except AccountError as e:
        raise HTTPException(
            status_code=409 if e.code == "account_exists" else 400, detail=e.code
        ) from e
    _set_session(response, context)
    return SessionState(authenticated=True, username=context.accounts.username())


@session_router.get("")
def session_state(request: Request, context: Context) -> SessionState:
    """로그인 상태 (로그인하지 않아도 401 이 아니다: 화면이 처음에 묻는다)."""
    if not context.accounts.exists():
        return SessionState(authenticated=False, setup_required=True)
    if context.signer.verify(request.cookies.get(SESSION_COOKIE)):
        return SessionState(authenticated=True, username=context.accounts.username())
    return SessionState(authenticated=False)


@session_router.delete("")
def logout(response: Response) -> SessionState:
    """세션 쿠키를 지운다."""
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")
    return SessionState(authenticated=False)


def _set_session(response: Response, context: WebContext) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        context.signer.issue(),
        max_age=int(context.signer.max_age_s),
        httponly=True,
        samesite="strict",
        path="/",
    )


async def _guard_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """쓰기 요청의 본문 형식을 제한하고, 모든 응답에 보안 헤더를 붙인다."""
    path = request.url.path
    response: Response
    if request.method in _WRITE_METHODS and path.startswith("/api/"):
        media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        expected = _UPLOAD_TYPE if path == _UPLOAD_PATH else _JSON_TYPE
        # DELETE 처럼 본문이 없는 요청도 화면은 JSON 형식으로 보낸다
        if media_type != expected:
            response = JSONResponse({"detail": "unsupported_media_type"}, status_code=415)
            _add_security_headers(response)
            return response
    response = await call_next(request)
    _add_security_headers(response)
    return response


def _add_security_headers(response: Response) -> None:
    for name, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)


def create_app(context: WebContext, static_dir: Path | None = None) -> FastAPI:
    """웹 앱을 만든다.

    Args:
        context: 요청 처리 객체.
        static_dir: 화면 빌드 폴더 (index.html 이 있으면 / 에서 낸다).
    """
    app = FastAPI(title="SubWyrhta", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.context = context
    app.middleware("http")(_guard_requests)
    app.include_router(session_router)
    app.include_router(api.router, dependencies=[Depends(require_session)])
    if static_dir is not None and (static_dir / "index.html").is_file():
        # API 라우트 뒤에 붙여야 /api 가 정적 파일에 가려지지 않는다
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    else:
        log.warning("웹 화면 파일이 없어 API 만 제공", static_dir=str(static_dir))
    return app


def build_context(
    *,
    config_path: Path,
    data_dir: Path,
    session_hours: int,
    start_worker: bool = True,
    start_probe: bool = True,
    worker_target: Any = None,
) -> WebContext:
    """데이터 폴더의 작업 기록·계정·워커로 컨텍스트를 만든다.

    Args:
        config_path: 설정 파일 (화면에서 저장한다).
        data_dir: 데이터 폴더 (작업·업로드·로그·비밀 저장소).
        session_hours: 세션 유지 시간.
        start_worker: 작업 워커를 시작한다 (테스트에서 끈다).
        start_probe: 시스템 점검을 시작한다 (테스트에서 끈다).
        worker_target: 작업 자식 프로세스 진입 함수 (테스트에서 바꾼다).
    """
    store = JobStore(data_dir)
    log_file = log_path(data_dir)

    def load_config(overrides: dict[str, Any] | None) -> AppConfig:
        return AppConfig.load(config_path=config_path, cli_overrides=overrides)

    worker_kwargs: dict[str, Any] = {
        "data_dir": data_dir,
        "load_config": load_config,
        "log_file": log_file,
    }
    if worker_target is not None:
        worker_kwargs["target"] = worker_target
    worker = JobWorker(store, **worker_kwargs)
    accounts = AccountStore(SecretStore.for_data_dir(data_dir))
    signer = SessionSigner(accounts.session_key, max_age_s=session_hours * 3600)
    system = SystemInfo()
    context = WebContext(
        data_dir=data_dir,
        config_path=config_path,
        store=store,
        worker=worker,
        accounts=accounts,
        signer=signer,
        log_file=log_file,
        system=system,
        started_at=time.time(),
    )
    if start_worker:
        worker.start()
    if start_probe:
        system.start()
    if not accounts.exists():
        log.warning("관리 계정이 없음: 웹 화면을 열어 계정을 만드세요 (처음 설정)")
    return context


def serve(
    *,
    config_path: Path,
    data_dir: Path,
    host: str | None,
    port: int | None,
    static_dir: Path | None,
) -> None:
    """웹 서버를 실행한다 (Ctrl+C·SIGTERM 으로 끝난다).

    Args:
        config_path: 설정 파일. 없으면 기본값으로 만든다.
        data_dir: 데이터 폴더.
        host: 바인딩 주소. None 이면 설정 파일 값.
        port: 포트. None 이면 설정 파일 값.
        static_dir: 화면 빌드 폴더. None 이면 설정 파일 값.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    ensure_config_file(config_path)
    config = AppConfig.load(config_path=config_path)
    log_file = log_path(data_dir)
    rotate_if_large(log_file)
    setup_logging(level=config.logging.level, json_log_file=log_file)

    context = build_context(
        config_path=config_path, data_dir=data_dir, session_hours=config.web.session_hours
    )
    app = create_app(context, static_dir or Path(config.web.static_dir))
    bind_host = host or config.web.host
    bind_port = port or config.web.port
    log.info("웹 서버 시작", url=f"http://{bind_host}:{bind_port}", data_dir=str(data_dir))
    # log_config=None: 이 프로세스의 structlog 형식을 그대로 쓴다 (uvicorn 기본 설정이 덮어쓰지 않게)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=bind_host,
            port=bind_port,
            log_config=None,
            access_log=False,
            lifespan="off",
        )
    )
    try:
        server.run()
    finally:
        context.worker.stop()
        context.store.close()
        log.info("웹 서버 종료")
