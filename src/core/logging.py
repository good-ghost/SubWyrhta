"""structlog 기반 로깅 설정."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    json_log_file: str | Path | None = None,
) -> None:
    """structlog 로깅을 설정한다.

    Args:
        level: 로그 레벨 (DEBUG / INFO / WARNING / ERROR).
        log_file: 로그 파일 경로. None이면 콘솔만 출력.
        json_log_file: 한 줄에 JSON 하나로 쓰는 로그 파일 (웹 화면의 로그 보기가 읽는다).
            웹 서버와 작업 자식 프로세스가 같은 파일에 덧붙인다.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # stdlib 로거 핸들러 설정
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
    ]

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(file_path), encoding="utf-8"))

    json_handler: logging.Handler | None = None
    if json_log_file is not None:
        json_path = Path(json_log_file)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_handler = logging.FileHandler(str(json_path), encoding="utf-8")
        handlers.append(json_handler)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
        force=True,
    )

    # structlog 프로세서 체인
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # stdlib 핸들러에 structlog 포맷터 적용.
    # foreign_pre_chain: structlog 를 거치지 않은 stdlib 로그(uvicorn 등)에도 시각·레벨을 붙인다
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
    )
    json_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
    )

    for handler in logging.root.handlers:
        handler.setFormatter(json_formatter if handler is json_handler else formatter)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """바인딩된 structlog 로거를 반환한다.

    Args:
        name: 로거 이름. None이면 호출자 모듈명 사용.

    Returns:
        structlog BoundLogger 인스턴스.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
