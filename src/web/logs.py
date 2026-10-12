"""로그 보기: JSON 로그 파일(한 줄에 JSON 하나)의 최근 항목을 읽는다.

웹 서버와 작업 자식 프로세스가 같은 파일에 덧붙인다. 화면은 cursor(바이트 위치)를 넘겨 그 뒤 항목만
받는다. 파일이 줄었으면(새로 시작) 처음부터 다시 읽는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

LOG_DIR = "logs"
LOG_FILE = "subwyrhta.log"
# 서버를 시작할 때 이 크기를 넘으면 이전 파일로 돌린다 (자식 프로세스도 같은 파일에 쓰므로 실행 중에는 돌리지 않는다)
ROTATE_BYTES = 20 * 1024 * 1024
# 처음 읽을 때 파일 끝에서 거슬러 읽는 양
TAIL_BYTES = 1024 * 1024
LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
# 메시지 본문에 덧붙이지 않는 키 (따로 칸이 있다)
_RESERVED = frozenset({"event", "level", "logger", "timestamp", "job_id", "exception"})


@dataclass(frozen=True)
class LogEntry:
    """로그 항목."""

    time: str
    level: str
    logger: str
    job_id: int | None
    message: str


@dataclass(frozen=True)
class LogChunk:
    """읽은 항목과 다음에 이어 읽을 위치. reset 이면 화면이 목록을 새로 그린다."""

    entries: list[LogEntry]
    cursor: str
    reset: bool


def log_path(data_dir: Path) -> Path:
    """데이터 폴더의 로그 파일."""
    return data_dir / LOG_DIR / LOG_FILE


def rotate_if_large(path: Path) -> None:
    """파일이 크면 ``.1``로 돌린다 (서버 시작 때만 부른다)."""
    if path.exists() and path.stat().st_size > ROTATE_BYTES:
        path.replace(path.with_suffix(path.suffix + ".1"))


def read_log(path: Path, *, cursor: str | None, level: str, limit: int) -> LogChunk:
    """cursor 뒤의 항목 중 level 이상인 것을 최대 limit 개 (오래된 것부터).

    Args:
        path: 로그 파일.
        cursor: 이전 응답의 cursor. None 이면 파일 끝 근처부터.
        level: 최소 레벨.
        limit: 최대 항목 수 (넘으면 최근 것만).
    """
    if not path.exists():
        return LogChunk(entries=[], cursor="0", reset=cursor is None)
    size = path.stat().st_size
    start, reset = _start_offset(cursor, size)
    with path.open("rb") as f:
        f.seek(start)
        data = f.read(size - start)
    lines = data.split(b"\n")
    if reset and start > 0:
        lines = lines[1:]  # 거슬러 읽은 첫 줄은 중간에서 잘렸을 수 있다
    # 마지막 줄이 아직 다 쓰이지 않았으면 다음에 읽는다
    tail = lines[-1]
    consumed = size - len(tail)
    minimum = LEVELS.index(level) if level in LEVELS else 0
    entries = [
        entry
        for line in lines[:-1]
        if (entry := _parse(line)) is not None and _rank(entry.level) >= minimum
    ]
    return LogChunk(entries=entries[-limit:], cursor=str(consumed), reset=reset)


def _start_offset(cursor: str | None, size: int) -> tuple[int, bool]:
    if cursor is not None and cursor.isdigit() and int(cursor) <= size:
        return int(cursor), False
    return max(0, size - TAIL_BYTES), True


def _rank(level: str) -> int:
    return LEVELS.index(level) if level in LEVELS else 0


def _parse(line: bytes) -> LogEntry | None:
    text = line.strip()
    if not text:
        return None
    try:
        record: Any = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return LogEntry(
            time="", level="INFO", logger="", job_id=None, message=text.decode(errors="replace")
        )
    if not isinstance(record, dict):
        return None
    extras = " ".join(f"{k}={v}" for k, v in record.items() if k not in _RESERVED)
    message = str(record.get("event", ""))
    if extras:
        message = f"{message} {extras}"
    if record.get("exception"):
        message = f"{message}\n{record['exception']}"
    job_id = record.get("job_id")
    return LogEntry(
        time=str(record.get("timestamp", "")),
        level=str(record.get("level", "info")).upper(),
        logger=str(record.get("logger", "")),
        job_id=int(job_id) if isinstance(job_id, int) else None,
        message=message,
    )
