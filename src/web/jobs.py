"""작업 큐: SQLite 에 작업 상태·진행·결과를 기록한다.

작업은 한 번에 하나씩 등록 순서대로 처리한다 (병렬 처리하지 않는다). 상태 흐름:
``queued → running → done | failed | cancelled``. 실패·취소한 작업은 다시 ``queued``로 돌릴 수 있다.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

log = structlog.get_logger(__name__)

DB_FILE = "jobs.db"
JOBS_DIR = "jobs"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
CHECKPOINT_FILE = "checkpoint.json"
WORK_DIR = "work"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUSES: tuple[str, ...] = (
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_CANCELLED,
)
RETRYABLE: frozenset[str] = frozenset({STATUS_FAILED, STATUS_CANCELLED})
FINISHED: frozenset[str] = frozenset({STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED})
# 서버가 멈춘 사이 실행 중이던 작업의 오류 코드 (화면이 문구로 바꾼다)
ERROR_INTERRUPTED = "interrupted"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    options TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    progress_done INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    outputs TEXT NOT NULL DEFAULT '[]',
    result TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    input_deleted INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, id);
"""


class JobError(ValueError):
    """작업 요청이 상태 규칙에 맞지 않는다. code 는 화면이 문구로 바꾼다."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class Job:
    """작업 기록."""

    id: int
    name: str
    size: int
    options: dict[str, Any]
    status: str
    stage: str | None
    progress_done: int
    progress_total: int
    error: str | None
    outputs: list[str]
    result: dict[str, Any] | None
    attempts: int
    input_deleted: bool
    created_at: float
    started_at: float | None
    finished_at: float | None
    updated_at: float


class JobStore:
    """작업 기록과 작업 폴더(``jobs/<id>/input``, ``jobs/<id>/output``)를 다룬다.

    웹 요청 스레드와 워커 스레드가 함께 쓰므로 연결 하나를 잠금으로 보호한다.

    Args:
        data_dir: 데이터 폴더.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(data_dir / DB_FILE), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._tx() as cur:
            cur.executescript(_SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def close(self) -> None:
        """DB 연결을 닫는다."""
        with self._lock:
            self._conn.close()

    # ── 경로 ──────────────────────────────────

    def job_dir(self, job_id: int) -> Path:
        """작업 폴더."""
        return self._data_dir / JOBS_DIR / str(job_id)

    def input_path(self, job: Job) -> Path:
        """작업 입력 파일 경로."""
        return self.job_dir(job.id) / INPUT_DIR / job.name

    def output_dir(self, job_id: int) -> Path:
        """작업 출력 폴더."""
        return self.job_dir(job_id) / OUTPUT_DIR

    def checkpoint_path(self, job_id: int) -> Path:
        """작업 체크포인트 경로 (다시 시도할 때 남은 청크만 추론한다)."""
        return self.job_dir(job_id) / CHECKPOINT_FILE

    def work_dir(self, job_id: int) -> Path:
        """작업 단계 결과 폴더 (다시 시도할 때 전처리·VAD·화자 분리를 건너뛴다)."""
        return self.job_dir(job_id) / WORK_DIR

    # ── 등록·조회 ──────────────────────────────

    def create(self, name: str, source: Path, options: dict[str, Any]) -> Job:
        """업로드한 파일을 작업 폴더로 옮기고 대기 작업을 만든다.

        Args:
            name: 원래 파일 이름 (출력 파일 이름의 바탕).
            source: 업로드가 끝난 파일 (작업 폴더로 옮긴다).
            options: 작업 옵션 (JobOptions.model_dump()).
        """
        now = time.time()
        size = source.stat().st_size
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO jobs (name, size, options, status, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (name, size, json.dumps(options), STATUS_QUEUED, now, now),
            )
            job_id = int(cur.lastrowid or 0)
        target = self.job_dir(job_id) / INPUT_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        log.info("작업 등록", job_id=job_id, name=name, size=size)
        return self.get(job_id)

    def get(self, job_id: int) -> Job:
        """작업 하나.

        Raises:
            JobError: ``job_not_found``.
        """
        with self._tx() as cur:
            row = cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobError("job_not_found")
        return _job_from_row(row)

    def query(self, statuses: list[str] | None = None) -> list[Job]:
        """작업 목록 (최근 등록 순). statuses 를 주면 그 상태만."""
        with self._tx() as cur:
            if statuses:
                marks = ",".join("?" for _ in statuses)
                rows = cur.execute(
                    f"SELECT * FROM jobs WHERE status IN ({marks}) ORDER BY id DESC",
                    statuses,
                ).fetchall()
            else:
                rows = cur.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
        return [_job_from_row(row) for row in rows]

    def counts(self) -> dict[str, int]:
        """상태별 작업 수."""
        result = dict.fromkeys(STATUSES, 0)
        with self._tx() as cur:
            for row in cur.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"):
                result[str(row["status"])] = int(row["n"])
        return result

    # ── 워커가 부르는 상태 변경 ─────────────────

    def claim_next(self) -> Job | None:
        """가장 먼저 등록된 대기 작업을 실행 중으로 바꿔 돌려준다. 없으면 None."""
        now = time.time()
        with self._tx() as cur:
            row = cur.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY id LIMIT 1", (STATUS_QUEUED,)
            ).fetchone()
            if row is None:
                return None
            cur.execute(
                "UPDATE jobs SET status = ?, stage = NULL, progress_done = 0, progress_total = 0,"
                " error = NULL, attempts = attempts + 1, started_at = ?, finished_at = NULL,"
                " updated_at = ? WHERE id = ?",
                (STATUS_RUNNING, now, now, int(row["id"])),
            )
        return self.get(int(row["id"]))

    def update_progress(self, job_id: int, stage: str, done: int, total: int) -> None:
        """실행 중인 작업의 단계·진행을 기록한다."""
        with self._tx() as cur:
            cur.execute(
                "UPDATE jobs SET stage = ?, progress_done = ?, progress_total = ?, updated_at = ?"
                " WHERE id = ? AND status = ?",
                (stage, done, total, time.time(), job_id, STATUS_RUNNING),
            )

    def finish(
        self,
        job_id: int,
        status: str,
        *,
        error: str | None = None,
        outputs: list[str] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        """작업을 끝낸다 (done·failed·cancelled)."""
        now = time.time()
        with self._tx() as cur:
            cur.execute(
                "UPDATE jobs SET status = ?, error = ?, outputs = ?, result = ?, finished_at = ?,"
                " updated_at = ? WHERE id = ?",
                (
                    status,
                    error,
                    json.dumps(outputs or []),
                    json.dumps(result) if result is not None else None,
                    now,
                    now,
                    job_id,
                ),
            )
        log.info("작업 종료", job_id=job_id, status=status, error=error)

    def mark_input_deleted(self, job_id: int) -> None:
        """입력 파일을 지웠다고 기록한다 (다시 시도할 수 없다)."""
        with self._tx() as cur:
            cur.execute(
                "UPDATE jobs SET input_deleted = 1, updated_at = ? WHERE id = ?",
                (time.time(), job_id),
            )

    def recover_interrupted(self) -> int:
        """서버가 멈춘 사이 실행 중이던 작업을 실패로 바꾼다 (다시 시도할 수 있다)."""
        now = time.time()
        with self._tx() as cur:
            cur.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ?, updated_at = ?"
                " WHERE status = ?",
                (STATUS_FAILED, ERROR_INTERRUPTED, now, now, STATUS_RUNNING),
            )
            count = cur.rowcount
        if count:
            log.warning("중단된 작업을 실패로 표시", count=count)
        return count

    # ── 화면이 부르는 상태 변경 ─────────────────

    def cancel_queued(self, job_id: int) -> bool:
        """대기 중이면 취소한다. 대기 중이 아니면 거짓 (실행 중 취소는 워커가 한다)."""
        now = time.time()
        with self._tx() as cur:
            cur.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, updated_at = ?"
                " WHERE id = ? AND status = ?",
                (STATUS_CANCELLED, now, now, job_id, STATUS_QUEUED),
            )
            return cur.rowcount > 0

    def retry(self, job_id: int) -> Job:
        """실패·취소한 작업을 다시 대기열에 넣는다.

        Raises:
            JobError: ``job_not_found``, ``job_not_retryable``, ``input_deleted``.
        """
        job = self.get(job_id)
        if job.status not in RETRYABLE:
            raise JobError("job_not_retryable")
        if job.input_deleted or not self.input_path(job).exists():
            raise JobError("input_deleted")
        with self._tx() as cur:
            cur.execute(
                "UPDATE jobs SET status = ?, stage = NULL, error = NULL, progress_done = 0,"
                " progress_total = 0, finished_at = NULL, updated_at = ? WHERE id = ? AND status = ?",
                (STATUS_QUEUED, time.time(), job_id, job.status),
            )
        log.info("작업 다시 시도", job_id=job_id)
        return self.get(job_id)

    def delete(self, job_id: int) -> None:
        """끝난(또는 대기 중인) 작업과 작업 폴더를 지운다.

        Raises:
            JobError: ``job_not_found``, ``job_running``.
        """
        job = self.get(job_id)
        if job.status == STATUS_RUNNING:
            raise JobError("job_running")
        with self._tx() as cur:
            cur.execute("DELETE FROM jobs WHERE id = ? AND status != ?", (job_id, STATUS_RUNNING))
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        log.info("작업 삭제", job_id=job_id)

    def add_output(self, job_id: int, name: str) -> None:
        """끝난 작업의 출력 목록에 파일 이름을 더한다 (자막을 고칠 때 남기는 원본).

        Raises:
            JobError: ``job_not_found``.
        """
        with self._tx() as cur:
            row = cur.execute("SELECT outputs FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobError("job_not_found")
            outputs = list(json.loads(row["outputs"]))
            if name in outputs:
                return
            cur.execute(
                "UPDATE jobs SET outputs = ?, updated_at = ? WHERE id = ?",
                (json.dumps([*outputs, name]), time.time(), job_id),
            )

    def output_file(self, job_id: int, name: str) -> Path:
        """작업 출력 파일 경로 (기록된 출력 목록에 있는 이름만).

        Raises:
            JobError: ``job_not_found``, ``file_not_found``.
        """
        job = self.get(job_id)
        if name not in job.outputs:
            raise JobError("file_not_found")
        path = self.output_dir(job_id) / name
        if not path.is_file():
            raise JobError("file_not_found")
        return path


def _job_from_row(row: sqlite3.Row) -> Job:
    result = row["result"]
    return Job(
        id=int(row["id"]),
        name=str(row["name"]),
        size=int(row["size"]),
        options=json.loads(row["options"]),
        status=str(row["status"]),
        stage=row["stage"],
        progress_done=int(row["progress_done"]),
        progress_total=int(row["progress_total"]),
        error=row["error"],
        outputs=list(json.loads(row["outputs"])),
        result=json.loads(result) if result else None,
        attempts=int(row["attempts"]),
        input_deleted=bool(row["input_deleted"]),
        created_at=float(row["created_at"]),
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        updated_at=float(row["updated_at"]),
    )
