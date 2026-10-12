"""작업 워커: 대기 작업을 한 번에 하나씩 자식 프로세스로 실행한다.

여러 파일을 병렬로 처리하지 않는다. 워커 스레드 하나가 작업을 등록 순서대로 꺼내 spawn 자식
프로세스에서 파이프라인을 돌리고, 자식이 보내는 진행을 작업 기록에 남긴다.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import structlog
from pydantic import ValidationError

from src.web.job_options import JobOptions
from src.web.jobs import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    Job,
    JobStore,
)
from src.web.runner import EVENT_DONE, EVENT_FAILED, EVENT_PROGRESS, JobPayload, run_job_process

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.config import AppConfig

log = structlog.get_logger(__name__)

# 대기 작업이 없을 때 다시 볼 간격
POLL_INTERVAL_S = 1.0
# 자식 프로세스 이벤트를 기다리는 간격 (취소 요청을 이 간격 안에 반영한다)
EVENT_WAIT_S = 0.5
# 취소할 때 terminate 뒤 kill 하기까지 기다리는 시간
TERMINATE_GRACE_S = 5.0
# 끝난 업로드 조각(작업으로 등록하지 않은 파일)을 지우는 기준과 간격
STALE_UPLOAD_S = 24 * 3600
UPLOAD_SWEEP_INTERVAL_S = 600.0
UPLOADS_DIR = "uploads"
# 자식 프로세스가 결과를 보내지 않고 끝났을 때의 오류 코드 (OOM 종료 등)
ERROR_CRASHED = "process_crashed"
ERROR_INVALID_OPTIONS = "invalid_options"


class ProcessLike(Protocol):
    """워커가 다루는 자식 프로세스 (multiprocessing.Process 와 같은 부분)."""

    @property
    def exitcode(self) -> int | None: ...

    def start(self) -> None: ...
    def is_alive(self) -> bool: ...
    def join(self, timeout: float | None = None) -> None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


@dataclass(frozen=True)
class CurrentJob:
    """지금 실행 중인 작업."""

    job_id: int
    name: str
    started_at: float


JobTarget = Callable[[JobPayload, Any], None]


class JobWorker:
    """작업 워커 스레드.

    Args:
        store: 작업 기록.
        data_dir: 데이터 폴더 (업로드 조각 정리).
        load_config: 지금 설정 파일 값을 읽는 함수 (작업을 시작할 때마다 부른다).
        log_file: 자식 프로세스가 덧붙일 JSON 로그 파일.
        target: 자식 프로세스 진입 함수 (테스트에서 바꾼다).
    """

    def __init__(
        self,
        store: JobStore,
        *,
        data_dir: Path,
        load_config: Callable[[dict[str, Any] | None], AppConfig],
        log_file: Path,
        target: JobTarget = run_job_process,
    ) -> None:
        self._store = store
        self._data_dir = data_dir
        self._load_config = load_config
        self._log_file = log_file
        self._target = target
        self._context = multiprocessing.get_context("spawn")
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._current: CurrentJob | None = None
        self._cancel_requested: int | None = None
        self._thread = threading.Thread(target=self._loop, name="job-worker", daemon=True)
        self._last_sweep = 0.0

    # ── 바깥에서 부르는 것 ──────────────────────

    def start(self) -> None:
        """중단된 작업을 실패로 표시하고 스레드를 시작한다."""
        self._store.recover_interrupted()
        self._thread.start()

    def stop(self, timeout: float = TERMINATE_GRACE_S * 2) -> None:
        """멈춘다. 실행 중인 자식 프로세스도 끝낸다 (작업은 취소로 남는다)."""
        self._stop.set()
        self._wake.set()
        current = self.current
        if current is not None:
            self.cancel(current.job_id)
        self._thread.join(timeout)

    def notify(self) -> None:
        """새 작업이 들어왔다 (대기 간격을 기다리지 않고 바로 본다)."""
        self._wake.set()

    @property
    def current(self) -> CurrentJob | None:
        """지금 실행 중인 작업. 없으면 None."""
        with self._lock:
            return self._current

    @property
    def alive(self) -> bool:
        """워커 스레드가 돌고 있다."""
        return self._thread.is_alive()

    def cancel(self, job_id: int) -> bool:
        """실행 중인 작업이면 취소를 요청한다 (자식 프로세스를 끝낸다). 아니면 거짓."""
        with self._lock:
            if self._current is None or self._current.job_id != job_id:
                return False
            self._cancel_requested = job_id
        log.info("실행 중 작업 취소 요청", job_id=job_id)
        return True

    # ── 스레드 ───────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sweep_uploads()
            job = self._store.claim_next()
            if job is None:
                self._wake.wait(POLL_INTERVAL_S)
                self._wake.clear()
                continue
            try:
                self._run(job)
            except Exception:
                # 한 작업의 예상하지 못한 오류로 워커가 멈추면 이후 작업이 모두 밀린다
                log.exception("작업 처리 중 워커 오류", job_id=job.id)
                self._store.finish(job.id, STATUS_FAILED, error=ERROR_CRASHED)
            finally:
                with self._lock:
                    self._current = None
                    self._cancel_requested = None

    def _run(self, job: Job) -> None:
        try:
            options = JobOptions.model_validate(job.options)
        except ValidationError as e:
            log.error("작업 옵션이 지금 설정과 맞지 않음", job_id=job.id, error=str(e))
            self._store.finish(job.id, STATUS_FAILED, error=ERROR_INVALID_OPTIONS)
            return

        config = self._load_config(options.to_overrides())
        payload = JobPayload(
            job_id=job.id,
            input_path=str(self._store.input_path(job)),
            output_dir=str(self._store.output_dir(job.id)),
            checkpoint_path=str(self._store.checkpoint_path(job.id)),
            config=dataclasses.asdict(config),
            formats=options.formats,
            resume=job.attempts > 1,
            log_file=str(self._log_file),
            log_level=config.logging.level,
            work_dir=str(self._store.work_dir(job.id)),
            audio_track=options.audio_track,
        )
        events: Any = self._context.Queue()
        process: ProcessLike = self._context.Process(
            target=self._target, args=(payload, events), name=f"subwyrhta-job-{job.id}"
        )
        with self._lock:
            self._current = CurrentJob(job_id=job.id, name=job.name, started_at=time.time())
        log.info("작업 시작", job_id=job.id, name=job.name, attempt=job.attempts)
        process.start()
        outcome = self._watch(job.id, process, events)
        self._complete(job, config, outcome)

    def _watch(self, job_id: int, process: ProcessLike, events: Any) -> tuple[Any, ...] | None:
        """자식 프로세스가 끝날 때까지 이벤트를 기록한다. 마지막 결과 이벤트(없으면 None)."""
        final: tuple[Any, ...] | None = None
        while True:
            if self._cancel_requested == job_id:
                _terminate(process)
                return ("cancelled",)
            try:
                event = events.get(timeout=EVENT_WAIT_S)
            except queue.Empty:
                if not process.is_alive():
                    break
                continue
            if event[0] == EVENT_PROGRESS:
                self._store.update_progress(job_id, str(event[1]), int(event[2]), int(event[3]))
            else:
                final = event
        process.join(TERMINATE_GRACE_S)
        if final is None and process.exitcode not in (0, None):
            log.error("작업 프로세스 비정상 종료", job_id=job_id, exitcode=process.exitcode)
        return final

    def _complete(self, job: Job, config: AppConfig, outcome: tuple[Any, ...] | None) -> None:
        if outcome is None:
            self._store.finish(job.id, STATUS_FAILED, error=ERROR_CRASHED)
            return
        kind = outcome[0]
        if kind == "cancelled":
            self._store.finish(job.id, STATUS_CANCELLED)
            return
        if kind == EVENT_FAILED:
            self._store.finish(job.id, STATUS_FAILED, error=str(outcome[1]))
            return
        if kind == EVENT_DONE:
            self._store.finish(job.id, STATUS_DONE, outputs=list(outcome[1]), result=outcome[2])
            if config.web.delete_input_after_done:
                self._store.input_path(job).unlink(missing_ok=True)
                self._store.mark_input_deleted(job.id)
            return
        self._store.finish(job.id, STATUS_FAILED, error=ERROR_CRASHED)

    def _sweep_uploads(self) -> None:
        """작업으로 등록하지 않고 남은 업로드 조각을 지운다."""
        now = time.time()
        if now - self._last_sweep < UPLOAD_SWEEP_INTERVAL_S:
            return
        self._last_sweep = now
        uploads = self._data_dir / UPLOADS_DIR
        if not uploads.is_dir():
            return
        for path in uploads.iterdir():
            try:
                if path.is_file() and now - path.stat().st_mtime > STALE_UPLOAD_S:
                    path.unlink()
                    log.info("오래된 업로드 삭제", path=str(path))
            except OSError as e:
                log.warning("업로드 정리 실패", path=str(path), error=str(e))


def _terminate(process: ProcessLike) -> None:
    """자식 프로세스를 끝낸다 (응답이 없으면 kill)."""
    process.terminate()
    process.join(TERMINATE_GRACE_S)
    if process.is_alive():
        process.kill()
        process.join(TERMINATE_GRACE_S)
