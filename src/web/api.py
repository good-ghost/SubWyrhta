"""웹 API: 상태·작업·업로드·다운로드·설정·로그·계정 (모두 로그인 필요)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from src.audio.extractor import SUPPORTED_EXTENSIONS
from src.engines.engine_factory import get_available_engines
from src.pipeline import STAGES, SUBWYRHTA_VERSION
from src.segmentation.llm_client import provider_readiness
from src.segmentation.segmentation_factory import get_available_modes
from src.web.auth import AccountError
from src.web.context import (
    Context,  # noqa: TC001 — FastAPI 가 실행 중에 의존성 타입을 읽는다
)
from src.web.job_options import (
    COMPUTE_TYPES,
    DEVICES,
    LANGUAGES,
    MODEL_SIZES,
    JobOptions,
    default_values,
)
from src.web.jobs import FINISHED, RETRYABLE, STATUS_QUEUED, STATUS_RUNNING, STATUSES, Job, JobError
from src.web.logs import read_log
from src.web.review import (
    MAX_TEXT_LENGTH,
    REPORT_SUFFIX,
    STATE_FILE,
    Cue,
    Note,
    ReviewError,
    attach_notes,
    backup_name,
    is_subtitle,
    mark_reviewed,
    parse_srt,
    read_report,
    read_reviewed,
    report_notes,
    save_text,
)
from src.web.settings_io import (
    SettingsError,
    editable_values,
    is_writable,
    save_settings,
    validate_values,
)
from src.web.worker import UPLOADS_DIR
from src.writers.writer_factory import get_available_formats

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api")

UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_NAME_LENGTH = 200
PREVIEW_BYTES = 64 * 1024
LOG_DEFAULT_LIMIT = 500
LOG_MAX_LIMIT = 2000
_UPLOAD_ID = re.compile(r"[0-9a-f]{32}")
_MB = 1024 * 1024
_JOB_ERRORS: dict[str, int] = {
    "job_not_found": 404,
    "file_not_found": 404,
    "job_not_retryable": 409,
    "job_running": 409,
    "input_deleted": 409,
    "job_not_cancellable": 409,
}


# ──────────────────────────────────────────────
# 응답 모델
# ──────────────────────────────────────────────


class ProgressOut(BaseModel):
    """진행 (전체를 모르면 total 0)."""

    done: int
    total: int


class JobOut(BaseModel):
    """작업."""

    id: int
    name: str
    size: int
    status: str
    stage: str | None
    progress: ProgressOut | None
    error: str | None
    outputs: list[str]
    result: dict[str, Any] | None
    options: dict[str, Any]
    attempts: int
    input_deleted: bool
    retryable: bool
    created_at: float
    started_at: float | None
    finished_at: float | None
    updated_at: float


class CurrentOut(BaseModel):
    """실행 중인 작업."""

    id: int
    name: str
    stage: str | None
    progress: ProgressOut | None
    started_at: float


class StatusOut(BaseModel):
    """서버·워커·큐·시스템 상태."""

    version: str
    started_at: float
    worker_alive: bool
    current: CurrentOut | None
    queue: dict[str, int]
    system_state: str
    system: dict[str, Any]
    hf_token: bool
    disk_free: int
    disk_total: int
    defaults: dict[str, Any]


class ChoicesOut(BaseModel):
    """선택지."""

    engines: list[str]
    model_sizes: list[str]
    compute_types: list[str]
    devices: list[str]
    languages: list[str]
    formats: list[str]
    modes: list[str]
    stages: list[str]


class OptionsOut(BaseModel):
    """새 작업 화면의 기본값·선택지·업로드 제한."""

    defaults: dict[str, Any]
    choices: ChoicesOut
    max_upload_mb: int
    extensions: list[str]


class UploadOut(BaseModel):
    """업로드한 파일."""

    id: str
    name: str
    size: int


class JobIn(BaseModel):
    """작업 등록 요청."""

    upload_id: str
    options: JobOptions


class CountOut(BaseModel):
    """처리한 개수."""

    count: int


class PreviewOut(BaseModel):
    """출력 파일 앞부분."""

    name: str
    text: str
    truncated: bool


class ReviewNoteOut(BaseModel):
    """검수 항목 하나 (종류, 메모)."""

    type: str
    note: str


class ReviewRowOut(BaseModel):
    """검수 화면의 자막 한 줄."""

    index: int
    start: float
    end: float
    text: str
    notes: list[ReviewNoteOut]
    # 사용자가 저장해 검수를 마친 자막
    reviewed: bool


class ReviewOut(BaseModel):
    """자막 검수 자료: 고를 수 있는 SRT 이름, 지금 SRT 의 자막과 검수 항목."""

    name: str
    subtitles: list[str]
    report: str | None
    llm_error: str
    edited: bool
    rows: list[ReviewRowOut]


class ReviewIn(BaseModel):
    """자막 하나의 글자 수정."""

    name: str = Field(max_length=MAX_NAME_LENGTH)
    index: int = Field(ge=0)
    # 줄바꿈·앞뒤 공백을 다듬기 전 길이라 여유를 둔다 (다듬은 뒤 길이는 review.clean_text 가 본다)
    text: str = Field(max_length=MAX_TEXT_LENGTH * 2)


class LlmReadinessOut(BaseModel):
    """LLM 프로바이더 준비 상태 (키 값은 내보내지 않는다)."""

    kind: str
    requirement: str
    ready: bool


class SettingsOut(BaseModel):
    """설정 화면 상태."""

    path: str
    writable: bool
    values: dict[str, Any]
    hf_token: bool
    # 프로바이더 → 구독 CLI 설치 여부·API 키(.env) 설정 여부
    llm: dict[str, LlmReadinessOut]
    choices: ChoicesOut


class SettingsIn(BaseModel):
    """설정 저장 요청."""

    values: dict[str, Any]


class LogEntryOut(BaseModel):
    """로그 항목."""

    time: str
    level: str
    logger: str
    job_id: int | None
    message: str


class LogChunkOut(BaseModel):
    """로그 항목과 이어 읽을 위치."""

    entries: list[LogEntryOut]
    cursor: str
    reset: bool


class AccountOut(BaseModel):
    """계정 정보 (비밀번호는 내보내지 않는다)."""

    username: str | None


class AccountIn(BaseModel):
    """계정 변경 요청. new_password 가 없으면 비밀번호는 그대로."""

    current_password: str
    username: str
    new_password: str | None = None


# ──────────────────────────────────────────────
# 상태·선택지
# ──────────────────────────────────────────────


def _choices() -> ChoicesOut:
    return ChoicesOut(
        engines=get_available_engines(),
        model_sizes=list(MODEL_SIZES),
        compute_types=list(COMPUTE_TYPES),
        devices=list(DEVICES),
        languages=list(LANGUAGES),
        formats=get_available_formats(),
        modes=get_available_modes(),
        stages=list(STAGES),
    )


def _hf_token_set(context: Context) -> bool:
    return bool(context.config().diarization.hf_token or os.environ.get("HF_TOKEN"))


@router.get("/status")
def status(context: Context) -> StatusOut:
    """서버·워커·큐·시스템 상태."""
    config = context.config()
    current = context.worker.current
    current_out: CurrentOut | None = None
    if current is not None:
        job = context.store.get(current.job_id)
        current_out = CurrentOut(
            id=job.id,
            name=job.name,
            stage=job.stage,
            progress=_progress(job),
            started_at=current.started_at,
        )
    state, system, _ = context.system.snapshot()
    disk = shutil.disk_usage(context.data_dir)
    return StatusOut(
        version=SUBWYRHTA_VERSION,
        started_at=context.started_at,
        worker_alive=context.worker.alive,
        current=current_out,
        queue=context.store.counts(),
        system_state=state,
        system=system,
        hf_token=_hf_token_set(context),
        disk_free=disk.free,
        disk_total=disk.total,
        defaults={
            "engine": config.stt.model,
            "model_size": config.stt.model_size,
            "device": config.stt.device,
            "compute_type": config.stt.compute_type,
            "source_lang": config.stt.source_lang,
            "format": config.output.format,
            "mode": config.segmentation.mode,
        },
    )


@router.get("/options")
def options(context: Context) -> OptionsOut:
    """새 작업 화면의 기본값 (설정 파일 값)과 선택지."""
    config = context.config()
    try:
        defaults = JobOptions.defaults(config).model_dump()
    except ValidationError as e:
        # 설정 파일에 화면 선택지 밖의 값이 있다 (손으로 고친 파일). 화면이 다시 고르게 그대로 보낸다
        log.warning("설정 기본값이 작업 옵션 규칙에 맞지 않음", error=str(e))
        defaults = default_values(config)
    return OptionsOut(
        defaults=defaults,
        choices=_choices(),
        max_upload_mb=config.web.max_upload_mb,
        extensions=sorted(SUPPORTED_EXTENSIONS),
    )


# ──────────────────────────────────────────────
# 업로드·작업
# ──────────────────────────────────────────────


def _uploads_dir(context: Context) -> Path:
    path = context.data_dir / UPLOADS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clean_name(raw: str) -> str:
    """업로드 파일 이름 검증: 경로를 떼고, 지원 확장자만 받는다."""
    name = Path(raw.replace("\\", "/")).name.strip()
    if not name or name.startswith(".") or "\x00" in name or len(name) > MAX_NAME_LENGTH:
        raise HTTPException(status_code=400, detail="invalid_name")
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="unsupported_format")
    return name


@router.post("/uploads")
async def upload(
    request: Request,
    context: Context,
    name: Annotated[str, Query(min_length=1, max_length=MAX_NAME_LENGTH * 2)],
) -> UploadOut:
    """요청 본문(application/octet-stream)을 업로드 폴더에 저장한다. 작업 등록은 /api/jobs 로 한다."""
    clean = _clean_name(name)
    limit = context.config().web.max_upload_mb * _MB
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail="file_too_large")

    upload_id = uuid.uuid4().hex
    folder = _uploads_dir(context)
    part = folder / f"{upload_id}.part"
    size = 0
    try:
        with part.open("wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail="file_too_large")
                await asyncio.to_thread(f.write, chunk)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    if size == 0:
        part.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="empty_file")
    part.replace(folder / upload_id)
    (folder / f"{upload_id}.json").write_text(
        json.dumps({"name": clean, "size": size}), encoding="utf-8"
    )
    log.info("업로드 완료", upload_id=upload_id, name=clean, size=size)
    return UploadOut(id=upload_id, name=clean, size=size)


@router.post("/jobs")
def create_job(body: JobIn, context: Context) -> JobOut:
    """업로드한 파일로 작업을 등록한다 (대기열 끝에 들어간다)."""
    if not _UPLOAD_ID.fullmatch(body.upload_id):
        raise HTTPException(status_code=400, detail="upload_not_found")
    folder = _uploads_dir(context)
    data_file = folder / body.upload_id
    meta_file = folder / f"{body.upload_id}.json"
    if not data_file.is_file() or not meta_file.is_file():
        raise HTTPException(status_code=404, detail="upload_not_found")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    job = context.store.create(str(meta["name"]), data_file, body.options.model_dump())
    meta_file.unlink(missing_ok=True)
    context.worker.notify()
    return _job_out(job)


@router.get("/jobs")
def list_jobs(
    context: Context,
    statuses: Annotated[list[str] | None, Query(alias="status")] = None,
) -> list[JobOut]:
    """작업 목록 (최근 등록 순). status 를 주면 그 상태만."""
    if statuses and any(s not in STATUSES for s in statuses):
        raise HTTPException(status_code=400, detail="invalid_status")
    return [_job_out(job) for job in context.store.query(statuses)]


@router.get("/jobs/{job_id}")
def get_job(job_id: int, context: Context) -> JobOut:
    """작업 하나."""
    return _job_out(_job(context, job_id))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, context: Context) -> JobOut:
    """대기 중이면 바로, 실행 중이면 자식 프로세스를 끝내 취소한다."""
    job = _job(context, job_id)
    if job.status == STATUS_QUEUED and context.store.cancel_queued(job_id):
        return _job_out(context.store.get(job_id))
    if job.status == STATUS_RUNNING and context.worker.cancel(job_id):
        return _job_out(_wait_finished(context, job_id))
    raise HTTPException(status_code=409, detail="job_not_cancellable")


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: int, context: Context) -> JobOut:
    """실패·취소한 작업을 다시 대기열에 넣는다 (Pre-VAD 체크포인트가 있으면 이어서 한다)."""
    try:
        job = context.store.retry(job_id)
    except JobError as e:
        raise _job_http(e) from e
    context.worker.notify()
    return _job_out(job)


@router.post("/jobs/retry-failed")
def retry_failed(context: Context) -> CountOut:
    """실패한 작업을 모두 다시 대기열에 넣는다 (입력이 지워진 작업은 건너뛴다)."""
    count = 0
    for job in context.store.query(["failed"]):
        try:
            context.store.retry(job.id)
            count += 1
        except JobError as e:
            log.info("다시 시도할 수 없는 작업", job_id=job.id, reason=e.code)
    if count:
        context.worker.notify()
    return CountOut(count=count)


@router.post("/jobs/clear-finished")
def clear_finished(context: Context) -> CountOut:
    """완료·실패·취소한 작업과 그 파일을 모두 지운다."""
    jobs = context.store.query(sorted(FINISHED))
    for job in jobs:
        context.store.delete(job.id)
    return CountOut(count=len(jobs))


@router.delete("/jobs/{job_id}")
def delete_job(job_id: int, context: Context) -> CountOut:
    """작업과 그 파일을 지운다 (실행 중이면 409)."""
    try:
        context.store.delete(job_id)
    except JobError as e:
        raise _job_http(e) from e
    return CountOut(count=1)


@router.get("/jobs/{job_id}/files/{name}")
def download(job_id: int, name: str, context: Context) -> FileResponse:
    """출력 파일 내려받기."""
    try:
        path = context.store.output_file(job_id, name)
    except JobError as e:
        raise _job_http(e) from e
    return FileResponse(path, filename=name, media_type=_media_type(name))


@router.get("/jobs/{job_id}/preview/{name}")
def preview(job_id: int, name: str, context: Context) -> PreviewOut:
    """출력 파일 앞부분 (화면 미리 보기)."""
    try:
        path = context.store.output_file(job_id, name)
    except JobError as e:
        raise _job_http(e) from e
    with path.open("rb") as f:
        data = f.read(PREVIEW_BYTES + 1)
    return PreviewOut(
        name=name,
        text=data[:PREVIEW_BYTES].decode("utf-8", errors="replace"),
        truncated=len(data) > PREVIEW_BYTES,
    )


@router.get("/jobs/{job_id}/review")
def get_review(job_id: int, context: Context, name: str | None = None) -> ReviewOut:
    """자막 검수 자료: SRT 자막에 검수 리포트 항목을 붙인 목록.

    Args:
        job_id: 작업 번호.
        context: 요청 맥락.
        name: 검수할 SRT 출력 이름 (없으면 첫 SRT).
    """
    job = _job(context, job_id)
    subtitles = [output for output in job.outputs if is_subtitle(output)]
    chosen = name if name is not None else (subtitles[0] if subtitles else None)
    if chosen is None or chosen not in subtitles:
        raise HTTPException(status_code=404, detail="subtitle_not_found")
    try:
        cues = parse_srt(_output_path(context, job_id, chosen).read_text(encoding="utf-8"))
    except (ReviewError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=422, detail="srt_invalid") from e
    report_name = next((output for output in job.outputs if output.endswith(REPORT_SUFFIX)), None)
    report = read_report(_output_path(context, job_id, report_name) if report_name else None)
    attached = attach_notes(cues, report_notes(report))
    reviewed = read_reviewed(_review_state(context, job_id), chosen)
    return ReviewOut(
        name=chosen,
        subtitles=subtitles,
        report=report_name,
        llm_error=str(report.get("llm_error") or ""),
        edited=backup_name(chosen) in job.outputs,
        rows=[_review_row(cue, attached.get(cue.index, []), cue.index in reviewed) for cue in cues],
    )


@router.put("/jobs/{job_id}/review")
def put_review(job_id: int, body: ReviewIn, context: Context) -> ReviewRowOut:
    """자막 하나를 저장하고 검수 완료로 적는다 (글자가 그대로여도 검수 완료다).

    글자가 바뀌면 SRT 에 쓰고, 처음 고칠 때 원본을 ``*.orig.srt`` 로 남겨 출력 목록에 더한다.
    """
    job = _job(context, job_id)
    if not is_subtitle(body.name) or body.name not in job.outputs:
        raise HTTPException(status_code=404, detail="subtitle_not_found")
    try:
        cue, backed_up = save_text(_output_path(context, job_id, body.name), body.index, body.text)
        if backed_up:
            context.store.add_output(job_id, backup_name(body.name))
        mark_reviewed(_review_state(context, job_id), body.name, cue.index)
    except ReviewError as e:
        status = 404 if e.code == "cue_not_found" else 422
        raise HTTPException(status_code=status, detail=e.code) from e
    except JobError as e:
        raise _job_http(e) from e
    # 검수 항목은 화면이 이미 갖고 있다 (글자와 검수 상태만 바뀐다)
    return _review_row(cue, [], reviewed=True)


def _review_state(context: Context, job_id: int) -> Path:
    return context.store.job_dir(job_id) / STATE_FILE


def _output_path(context: Context, job_id: int, name: str) -> Path:
    try:
        return context.store.output_file(job_id, name)
    except JobError as e:
        raise _job_http(e) from e


def _review_row(cue: Cue, notes: list[Note], reviewed: bool) -> ReviewRowOut:
    return ReviewRowOut(
        index=cue.index,
        start=cue.start,
        end=cue.end,
        text=cue.text,
        notes=[ReviewNoteOut(type=note.type, note=note.note) for note in notes],
        reviewed=reviewed,
    )


def _job(context: Context, job_id: int) -> Job:
    try:
        return context.store.get(job_id)
    except JobError as e:
        raise _job_http(e) from e


def _wait_finished(context: Context, job_id: int, timeout_s: float = 15.0) -> Job:
    """취소를 요청한 작업이 끝날 때까지 잠깐 기다린다 (워커가 프로세스를 끝내고 기록한다)."""
    deadline = time.monotonic() + timeout_s
    job = context.store.get(job_id)
    while job.status == STATUS_RUNNING and time.monotonic() < deadline:
        time.sleep(0.2)
        job = context.store.get(job_id)
    return job


def _job_http(error: JobError) -> HTTPException:
    return HTTPException(status_code=_JOB_ERRORS.get(error.code, 400), detail=error.code)


def _progress(job: Job) -> ProgressOut | None:
    if job.status != STATUS_RUNNING or job.stage is None:
        return None
    return ProgressOut(done=job.progress_done, total=job.progress_total)


def _job_out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        name=job.name,
        size=job.size,
        status=job.status,
        stage=job.stage,
        progress=_progress(job),
        error=job.error,
        outputs=job.outputs,
        result=job.result,
        options=job.options,
        attempts=job.attempts,
        input_deleted=job.input_deleted,
        retryable=job.status in RETRYABLE and not job.input_deleted,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        updated_at=job.updated_at,
    )


def _media_type(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {
        ".srt": "application/x-subrip",
        ".vtt": "text/vtt",
        ".ass": "text/x-ssa",
        ".json": "application/json",
    }.get(suffix, "application/octet-stream")


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────


def _settings_out(context: Context) -> SettingsOut:
    return SettingsOut(
        path=str(context.config_path),
        writable=is_writable(context.config_path),
        values=editable_values(context.config()),
        hf_token=_hf_token_set(context),
        llm={
            name: LlmReadinessOut(kind=item.kind, requirement=item.requirement, ready=item.ready)
            for name, item in provider_readiness().items()
        },
        choices=_choices(),
    )


@router.get("/settings")
def get_settings(context: Context) -> SettingsOut:
    """설정 화면 값 (비밀 값은 내보내지 않는다)."""
    return _settings_out(context)


@router.put("/settings")
def put_settings(body: SettingsIn, context: Context) -> SettingsOut:
    """설정을 검증해 config.yaml 에 저장한다. 다음 작업부터 반영된다."""
    try:
        settings = validate_values(body.values)
        save_settings(context.config_path, settings)
    except SettingsError as e:
        status_code = 409 if e.code == "read_only" else 400
        raise HTTPException(
            status_code=status_code, detail={"code": e.code, "errors": e.errors}
        ) from e
    return _settings_out(context)


# ──────────────────────────────────────────────
# 로그·계정
# ──────────────────────────────────────────────


@router.get("/logs")
def logs(
    context: Context,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    after: str | None = None,
    limit: Annotated[int, Query(ge=1, le=LOG_MAX_LIMIT)] = LOG_DEFAULT_LIMIT,
) -> LogChunkOut:
    """로그 파일의 최근 항목. after(이전 cursor)를 주면 그 뒤 항목만."""
    chunk = read_log(context.log_file, cursor=after, level=level, limit=limit)
    return LogChunkOut(
        entries=[LogEntryOut(**vars(entry)) for entry in chunk.entries],
        cursor=chunk.cursor,
        reset=chunk.reset,
    )


@router.get("/account")
def get_account(context: Context) -> AccountOut:
    """지금 사용자 이름."""
    return AccountOut(username=context.accounts.username())


@router.put("/account")
def put_account(body: AccountIn, context: Context) -> AccountOut:
    """사용자 이름·비밀번호를 바꾼다. 틀린 지금 비밀번호는 403, 규칙 위반은 400."""
    try:
        context.accounts.change(
            body.current_password, username=body.username, new_password=body.new_password or None
        )
    except AccountError as e:
        raise HTTPException(
            status_code=403 if e.code == "wrong_password" else 400, detail=e.code
        ) from e
    return AccountOut(username=context.accounts.username())
