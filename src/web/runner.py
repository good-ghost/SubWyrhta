"""작업 자식 프로세스: 파이프라인 하나를 실행하고 진행·결과를 큐로 부모에게 보낸다.

작업마다 새 프로세스(spawn)에서 돈다. 끝나면 프로세스가 종료되어 모델이 쓴 RAM·VRAM 이 모두 돌아가고,
실행 중 취소는 부모가 프로세스를 끝내는 것으로 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.config import AppConfig
from src.core.exceptions import SubWyrhtaError
from src.core.logging import setup_logging
from src.pipeline import PipelineOptions, PipelineResult, TranscriptionPipeline

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

log = structlog.get_logger(__name__)

# 큐 메시지 종류 (튜플 첫 칸)
EVENT_PROGRESS = "progress"
EVENT_DONE = "done"
EVENT_FAILED = "failed"


@dataclass(frozen=True)
class JobPayload:
    """자식 프로세스에 넘기는 작업 정보 (pickle 가능한 값만).

    Attributes:
        job_id: 작업 번호 (로그에 남긴다).
        input_path: 입력 파일.
        output_dir: 출력 폴더.
        checkpoint_path: 체크포인트 파일.
        config: 작업 옵션을 적용한 설정 (``dataclasses.asdict``).
        formats: 출력 포맷 목록.
        resume: 체크포인트에서 재개한다 (다시 시도한 작업).
        log_file: JSON 로그 파일.
        log_level: 로그 레벨.
        work_dir: 단계 결과 폴더. 빈 값이면 남기지 않는다.
        audio_track: 오디오 트랙 (0-based). None 이면 자동 (언어에 맞는 트랙).
    """

    job_id: int
    input_path: str
    output_dir: str
    checkpoint_path: str
    config: dict[str, Any]
    formats: list[str]
    resume: bool
    log_file: str
    log_level: str
    # 기본값: 이 필드가 없던 서버(재시작 전)가 보낸 값을 풀어도 속성이 있도록
    work_dir: str = ""
    audio_track: int | None = None


def run_job_process(payload: JobPayload, events: Queue[tuple[Any, ...]]) -> None:
    """자식 프로세스 진입점.

    큐로 ``("progress", 단계, 끝낸 수, 전체 수)``를 보내고, 마지막에 ``("done", 출력 파일 이름 목록,
    요약)`` 또는 ``("failed", 오류 문구)``를 보낸다.

    Args:
        payload: 작업 정보.
        events: 부모에게 보낼 큐.
    """
    setup_logging(level=payload.log_level, json_log_file=payload.log_file)
    structlog.contextvars.bind_contextvars(job_id=payload.job_id)

    config = AppConfig.from_dict(payload.config)
    config.output.dir = payload.output_dir

    def progress(stage: str, done: int, total: int) -> None:
        events.put((EVENT_PROGRESS, stage, done, total))

    options = PipelineOptions(
        formats=payload.formats,
        resume=payload.resume,
        checkpoint_path=payload.checkpoint_path,
        keep_stages=bool(payload.work_dir),
        work_dir=payload.work_dir or None,
        audio_track=payload.audio_track,
    )
    try:
        with TranscriptionPipeline(config, progress=progress) as pipeline:
            outcome = pipeline.run(payload.input_path, options)
    except SubWyrhtaError as e:
        log.error("작업 실패", error=str(e))
        events.put((EVENT_FAILED, str(e)))
        return
    except Exception as e:
        # 라이브러리(torch·whisperx 등) 오류도 작업 실패로 알린다. 원인은 로그에 스택과 함께 남긴다
        log.exception("작업 실패 (예상하지 못한 오류)")
        events.put((EVENT_FAILED, f"{type(e).__name__}: {e}"))
        return
    events.put((EVENT_DONE, [Path(p).name for p in outcome.outputs], summarize(outcome)))


def summarize(outcome: PipelineResult) -> dict[str, Any]:
    """화면에 보일 결과 요약."""
    result = outcome.result
    summary: dict[str, Any] = {
        "language": result.metadata.language,
        "duration_s": round(result.metadata.duration_s, 3),
        "model": result.metadata.model,
        "segments": result.segment_count,
        "num_speakers": outcome.num_speakers,
        "quality": None,
    }
    report = outcome.report
    if report is not None:
        summary["quality"] = {
            "grade": report.quality_summary.grade,
            "avg_confidence": round(report.quality_summary.avg_confidence, 4),
            "low_confidence": report.quality_summary.low_confidence_count,
            "errors": report.error_count,
            "warnings": report.warning_count,
            "auto_fixed": report.auto_fixed_count,
        }
    return summary
