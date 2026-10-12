"""SRT 자막 포맷 출력."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.core.exceptions import WriterError
from src.writers.base import SubtitleWriter

if TYPE_CHECKING:
    from src.models.result import TranscriptionResult
    from src.models.segment import TranscriptionSegment

log = structlog.get_logger(__name__)


def _format_timestamp(seconds: float) -> str:
    """초를 SRT 타임스탬프 형식으로 변환한다.

    Args:
        seconds: 시간(초).

    Returns:
        'HH:MM:SS,mmm' 형식 문자열.
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = round((seconds % 1) * 1000)
    # 밀리초 반올림으로 1000이 되면 초 올림
    if millis >= 1000:
        millis = 0
        secs += 1
        if secs >= 60:
            secs = 0
            minutes += 1
            if minutes >= 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segment_to_srt_block(segment: TranscriptionSegment) -> str:
    """TranscriptionSegment를 SRT 블록 문자열로 변환한다."""
    start = _format_timestamp(segment.start)
    end = _format_timestamp(segment.end)
    return f"{segment.index}\n{start} --> {end}\n{segment.text}\n"


class SRTWriter(SubtitleWriter):
    """SRT 포맷 자막 출력."""

    @property
    def format_name(self) -> str:
        return "srt"

    @property
    def file_extension(self) -> str:
        return "srt"

    def write(self, result: TranscriptionResult, output_path: str | Path) -> Path:
        """결과를 SRT 파일로 출력한다.

        Args:
            result: TranscriptionResult 인스턴스.
            output_path: 출력 파일 경로.

        Returns:
            저장된 파일의 Path.

        Raises:
            WriterError: 출력 실패 시.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            blocks = [_segment_to_srt_block(seg) for seg in result.segments]
            content = "\n".join(blocks)

            path.write_text(content, encoding="utf-8")

            log.info(
                "SRT 파일 저장 완료",
                path=str(path),
                segments=len(result.segments),
            )
            return path

        except OSError as e:
            raise WriterError(f"SRT 파일 저장 실패: {path} ({e})") from e
