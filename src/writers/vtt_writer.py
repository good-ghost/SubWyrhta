"""WebVTT 자막 포맷 출력."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.core.exceptions import WriterError
from src.writers.base import SubtitleWriter

if TYPE_CHECKING:
    from src.models.result import TranscriptionResult

log = structlog.get_logger(__name__)


def _format_vtt_timestamp(seconds: float) -> str:
    """초를 WebVTT 타임스탬프 형식으로 변환한다.

    Returns:
        'HH:MM:SS.mmm' 형식 문자열.
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = round((seconds % 1) * 1000)
    if millis >= 1000:
        millis = 0
        secs += 1
        if secs >= 60:
            secs = 0
            minutes += 1
            if minutes >= 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class VTTWriter(SubtitleWriter):
    """WebVTT 포맷 자막 출력."""

    @property
    def format_name(self) -> str:
        return "vtt"

    @property
    def file_extension(self) -> str:
        return "vtt"

    def write(self, result: TranscriptionResult, output_path: str | Path) -> Path:
        """결과를 VTT 파일로 출력한다.

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
            lines: list[str] = ["WEBVTT", ""]

            for seg in result.segments:
                start = _format_vtt_timestamp(seg.start)
                end = _format_vtt_timestamp(seg.end)
                lines.append(f"{start} --> {end}")
                lines.append(seg.text)
                lines.append("")

            content = "\n".join(lines)
            path.write_text(content, encoding="utf-8")

            log.info("VTT 파일 저장 완료", path=str(path), segments=len(result.segments))
            return path

        except OSError as e:
            raise WriterError(f"VTT 파일 저장 실패: {path} ({e})") from e
