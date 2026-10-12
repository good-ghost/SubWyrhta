"""JSON 자막 포맷 출력.

단어 타임스탬프 + 신뢰도 + 화자 정보 + quality를 포함하는 원시 데이터 출력.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.core.exceptions import WriterError
from src.writers.base import SubtitleWriter

if TYPE_CHECKING:
    from src.models.result import TranscriptionResult

log = structlog.get_logger(__name__)


class JSONWriter(SubtitleWriter):
    """JSON 포맷 자막 출력.

    REQUIREMENT.md 8.2절의 JSON 스키마를 따른다.
    """

    def __init__(self, include_words: bool = True, include_quality: bool = True) -> None:
        self._include_words = include_words
        self._include_quality = include_quality

    @property
    def format_name(self) -> str:
        return "json"

    @property
    def file_extension(self) -> str:
        return "json"

    def write(self, result: TranscriptionResult, output_path: str | Path) -> Path:
        """결과를 JSON 파일로 출력한다.

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
            data = result.to_dict(
                include_words=self._include_words,
                include_quality=self._include_quality,
            )
            content = json.dumps(data, ensure_ascii=False, indent=2)
            path.write_text(content, encoding="utf-8")

            log.info("JSON 파일 저장 완료", path=str(path), segments=len(result.segments))
            return path

        except OSError as e:
            raise WriterError(f"JSON 파일 저장 실패: {path} ({e})") from e
