"""자막 출력 추상 기반 클래스 및 출력 파일명 유틸리티."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.models.result import TranscriptionResult

log = structlog.get_logger(__name__)


def build_output_path(
    source_file: str,
    language: str,
    ext: str,
    output_dir: str = "./output",
    output_name: str | None = None,
    diarized: bool = False,
) -> Path:
    """출력 파일 경로를 생성한다.

    규칙:
        - 기본: ``{원본파일명}.{언어코드}.{ext}``
        - 화자 분리: ``{원본파일명}.{언어코드}.diarized.{ext}``
        - 충돌 시: ``{원본파일명}.{언어코드}_{N}.{ext}``

    Args:
        source_file: 원본 파일 경로.
        language: 언어 코드.
        ext: 출력 확장자 (srt, ass, vtt, json).
        output_dir: 출력 디렉토리.
        output_name: 출력 파일명 직접 지정. None이면 규칙 적용.
        diarized: 화자 분리 여부.

    Returns:
        충돌 없는 출력 파일 Path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_name is not None:
        # 직접 지정 시 확장자 보장
        name = output_name if output_name.endswith(f".{ext}") else f"{output_name}.{ext}"
        return _resolve_conflict(out_dir / name)

    stem = Path(source_file).stem
    lang = language if language else "unknown"

    base_name = f"{stem}.{lang}.diarized.{ext}" if diarized else f"{stem}.{lang}.{ext}"

    return _resolve_conflict(out_dir / base_name)


def _resolve_conflict(path: Path) -> Path:
    """파일명 충돌 시 자동 넘버링한다.

    ``video.en.srt`` → ``video.en_2.srt`` → ``video.en_3.srt`` ...
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2

    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


class SubtitleWriter(ABC):
    """자막 출력 추상 기반 클래스.

    모든 자막 포맷 Writer는 이 ABC를 상속한다.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """포맷 이름 (예: 'srt', 'ass', 'vtt', 'json')."""

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """파일 확장자 (예: 'srt')."""

    @abstractmethod
    def write(self, result: TranscriptionResult, output_path: str | Path) -> Path:
        """결과를 자막 파일로 출력한다.

        Args:
            result: TranscriptionResult 인스턴스.
            output_path: 출력 파일 경로.

        Returns:
            저장된 파일의 Path.

        Raises:
            WriterError: 출력 실패 시.
        """

    def write_auto(
        self,
        result: TranscriptionResult,
        output_dir: str = "./output",
        output_name: str | None = None,
        diarized: bool = False,
    ) -> Path:
        """출력 파일명을 자동 생성하여 저장한다.

        Args:
            result: TranscriptionResult 인스턴스.
            output_dir: 출력 디렉토리.
            output_name: 출력 파일명 직접 지정.
            diarized: 화자 분리 여부.

        Returns:
            저장된 파일의 Path.
        """
        path = build_output_path(
            source_file=result.metadata.source_file,
            language=result.metadata.language,
            ext=self.file_extension,
            output_dir=output_dir,
            output_name=output_name,
            diarized=diarized,
        )
        return self.write(result, path)
