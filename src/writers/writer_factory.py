"""자막 Writer 팩토리."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.core.exceptions import ConfigError

if TYPE_CHECKING:
    from src.writers.base import SubtitleWriter

log = structlog.get_logger(__name__)

# 포맷 이름 → (모듈 경로, 클래스 이름) 매핑
_WRITER_REGISTRY: dict[str, tuple[str, str]] = {
    "srt": ("src.writers.srt_writer", "SRTWriter"),
    "ass": ("src.writers.ass_writer", "ASSWriter"),
    "vtt": ("src.writers.vtt_writer", "VTTWriter"),
    "json": ("src.writers.json_writer", "JSONWriter"),
}


def get_available_formats() -> list[str]:
    """사용 가능한 출력 포맷 목록을 반환한다."""
    return sorted(_WRITER_REGISTRY.keys())


def create_writer(format_name: str) -> SubtitleWriter:
    """포맷 이름에 따라 SubtitleWriter 인스턴스를 생성한다.

    Args:
        format_name: 출력 포맷 (srt, ass, vtt, json).

    Returns:
        SubtitleWriter 구현체 인스턴스.

    Raises:
        ConfigError: 알 수 없는 포맷일 때.
    """
    fmt = format_name.lower()

    if fmt not in _WRITER_REGISTRY:
        available = ", ".join(get_available_formats())
        raise ConfigError(f"알 수 없는 출력 포맷: {format_name!r}. 사용 가능: {available}")

    module_path, class_name = _WRITER_REGISTRY[fmt]

    try:
        import importlib

        module = importlib.import_module(module_path)
        writer_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ConfigError(f"Writer {format_name!r}을(를) 불러올 수 없습니다: {e}") from e

    log.info("Writer 생성", format=fmt)
    return writer_class()  # type: ignore[no-any-return]
