"""세그멘테이션 전략 추상 기반 클래스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.core.config import SegmentationConfig
    from src.models.segment import TranscriptionSegment
    from src.models.word import WordSegment

log = structlog.get_logger(__name__)


class SegmentationStrategy(ABC):
    """세그멘테이션 전략 추상 기반 클래스.

    모든 세그멘테이션 모드는 이 ABC를 상속한다.
    """

    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """세그멘테이션 모드 이름 (예: 'basic', 'advanced')."""

    @abstractmethod
    def segment(
        self,
        words: list[WordSegment],
        language: str = "en",
    ) -> list[TranscriptionSegment]:
        """단어 목록을 자막 세그먼트로 분할한다.

        Args:
            words: 단어 단위 타임스탬프 목록.
            language: 언어 코드 (분할 규칙에 영향).

        Returns:
            분할된 TranscriptionSegment 목록.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode_name!r})"
