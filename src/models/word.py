"""단어 단위 타임스탬프 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WordSegment:
    """단어 단위 start/end/confidence 정보.

    Attributes:
        word: 인식된 단어 텍스트.
        start: 단어 시작 시간(초).
        end: 단어 종료 시간(초).
        confidence: 인식 신뢰도 (0.0 ~ 1.0).
    """

    word: str
    start: float
    end: float
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        """단어 지속 시간(초)."""
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환한다."""
        return {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WordSegment:
        """딕셔너리에서 인스턴스를 생성한다."""
        return cls(
            word=data["word"],
            start=float(data["start"]),
            end=float(data["end"]),
            confidence=float(data.get("confidence", 0.0)),
        )
