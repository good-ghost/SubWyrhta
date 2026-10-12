"""자막 세그먼트 데이터 모델."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models.word import WordSegment


@dataclass
class TranscriptionSegment:
    """자막 단위 세그먼트.

    Attributes:
        index: 자막 인덱스 (1-based).
        text: 자막 텍스트.
        start: 세그먼트 시작 시간(초).
        end: 세그먼트 종료 시간(초).
        confidence: 세그먼트 평균 신뢰도 (0.0 ~ 1.0).
        speaker: 화자 ID 또는 ID 목록. None이면 화자 미지정.
        words: 단어 단위 타임스탬프 목록.
    """

    index: int
    text: str
    start: float
    end: float
    confidence: float = 0.0
    speaker: str | list[str] | None = None
    words: list[WordSegment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """세그먼트 지속 시간(초)."""
        return self.end - self.start

    @property
    def char_count(self) -> int:
        """텍스트 글자 수 (공백 포함)."""
        return len(self.text)

    @property
    def cps(self) -> float:
        """초당 글자 수 (Characters Per Second)."""
        if self.duration <= 0:
            return 0.0
        return self.char_count / self.duration

    def to_dict(self, include_words: bool = True) -> dict[str, Any]:
        """딕셔너리로 변환한다.

        Args:
            include_words: True이면 words 목록을 포함한다.
        """
        result: dict[str, Any] = {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "confidence": round(self.confidence, 4),
        }
        if self.speaker is not None:
            result["speaker"] = self.speaker
        if include_words and self.words:
            result["words"] = [w.to_dict() for w in self.words]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptionSegment:
        """딕셔너리에서 인스턴스를 생성한다."""
        words = [WordSegment.from_dict(w) for w in data.get("words", [])]
        return cls(
            index=int(data["index"]),
            text=data["text"],
            start=float(data["start"]),
            end=float(data["end"]),
            confidence=float(data.get("confidence", 0.0)),
            speaker=data.get("speaker"),
            words=words,
        )
