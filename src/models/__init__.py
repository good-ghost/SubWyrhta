"""데이터 모델: WordSegment, TranscriptionSegment, TranscriptionResult."""

from src.models.result import Metadata, QualitySummary, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment

__all__ = [
    "Metadata",
    "QualitySummary",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WordSegment",
]
