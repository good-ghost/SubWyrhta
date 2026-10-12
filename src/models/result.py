"""전체 인식 결과 데이터 모델."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.segment import TranscriptionSegment


@dataclass
class QualitySummary:
    """품질 요약 정보.

    Attributes:
        avg_confidence: 전체 평균 신뢰도.
        low_confidence_count: 저신뢰도 세그먼트 수.
        estimated_wer: 추정 Word Error Rate.
        grade: 품질 등급 (A/B/C/D/F).
    """

    avg_confidence: float = 0.0
    low_confidence_count: int = 0
    estimated_wer: float = 0.0
    grade: str = ""

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환한다."""
        return {
            "avg_confidence": round(self.avg_confidence, 4),
            "low_confidence_count": self.low_confidence_count,
            "estimated_wer": round(self.estimated_wer, 4),
            "grade": self.grade,
        }


# 예전 이름(BySTT) 시절 JSON 의 버전 키
_LEGACY_VERSION_KEY = "bystt_version"


@dataclass
class Metadata:
    """인식 메타데이터.

    Attributes:
        source_file: 원본 파일 경로.
        duration_s: 오디오 전체 길이(초).
        language: 인식된/지정된 언어 코드.
        model: 사용된 STT 모델 (예: 'whisperx/large-v3').
        created_at: 생성 시각 (ISO 8601).
        subwyrhta_version: SubWyrhta 버전.
    """

    source_file: str = ""
    duration_s: float = 0.0
    language: str = ""
    model: str = ""
    created_at: str = ""
    subwyrhta_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환한다."""
        return {
            "source_file": self.source_file,
            "duration_s": round(self.duration_s, 3),
            "language": self.language,
            "model": self.model,
            "created_at": self.created_at,
            "subwyrhta_version": self.subwyrhta_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metadata:
        """딕셔너리에서 인스턴스를 생성한다."""
        return cls(
            source_file=data.get("source_file", ""),
            duration_s=float(data.get("duration_s", 0.0)),
            language=data.get("language", ""),
            model=data.get("model", ""),
            created_at=data.get("created_at", ""),
            # 이름을 바꾸기 전(BySTT)에 만든 JSON 은 bystt_version 에 담겨 있다
            subwyrhta_version=data.get("subwyrhta_version", data.get(_LEGACY_VERSION_KEY, "")),
        )


@dataclass
class TranscriptionResult:
    """전체 인식 결과.

    metadata, segments, quality를 포함하며 JSON/SRT 등으로 저장할 수 있다.

    Attributes:
        metadata: 인식 메타데이터.
        segments: 자막 세그먼트 목록.
        quality: 품질 요약 정보.
    """

    metadata: Metadata = field(default_factory=Metadata)
    segments: list[TranscriptionSegment] = field(default_factory=list)
    quality: QualitySummary = field(default_factory=QualitySummary)

    @property
    def segment_count(self) -> int:
        """세그먼트 수."""
        return len(self.segments)

    @property
    def total_duration(self) -> float:
        """전체 자막이 커버하는 시간(초). 마지막 세그먼트 end - 첫 세그먼트 start."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end - self.segments[0].start

    def reindex(self) -> None:
        """세그먼트 인덱스를 1부터 순차적으로 재부여한다."""
        for i, seg in enumerate(self.segments, start=1):
            seg.index = i

    def to_dict(self, include_words: bool = True, include_quality: bool = True) -> dict[str, Any]:
        """전체 결과를 딕셔너리로 변환한다.

        Args:
            include_words: True이면 단어 타임스탬프를 포함한다.
            include_quality: True이면 품질 요약을 포함한다.
        """
        result: dict[str, Any] = {
            "metadata": self.metadata.to_dict(),
            "segments": [s.to_dict(include_words=include_words) for s in self.segments],
        }
        if include_quality and self.quality.grade:
            result["quality"] = self.quality.to_dict()
        return result

    def save(self, path: str | Path, include_words: bool = True) -> Path:
        """결과를 JSON 파일로 저장한다.

        Args:
            path: 저장 경로.
            include_words: True이면 단어 타임스탬프를 포함한다.

        Returns:
            저장된 파일의 Path 객체.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict(include_words=include_words)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptionResult:
        """딕셔너리에서 인스턴스를 생성한다."""
        metadata = Metadata.from_dict(data.get("metadata", {}))
        segments = [TranscriptionSegment.from_dict(s) for s in data.get("segments", [])]
        quality_data = data.get("quality", {})
        quality = QualitySummary(
            avg_confidence=float(quality_data.get("avg_confidence", 0.0)),
            low_confidence_count=int(quality_data.get("low_confidence_count", 0)),
            estimated_wer=float(quality_data.get("estimated_wer", 0.0)),
            grade=quality_data.get("grade", ""),
        )
        return cls(metadata=metadata, segments=segments, quality=quality)

    @classmethod
    def load(cls, path: str | Path) -> TranscriptionResult:
        """JSON 파일에서 결과를 로드한다.

        Args:
            path: JSON 파일 경로.

        Returns:
            TranscriptionResult 인스턴스.
        """
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
