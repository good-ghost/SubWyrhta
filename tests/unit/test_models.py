"""데이터 모델 단위 테스트."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.models.result import Metadata, QualitySummary, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment

if TYPE_CHECKING:
    from pathlib import Path


class TestWordSegment:
    """WordSegment 테스트."""

    def test_basic(self) -> None:
        w = WordSegment(word="Hello", start=0.5, end=0.9, confidence=0.98)
        assert w.word == "Hello"
        assert w.duration == pytest.approx(0.4)

    def test_to_dict(self) -> None:
        w = WordSegment(word="test", start=1.0, end=1.5, confidence=0.95)
        d = w.to_dict()
        assert d == {"word": "test", "start": 1.0, "end": 1.5, "confidence": 0.95}

    def test_from_dict(self) -> None:
        data = {"word": "hello", "start": 0.1, "end": 0.5, "confidence": 0.9}
        w = WordSegment.from_dict(data)
        assert w.word == "hello"
        assert w.confidence == 0.9

    def test_from_dict_no_confidence(self) -> None:
        data = {"word": "test", "start": 0.0, "end": 0.5}
        w = WordSegment.from_dict(data)
        assert w.confidence == 0.0

    def test_roundtrip(self) -> None:
        w = WordSegment(word="world", start=1.234, end=2.567, confidence=0.876)
        w2 = WordSegment.from_dict(w.to_dict())
        assert w2.word == w.word
        assert w2.start == pytest.approx(w.start, abs=0.001)
        assert w2.end == pytest.approx(w.end, abs=0.001)

    def test_default_confidence(self) -> None:
        w = WordSegment(word="x", start=0.0, end=1.0)
        assert w.confidence == 0.0


class TestTranscriptionSegment:
    """TranscriptionSegment 테스트."""

    def test_basic(self) -> None:
        seg = TranscriptionSegment(index=1, text="Hello world", start=0.5, end=1.5)
        assert seg.duration == pytest.approx(1.0)
        assert seg.char_count == 11
        assert seg.cps == pytest.approx(11.0)

    def test_cps_zero_duration(self) -> None:
        seg = TranscriptionSegment(index=1, text="Test", start=1.0, end=1.0)
        assert seg.cps == 0.0

    def test_speaker_single(self) -> None:
        seg = TranscriptionSegment(index=1, text="Hi", start=0.0, end=1.0, speaker="SPEAKER_00")
        d = seg.to_dict()
        assert d["speaker"] == "SPEAKER_00"

    def test_speaker_multi(self) -> None:
        seg = TranscriptionSegment(
            index=1,
            text="A / B",
            start=0.0,
            end=1.0,
            speaker=["SPEAKER_00", "SPEAKER_01"],
        )
        d = seg.to_dict()
        assert d["speaker"] == ["SPEAKER_00", "SPEAKER_01"]

    def test_speaker_none(self) -> None:
        seg = TranscriptionSegment(index=1, text="Hi", start=0.0, end=1.0)
        d = seg.to_dict()
        assert "speaker" not in d

    def test_to_dict_with_words(self) -> None:
        words = [WordSegment(word="Hello", start=0.5, end=0.9, confidence=0.98)]
        seg = TranscriptionSegment(index=1, text="Hello", start=0.5, end=0.9, words=words)
        d = seg.to_dict(include_words=True)
        assert len(d["words"]) == 1

    def test_to_dict_without_words(self) -> None:
        words = [WordSegment(word="Hello", start=0.5, end=0.9, confidence=0.98)]
        seg = TranscriptionSegment(index=1, text="Hello", start=0.5, end=0.9, words=words)
        d = seg.to_dict(include_words=False)
        assert "words" not in d

    def test_roundtrip(self) -> None:
        words = [
            WordSegment(word="Hello", start=0.5, end=0.9, confidence=0.98),
            WordSegment(word="world", start=1.0, end=1.4, confidence=0.96),
        ]
        seg = TranscriptionSegment(
            index=1,
            text="Hello world",
            start=0.5,
            end=1.44,
            confidence=0.97,
            speaker="SPEAKER_00",
            words=words,
        )
        seg2 = TranscriptionSegment.from_dict(seg.to_dict())
        assert seg2.text == seg.text
        assert seg2.speaker == seg.speaker
        assert len(seg2.words) == 2


class TestTranscriptionResult:
    """TranscriptionResult 테스트."""

    @pytest.fixture()
    def sample_result(self) -> TranscriptionResult:
        return TranscriptionResult(
            metadata=Metadata(
                source_file="video.mp4",
                duration_s=60.0,
                language="en",
                model="whisperx/large-v3",
                created_at="2025-01-01T00:00:00Z",
                subwyrhta_version="0.1.0",
            ),
            segments=[
                TranscriptionSegment(index=1, text="First", start=0.0, end=1.0, confidence=0.9),
                TranscriptionSegment(index=2, text="Second", start=1.5, end=3.0, confidence=0.8),
            ],
        )

    def test_segment_count(self, sample_result: TranscriptionResult) -> None:
        assert sample_result.segment_count == 2

    def test_total_duration(self, sample_result: TranscriptionResult) -> None:
        assert sample_result.total_duration == pytest.approx(3.0)

    def test_total_duration_empty(self) -> None:
        result = TranscriptionResult()
        assert result.total_duration == 0.0

    def test_reindex(self, sample_result: TranscriptionResult) -> None:
        sample_result.segments.insert(
            0,
            TranscriptionSegment(
                index=99,
                text="Zero",
                start=0.0,
                end=0.1,
            ),
        )
        sample_result.reindex()
        assert [s.index for s in sample_result.segments] == [1, 2, 3]

    def test_to_dict(self, sample_result: TranscriptionResult) -> None:
        d = sample_result.to_dict()
        assert d["metadata"]["source_file"] == "video.mp4"
        assert len(d["segments"]) == 2
        assert "quality" not in d  # grade가 빈 문자열이면 포함 안 됨

    def test_to_dict_with_quality(self, sample_result: TranscriptionResult) -> None:
        sample_result.quality = QualitySummary(
            avg_confidence=0.95,
            low_confidence_count=1,
            estimated_wer=0.03,
            grade="A",
        )
        d = sample_result.to_dict(include_quality=True)
        assert d["quality"]["grade"] == "A"

    def test_save_load(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        out = sample_result.save(tmp_path / "result.json")
        loaded = TranscriptionResult.load(out)
        assert loaded.segment_count == 2
        assert loaded.metadata.source_file == "video.mp4"
        assert loaded.metadata.language == "en"
        assert loaded.segments[0].text == "First"

    def test_save_creates_dirs(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        out = sample_result.save(tmp_path / "sub" / "dir" / "result.json")
        assert out.exists()

    def test_metadata_roundtrip(self) -> None:
        m = Metadata(
            source_file="test.mp4",
            duration_s=120.5,
            language="ko",
            model="faster-whisper/small",
            created_at="2025-06-01T12:00:00Z",
            subwyrhta_version="0.1.0",
        )
        m2 = Metadata.from_dict(m.to_dict())
        assert m2.source_file == m.source_file
        assert m2.duration_s == m.duration_s
        assert m2.language == m.language
        assert m2.subwyrhta_version == m.subwyrhta_version

    def test_metadata_reads_legacy_version_key(self) -> None:
        """이름을 바꾸기 전(BySTT)에 만든 JSON 의 bystt_version 도 읽는다."""
        m = Metadata.from_dict({"source_file": "old.mp4", "bystt_version": "0.0.9"})
        assert m.subwyrhta_version == "0.0.9"
        assert m.to_dict()["subwyrhta_version"] == "0.0.9"
        assert "bystt_version" not in m.to_dict()

    def test_metadata_new_version_key_wins(self) -> None:
        m = Metadata.from_dict({"subwyrhta_version": "0.2.0", "bystt_version": "0.0.9"})
        assert m.subwyrhta_version == "0.2.0"
