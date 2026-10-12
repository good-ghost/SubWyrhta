"""VTT/JSON Writer 단위 테스트."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.models.result import Metadata, QualitySummary, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.writers.json_writer import JSONWriter
from src.writers.vtt_writer import VTTWriter, _format_vtt_timestamp
from src.writers.writer_factory import create_writer

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def sample_result() -> TranscriptionResult:
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
            TranscriptionSegment(
                1,
                "Hello world.",
                0.5,
                1.5,
                0.97,
                speaker="SPEAKER_00",
                words=[WordSegment("Hello", 0.5, 0.9, 0.98), WordSegment("world.", 1.0, 1.4, 0.96)],
            ),
            TranscriptionSegment(2, "Second line.", 2.0, 3.5, 0.85),
        ],
        quality=QualitySummary(
            avg_confidence=0.91, low_confidence_count=0, estimated_wer=0.09, grade="A"
        ),
    )


class TestVTTTimestamp:
    def test_zero(self) -> None:
        assert _format_vtt_timestamp(0.0) == "00:00:00.000"

    def test_basic(self) -> None:
        assert _format_vtt_timestamp(61.5) == "00:01:01.500"

    def test_negative(self) -> None:
        assert _format_vtt_timestamp(-1.0) == "00:00:00.000"


class TestVTTWriter:
    def test_properties(self) -> None:
        w = VTTWriter()
        assert w.format_name == "vtt"
        assert w.file_extension == "vtt"

    def test_factory(self) -> None:
        assert isinstance(create_writer("vtt"), VTTWriter)

    def test_write(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        w = VTTWriter()
        out = w.write(sample_result, tmp_path / "test.vtt")
        content = out.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT")
        assert "00:00:00.500 --> 00:00:01.500" in content
        assert "Hello world." in content

    def test_write_empty(self, tmp_path: Path) -> None:
        w = VTTWriter()
        out = w.write(TranscriptionResult(), tmp_path / "empty.vtt")
        assert out.read_text().startswith("WEBVTT")

    def test_write_auto(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        w = VTTWriter()
        out = w.write_auto(sample_result, output_dir=str(tmp_path))
        assert out.name == "video.en.vtt"


class TestJSONWriter:
    def test_properties(self) -> None:
        w = JSONWriter()
        assert w.format_name == "json"
        assert w.file_extension == "json"

    def test_factory(self) -> None:
        assert isinstance(create_writer("json"), JSONWriter)

    def test_write_full(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        w = JSONWriter()
        out = w.write(sample_result, tmp_path / "test.json")
        data = json.loads(out.read_text())
        assert data["metadata"]["source_file"] == "video.mp4"
        assert data["metadata"]["subwyrhta_version"] == "0.1.0"
        assert len(data["segments"]) == 2
        assert len(data["segments"][0]["words"]) == 2
        assert data["quality"]["grade"] == "A"

    def test_write_no_words(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        w = JSONWriter(include_words=False)
        out = w.write(sample_result, tmp_path / "nw.json")
        data = json.loads(out.read_text())
        assert "words" not in data["segments"][0]

    def test_write_no_quality(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        w = JSONWriter(include_quality=False)
        out = w.write(sample_result, tmp_path / "nq.json")
        data = json.loads(out.read_text())
        assert "quality" not in data

    def test_write_empty(self, tmp_path: Path) -> None:
        w = JSONWriter()
        out = w.write(TranscriptionResult(), tmp_path / "empty.json")
        data = json.loads(out.read_text())
        assert data["segments"] == []

    def test_write_auto(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        w = JSONWriter()
        out = w.write_auto(sample_result, output_dir=str(tmp_path))
        assert out.name == "video.en.json"
