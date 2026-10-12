"""ASS Writer 단위 테스트."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.models.result import Metadata, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.writers.ass_writer import (
    ASSWriter,
    _build_style_map,
    _collect_speakers,
    _format_ass_time,
    _get_style_name,
)
from src.writers.writer_factory import create_writer

if TYPE_CHECKING:
    from pathlib import Path


class TestFormatASSTime:
    def test_zero(self) -> None:
        assert _format_ass_time(0.0) == "0:00:00.00"

    def test_seconds(self) -> None:
        assert _format_ass_time(1.5) == "0:00:01.50"

    def test_minutes(self) -> None:
        assert _format_ass_time(61.25) == "0:01:01.25"

    def test_hours(self) -> None:
        assert _format_ass_time(3661.99) == "1:01:01.99"

    def test_negative(self) -> None:
        assert _format_ass_time(-1.0) == "0:00:00.00"


class TestCollectSpeakers:
    def test_unique_ordered(self) -> None:
        segs = [
            TranscriptionSegment(1, "a", 0, 1, speaker="B"),
            TranscriptionSegment(2, "b", 1, 2, speaker="A"),
            TranscriptionSegment(3, "c", 2, 3, speaker="B"),
        ]
        assert _collect_speakers(segs) == ["B", "A"]

    def test_multi_speaker(self) -> None:
        segs = [TranscriptionSegment(1, "a", 0, 1, speaker=["A", "C"])]
        assert _collect_speakers(segs) == ["A", "C"]

    def test_no_speakers(self) -> None:
        segs = [TranscriptionSegment(1, "a", 0, 1)]
        assert _collect_speakers(segs) == []


class TestBuildStyleMap:
    def test_two_speakers_positions(self) -> None:
        sm = _build_style_map(["A", "B"])
        assert sm["A"]["MarginL"] == 20
        assert sm["A"]["MarginR"] == 200
        assert sm["B"]["MarginL"] == 200
        assert sm["B"]["MarginR"] == 20

    def test_three_speakers_center(self) -> None:
        sm = _build_style_map(["A", "B", "C"])
        for s in ["A", "B", "C"]:
            assert sm[s]["MarginL"] == 20
            assert sm[s]["MarginR"] == 20

    def test_colors_unique(self) -> None:
        sm = _build_style_map(["A", "B", "C"])
        colors = [sm[s]["color"] for s in ["A", "B", "C"]]
        assert len(set(colors)) == 3

    def test_empty(self) -> None:
        assert _build_style_map([]) == {}


class TestGetStyleName:
    def test_single_speaker(self) -> None:
        sm = _build_style_map(["A"])
        seg = TranscriptionSegment(1, "x", 0, 1, speaker="A")
        assert _get_style_name(seg, sm) == "A"

    def test_no_speaker(self) -> None:
        sm = _build_style_map(["A"])
        seg = TranscriptionSegment(1, "x", 0, 1)
        assert _get_style_name(seg, sm) == "Default"

    def test_multi_speaker(self) -> None:
        sm = _build_style_map(["A", "B"])
        seg = TranscriptionSegment(1, "x", 0, 1, speaker=["A", "B"])
        assert _get_style_name(seg, sm) == "A"

    def test_unknown_speaker(self) -> None:
        sm = _build_style_map(["A"])
        seg = TranscriptionSegment(1, "x", 0, 1, speaker="Z")
        assert _get_style_name(seg, sm) == "Default"


class TestASSWriter:
    @pytest.fixture()
    def writer(self) -> ASSWriter:
        return ASSWriter()

    @pytest.fixture()
    def sample_result(self) -> TranscriptionResult:
        return TranscriptionResult(
            metadata=Metadata(source_file="video.mp4", language="en", subwyrhta_version="0.1.0"),
            segments=[
                TranscriptionSegment(1, "Hello.", 0.5, 1.5, speaker="SPEAKER_00"),
                TranscriptionSegment(2, "World.", 2.0, 3.5, speaker="SPEAKER_01"),
                TranscriptionSegment(3, "Both.", 4.0, 5.0, speaker=["SPEAKER_00", "SPEAKER_01"]),
                TranscriptionSegment(4, "Narrator.", 6.0, 7.0),
            ],
        )

    def test_properties(self, writer: ASSWriter) -> None:
        assert writer.format_name == "ass"
        assert writer.file_extension == "ass"

    def test_factory(self) -> None:
        w = create_writer("ass")
        assert isinstance(w, ASSWriter)

    def test_write_structure(
        self, writer: ASSWriter, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        out = writer.write(sample_result, tmp_path / "test.ass")
        content = out.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content

    def test_write_speaker_styles(
        self, writer: ASSWriter, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        out = writer.write(sample_result, tmp_path / "test.ass")
        content = out.read_text(encoding="utf-8")
        assert "Style: SPEAKER_00" in content
        assert "Style: SPEAKER_01" in content
        assert "Style: Default" in content

    def test_write_dialogues(
        self, writer: ASSWriter, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        out = writer.write(sample_result, tmp_path / "test.ass")
        content = out.read_text(encoding="utf-8")
        lines = [line for line in content.split("\n") if line.startswith("Dialogue:")]
        assert len(lines) == 4
        assert "Hello." in lines[0]
        assert "Narrator." in lines[3]
        assert "Default" in lines[3]

    def test_write_auto_diarized(
        self, writer: ASSWriter, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        out = writer.write_auto(sample_result, output_dir=str(tmp_path), diarized=True)
        assert out.name == "video.en.diarized.ass"

    def test_write_empty(self, writer: ASSWriter, tmp_path: Path) -> None:
        r = TranscriptionResult(metadata=Metadata(source_file="x.mp4", language="en"))
        out = writer.write(r, tmp_path / "empty.ass")
        content = out.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "Dialogue:" not in content

    def test_newline_conversion(self, writer: ASSWriter, tmp_path: Path) -> None:
        r = TranscriptionResult(
            metadata=Metadata(source_file="x.mp4", language="en"),
            segments=[TranscriptionSegment(1, "Line1\nLine2", 0, 1)],
        )
        out = writer.write(r, tmp_path / "nl.ass")
        content = out.read_text(encoding="utf-8")
        assert "\\N" in content
        assert "Line1\\NLine2" in content
