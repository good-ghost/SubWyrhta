"""SRT Writer 및 출력 유틸리티 단위 테스트."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.core.exceptions import ConfigError
from src.models.result import Metadata, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.writers.base import SubtitleWriter, build_output_path
from src.writers.srt_writer import SRTWriter, _format_timestamp, _segment_to_srt_block
from src.writers.writer_factory import create_writer, get_available_formats

if TYPE_CHECKING:
    from pathlib import Path


class TestFormatTimestamp:
    """SRT 타임스탬프 포맷 테스트."""

    def test_zero(self) -> None:
        assert _format_timestamp(0.0) == "00:00:00,000"

    def test_seconds(self) -> None:
        assert _format_timestamp(1.5) == "00:00:01,500"

    def test_minutes(self) -> None:
        assert _format_timestamp(61.123) == "00:01:01,123"

    def test_hours(self) -> None:
        assert _format_timestamp(3661.999) == "01:01:01,999"

    def test_negative(self) -> None:
        assert _format_timestamp(-1.0) == "00:00:00,000"

    def test_rounding_overflow(self) -> None:
        # 3599.9999 → 밀리초 반올림으로 1:00:00,000
        assert _format_timestamp(3599.9999) == "01:00:00,000"

    def test_millisecond_precision(self) -> None:
        assert _format_timestamp(0.001) == "00:00:00,001"
        assert _format_timestamp(0.999) == "00:00:00,999"


class TestSegmentToSRTBlock:
    """SRT 블록 변환 테스트."""

    def test_basic(self) -> None:
        seg = TranscriptionSegment(index=1, text="Hello world", start=0.5, end=1.44)
        block = _segment_to_srt_block(seg)
        lines = block.strip().split("\n")
        assert lines[0] == "1"
        assert lines[1] == "00:00:00,500 --> 00:00:01,440"
        assert lines[2] == "Hello world"

    def test_multiline_text(self) -> None:
        seg = TranscriptionSegment(index=2, text="Line one\nLine two", start=2.0, end=4.0)
        block = _segment_to_srt_block(seg)
        assert "Line one\nLine two" in block


class TestBuildOutputPath:
    """출력 경로 생성 테스트."""

    def test_default_naming(self, tmp_path: Path) -> None:
        p = build_output_path("video.mp4", "en", "srt", output_dir=str(tmp_path))
        assert p.name == "video.en.srt"
        assert p.parent == tmp_path

    def test_diarized_naming(self, tmp_path: Path) -> None:
        p = build_output_path("video.mp4", "en", "srt", output_dir=str(tmp_path), diarized=True)
        assert p.name == "video.en.diarized.srt"

    def test_custom_name(self, tmp_path: Path) -> None:
        p = build_output_path(
            "video.mp4", "en", "srt", output_dir=str(tmp_path), output_name="custom"
        )
        assert p.name == "custom.srt"

    def test_custom_name_with_ext(self, tmp_path: Path) -> None:
        p = build_output_path(
            "video.mp4", "en", "srt", output_dir=str(tmp_path), output_name="custom.srt"
        )
        assert p.name == "custom.srt"

    def test_auto_numbering(self, tmp_path: Path) -> None:
        # 첫 파일 생성
        first = build_output_path("video.mp4", "en", "srt", output_dir=str(tmp_path))
        first.write_text("dummy")
        # 두 번째는 _2
        second = build_output_path("video.mp4", "en", "srt", output_dir=str(tmp_path))
        assert second.name == "video.en_2.srt"
        second.write_text("dummy")
        # 세 번째는 _3
        third = build_output_path("video.mp4", "en", "srt", output_dir=str(tmp_path))
        assert third.name == "video.en_3.srt"

    def test_creates_directory(self, tmp_path: Path) -> None:
        p = build_output_path("video.mp4", "en", "srt", output_dir=str(tmp_path / "sub" / "dir"))
        assert p.parent.exists()

    def test_unknown_language(self, tmp_path: Path) -> None:
        p = build_output_path("video.mp4", "", "srt", output_dir=str(tmp_path))
        assert p.name == "video.unknown.srt"


class TestSRTWriter:
    """SRTWriter 테스트."""

    @pytest.fixture()
    def sample_result(self) -> TranscriptionResult:
        return TranscriptionResult(
            metadata=Metadata(source_file="video.mp4", language="en"),
            segments=[
                TranscriptionSegment(index=1, text="First line.", start=0.5, end=1.5),
                TranscriptionSegment(index=2, text="Second line.", start=2.0, end=3.5),
                TranscriptionSegment(index=3, text="Third line.", start=4.0, end=5.0),
            ],
        )

    def test_properties(self) -> None:
        writer = SRTWriter()
        assert writer.format_name == "srt"
        assert writer.file_extension == "srt"

    def test_write(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        writer = SRTWriter()
        out = writer.write(sample_result, tmp_path / "output.srt")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "1\n00:00:00,500 --> 00:00:01,500\nFirst line." in content
        assert "2\n00:00:02,000 --> 00:00:03,500\nSecond line." in content
        assert "3\n00:00:04,000 --> 00:00:05,000\nThird line." in content

    def test_write_empty(self, tmp_path: Path) -> None:
        writer = SRTWriter()
        result = TranscriptionResult()
        out = writer.write(result, tmp_path / "empty.srt")
        assert out.exists()
        assert out.read_text(encoding="utf-8") == ""

    def test_write_auto(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        writer = SRTWriter()
        out = writer.write_auto(sample_result, output_dir=str(tmp_path))
        assert out.name == "video.en.srt"
        assert out.exists()

    def test_write_creates_parent_dirs(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        writer = SRTWriter()
        out = writer.write(sample_result, tmp_path / "a" / "b" / "test.srt")
        assert out.exists()


class TestSubtitleWriterABC:
    """SubtitleWriter ABC 테스트."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            SubtitleWriter()  # type: ignore[abstract]


class TestWriterFactory:
    """WriterFactory 테스트."""

    def test_available_formats(self) -> None:
        formats = get_available_formats()
        assert "srt" in formats
        assert "json" in formats

    def test_create_srt(self) -> None:
        writer = create_writer("srt")
        assert isinstance(writer, SRTWriter)

    def test_create_case_insensitive(self) -> None:
        writer = create_writer("SRT")
        assert isinstance(writer, SRTWriter)

    def test_unknown_format(self) -> None:
        with pytest.raises(ConfigError, match="알 수 없는 출력 포맷"):
            create_writer("txt")
