"""ffmpeg 오디오 추출기 단위 테스트 (Mock 기반)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.audio.extractor import (
    AUDIO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    AudioExtractor,
    _find_ffmpeg,
    _find_ffprobe,
    _validate_input,
)
from src.core.exceptions import AudioExtractionError


class TestSupportedExtensions:
    """지원 확장자 테스트."""

    def test_video_extensions(self) -> None:
        for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
            assert ext in VIDEO_EXTENSIONS

    def test_audio_extensions(self) -> None:
        for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
            assert ext in AUDIO_EXTENSIONS

    def test_all_extensions(self) -> None:
        assert VIDEO_EXTENSIONS | AUDIO_EXTENSIONS == SUPPORTED_EXTENSIONS
        assert len(SUPPORTED_EXTENSIONS) == 19
        assert ".mka" in AUDIO_EXTENSIONS  # TO-DO 8


class TestValidateInput:
    """_validate_input 테스트."""

    def test_nonexistent_file(self) -> None:
        with pytest.raises(AudioExtractionError, match="존재하지 않습니다"):
            _validate_input(Path("/nonexistent/file.mp4"))

    def test_not_a_file(self, tmp_path: Path) -> None:
        with pytest.raises(AudioExtractionError, match="파일이 아닙니다"):
            _validate_input(tmp_path)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "test.xyz"
        f.write_text("dummy")
        with pytest.raises(AudioExtractionError, match="지원하지 않는"):
            _validate_input(f)

    def test_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.mp4"
        f.write_text("dummy")
        _validate_input(f)  # 예외 없음


class TestFindFFmpeg:
    """ffmpeg/ffprobe 찾기 테스트."""

    def test_find_ffmpeg(self) -> None:
        # 현재 시스템에 ffmpeg가 설치되어 있으면 경로 반환
        path = _find_ffmpeg()
        assert path is not None
        assert "ffmpeg" in path

    def test_find_ffprobe(self) -> None:
        path = _find_ffprobe()
        assert path is not None
        assert "ffprobe" in path.lower() or "ffmpeg" in path.lower()


class TestAudioExtractorBuildCommand:
    """AudioExtractor._build_command 테스트."""

    def test_basic_command(self) -> None:
        ext = AudioExtractor()
        cmd = ext._build_command(
            input_path=Path("input.mp4"),
            output_path=Path("output.wav"),
            audio_track=None,
            start=None,
            end=None,
        )
        assert cmd[0] == ext._ffmpeg
        assert "-i" in cmd
        assert "input.mp4" in cmd
        assert "output.wav" in cmd
        assert "-ac" in cmd
        assert "1" in cmd
        assert "-ar" in cmd
        assert "16000" in cmd
        assert "-vn" in cmd

    def test_with_audio_track(self) -> None:
        ext = AudioExtractor()
        cmd = ext._build_command(
            input_path=Path("input.mkv"),
            output_path=Path("out.wav"),
            audio_track=2,
            start=None,
            end=None,
        )
        assert "0:a:2" in " ".join(cmd)

    def test_default_audio_track(self) -> None:
        ext = AudioExtractor()
        cmd = ext._build_command(
            input_path=Path("input.mkv"),
            output_path=Path("out.wav"),
            audio_track=None,
            start=None,
            end=None,
        )
        assert "0:a:0" in " ".join(cmd)

    def test_with_time_range(self) -> None:
        ext = AudioExtractor()
        cmd = ext._build_command(
            input_path=Path("input.mp4"),
            output_path=Path("out.wav"),
            audio_track=None,
            start="00:01:00",
            end="00:02:30",
        )
        cmd_str = " ".join(cmd)
        assert "-ss" in cmd_str
        assert "00:01:00" in cmd_str
        assert "-to" in cmd_str
        assert "00:02:30" in cmd_str

    def test_ss_before_input(self) -> None:
        """시간 범위 옵션(-ss/-to)이 입력 파일 앞에 위치해야 한다."""
        ext = AudioExtractor()
        cmd = ext._build_command(
            input_path=Path("input.mp4"),
            output_path=Path("out.wav"),
            audio_track=None,
            start="10",
            end="20",
        )
        ss_idx = cmd.index("-ss")
        i_idx = cmd.index("-i")
        assert ss_idx < i_idx


class TestAudioExtractorContextManager:
    """AudioExtractor 컨텍스트 매니저 테스트."""

    def test_context_manager(self) -> None:
        with AudioExtractor(keep_audio=False) as ext:
            assert ext.keep_audio is False

    def test_keep_audio_flag(self) -> None:
        ext = AudioExtractor(keep_audio=True)
        assert ext.keep_audio is True


class TestAudioExtractorCleanup:
    """AudioExtractor.cleanup 테스트."""

    def test_cleanup_removes_temp_files(self, tmp_path: Path) -> None:
        ext = AudioExtractor(keep_audio=False)
        tmp_file = tmp_path / "temp.wav"
        tmp_file.write_text("dummy")
        ext._temp_files.append(tmp_file)
        assert tmp_file.exists()
        ext.cleanup()
        assert not tmp_file.exists()
        assert len(ext._temp_files) == 0

    def test_cleanup_keeps_with_flag(self, tmp_path: Path) -> None:
        ext = AudioExtractor(keep_audio=True)
        tmp_file = tmp_path / "temp.wav"
        tmp_file.write_text("dummy")
        ext._temp_files.append(tmp_file)
        ext.cleanup()
        assert tmp_file.exists()

    def test_cleanup_handles_missing_file(self, tmp_path: Path) -> None:
        ext = AudioExtractor(keep_audio=False)
        ext._temp_files.append(tmp_path / "nonexistent.wav")
        ext.cleanup()  # 예외 없이 완료


class TestStderrTail:
    """ffmpeg 오류 문구는 stderr 끝 줄만 남긴다 (배너·빌드 설정 제외)."""

    def test_keeps_last_lines(self) -> None:
        from src.audio.extractor import stderr_tail

        stderr = "ffmpeg version 7\nbuilt with gcc\n\nconfiguration: --x\nmoov atom not found\nError opening input\n"
        assert (
            stderr_tail(stderr) == "configuration: --x / moov atom not found / Error opening input"
        )
        assert stderr_tail(None) == ""
