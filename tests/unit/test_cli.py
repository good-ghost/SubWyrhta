"""CLI 통합 테스트 (click CliRunner)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from src.main import cli
from src.version import __version__

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def sample_wav(tmp_path: Path) -> Generator[Path, None, None]:
    """ffmpeg로 3초 테스트 WAV 파일을 생성한다.

    snap ffmpeg는 /tmp 하위 경로에 쓰기 제한이 있으므로
    프로젝트 디렉토리에 생성 후 정리한다.
    """
    from src.audio.extractor import _find_ffmpeg

    try:
        ffmpeg = _find_ffmpeg()
    except Exception:
        pytest.skip("ffmpeg not found")

    # snap ffmpeg는 임의 tmp 경로 접근 불가 → 프로젝트 루트에 생성
    wav_path = Path(__file__).resolve().parent.parent.parent / ".test_cli_sample.wav"
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not wav_path.exists():
        pytest.skip("ffmpeg failed to create test wav")

    yield wav_path

    if wav_path.exists():
        wav_path.unlink()


class TestCLIHelp:
    """CLI 도움말 테스트."""

    def test_main_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.output
        assert "info" in result.output
        assert "list-models" in result.output

    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        # 버전의 기준은 pyproject.toml 하나다 (src/version.py)
        assert __version__ in result.output

    def test_transcribe_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--source-lang" in result.output
        assert "--dry-run" in result.output
        assert "--output-dir" in result.output
        assert "--output-format" in result.output

    def test_info_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["info", "--help"])
        assert result.exit_code == 0


class TestListModels:
    """list-models 서브커맨드 테스트."""

    def test_output(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["list-models"])
        assert result.exit_code == 0
        assert "whisperx" in result.output
        assert "faster-whisper" in result.output
        assert "large-v3" in result.output
        assert "distil-large-v3" in result.output
        assert "디바이스 정보" in result.output


class TestInfoCommand:
    """info 서브커맨드 테스트."""

    def test_info(self, runner: CliRunner, sample_wav: Path) -> None:
        result = runner.invoke(cli, ["info", str(sample_wav)])
        assert result.exit_code == 0
        assert "파일:" in result.output
        assert "길이:" in result.output
        assert "3.0" in result.output

    def test_info_nonexistent(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["info", "/nonexistent/file.mp4"])
        assert result.exit_code != 0


class TestTranscribeDryRun:
    """transcribe --dry-run 테스트."""

    def test_dry_run(self, runner: CliRunner, sample_wav: Path) -> None:
        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(sample_wav),
                "--dry-run",
                "--model",
                "faster-whisper",
                "--source-lang",
                "en",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0
        assert "Dry Run" in result.output
        assert "faster-whisper" in result.output
        assert "en" in result.output
        assert "cpu" in result.output

    def test_dry_run_with_options(self, runner: CliRunner, sample_wav: Path) -> None:
        result = runner.invoke(
            cli,
            [
                "transcribe",
                str(sample_wav),
                "--dry-run",
                "--batch-size",
                "8",
                "--beam-size",
                "3",
                "--temperature",
                "0.5",
                "--output-format",
                "vtt",
            ],
        )
        assert result.exit_code == 0
        assert "8" in result.output
        assert "vtt" in result.output


class TestVerboseQuiet:
    """--verbose / --quiet 로깅 레벨 테스트."""

    def test_verbose(self, runner: CliRunner, sample_wav: Path) -> None:
        result = runner.invoke(cli, ["--verbose", "transcribe", str(sample_wav), "--dry-run"])
        assert result.exit_code == 0

    def test_quiet(self, runner: CliRunner, sample_wav: Path) -> None:
        result = runner.invoke(cli, ["--quiet", "transcribe", str(sample_wav), "--dry-run"])
        assert result.exit_code == 0


class TestConfigFile:
    """--config 옵션 테스트."""

    def test_config_override(self, runner: CliRunner, sample_wav: Path, tmp_path: Path) -> None:
        import yaml

        config_data = {"stt": {"model": "faster-whisper", "model_size": "tiny"}}
        config_file = tmp_path / "custom.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = runner.invoke(
            cli,
            [
                "--config",
                str(config_file),
                "transcribe",
                str(sample_wav),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "faster-whisper" in result.output
        assert "tiny" in result.output


class TestBatchCli:
    def test_output_name_rejected_for_batch(self, runner: CliRunner, tmp_path: Path) -> None:
        for name in ("a.wav", "b.wav"):
            (tmp_path / name).write_bytes(b"x")
        result = runner.invoke(cli, ["transcribe", str(tmp_path), "--output-name", "x"])
        assert result.exit_code != 0
        assert "--output-name" in result.output

    def test_batch_dry_run_lists_files(self, runner: CliRunner, tmp_path: Path) -> None:
        for name in ("a.wav", "b.mp3"):
            (tmp_path / name).write_bytes(b"x")
        result = runner.invoke(cli, ["transcribe", str(tmp_path / "*.wav"), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "배치 1개" in result.output and "a.wav" in result.output

    def test_no_input(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["transcribe"])
        assert result.exit_code != 0
        assert "--file-list" in result.output

    def test_batch_failure_exit_code(self, runner: CliRunner, tmp_path: Path) -> None:
        for name in ("a.wav", "b.wav"):
            (tmp_path / name).write_bytes(b"x")
        with patch("src.main.TranscribeJob.run_one", side_effect=RuntimeError("boom")):
            result = runner.invoke(
                cli, ["transcribe", str(tmp_path), "--failed-list", str(tmp_path / "f.txt")]
            )
        assert result.exit_code == 1
        assert "실패 2" in result.output
        assert (tmp_path / "f.txt").exists()
