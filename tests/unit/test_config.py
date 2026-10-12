"""설정 로더 단위 테스트."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from src.core.config import (
    AppConfig,
    STTConfig,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestSTTConfig:
    """STTConfig 기본값 테스트."""

    def test_defaults(self) -> None:
        cfg = STTConfig()
        assert cfg.model == "whisperx"
        assert cfg.model_size == "large-v3"
        assert cfg.device == "auto"
        assert cfg.compute_type == "auto"
        assert cfg.batch_size == 16
        assert cfg.beam_size == 5
        assert cfg.temperature == 0.0
        assert cfg.source_lang == "auto"


class TestAppConfigLoad:
    """AppConfig.load() 테스트."""

    def test_load_defaults(self) -> None:
        """config_path=None이면 모든 기본값이 적용된다."""
        cfg = AppConfig.load()
        assert cfg.stt.model == "whisperx"
        assert cfg.audio.vad.enabled is True
        assert cfg.segmentation.mode == "advanced"
        assert cfg.output.format == "srt"
        assert cfg.logging.level == "INFO"

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """YAML 파일에서 설정을 로드한다."""
        yaml_data = {
            "stt": {"model": "faster-whisper", "model_size": "small", "batch_size": 8},
            "segmentation": {"mode": "basic", "max_cps": 12.0},
            "output": {"format": "vtt"},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_data), encoding="utf-8")

        cfg = AppConfig.load(config_path=config_file)
        assert cfg.stt.model == "faster-whisper"
        assert cfg.stt.model_size == "small"
        assert cfg.stt.batch_size == 8
        assert cfg.segmentation.mode == "basic"
        assert cfg.segmentation.max_cps == 12.0
        assert cfg.output.format == "vtt"
        # 미지정 필드는 기본값
        assert cfg.stt.device == "auto"
        assert cfg.audio.vad.enabled is True

    def test_load_nonexistent_file(self) -> None:
        """존재하지 않는 파일은 기본값으로 로드된다."""
        cfg = AppConfig.load(config_path="/nonexistent/config.yaml")
        assert cfg.stt.model == "whisperx"

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """빈 YAML 파일은 기본값으로 로드된다."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")
        cfg = AppConfig.load(config_path=config_file)
        assert cfg.stt.model == "whisperx"

    def test_cli_overrides(self) -> None:
        """CLI 오버라이드가 적용된다."""
        cfg = AppConfig.load(
            cli_overrides={
                "stt.device": "cpu",
                "stt.batch_size": 4,
                "segmentation.max_cps": 9.0,
                "output.format": "json",
            }
        )
        assert cfg.stt.device == "cpu"
        assert cfg.stt.batch_size == 4
        assert cfg.segmentation.max_cps == 9.0
        assert cfg.output.format == "json"

    def test_cli_overrides_over_yaml(self, tmp_path: Path) -> None:
        """CLI 오버라이드가 YAML 설정을 덮어쓴다."""
        yaml_data = {"stt": {"device": "cuda", "batch_size": 16}}
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_data), encoding="utf-8")

        cfg = AppConfig.load(
            config_path=config_file,
            cli_overrides={"stt.device": "cpu"},
        )
        assert cfg.stt.device == "cpu"
        assert cfg.stt.batch_size == 16  # YAML 값 유지

    def test_env_override_hf_token(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """HF_TOKEN 환경변수가 diarization.hf_token을 오버라이드한다."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_token_123")
        cfg = AppConfig.load()
        assert cfg.diarization.hf_token == "hf_test_token_123"

    def test_env_override_model_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SUBWYRHTA_MODEL_DIR 환경변수가 stt.model_dir을 오버라이드한다."""
        monkeypatch.setenv("SUBWYRHTA_MODEL_DIR", "/custom/models")
        cfg = AppConfig.load()
        assert cfg.stt.model_dir == "/custom/models"

    def test_env_override_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SUBWYRHTA_DEVICE 환경변수가 stt.device를 오버라이드한다."""
        monkeypatch.setenv("SUBWYRHTA_DEVICE", "mps")
        cfg = AppConfig.load()
        assert cfg.stt.device == "mps"


class TestNestedAudioConfig:
    """AudioConfig 중첩 설정 테스트."""

    def test_vad_defaults(self) -> None:
        cfg = AppConfig.load()
        assert cfg.audio.vad.enabled is True
        assert cfg.audio.vad.threshold == 0.5
        assert cfg.audio.vad.pre_vad is True
        assert cfg.audio.vad.merge_gap_s == 0.5

    def test_denoise_defaults(self) -> None:
        cfg = AppConfig.load()
        assert cfg.audio.denoise.enabled is False
        assert cfg.audio.denoise.model == "noisereduce"

    def test_vad_from_yaml(self, tmp_path: Path) -> None:
        yaml_data = {
            "audio": {
                "vad": {"threshold": 0.7, "pre_vad": False},
                "denoise": {"enabled": True},
            }
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_data), encoding="utf-8")

        cfg = AppConfig.load(config_path=config_file)
        assert cfg.audio.vad.threshold == 0.7
        assert cfg.audio.vad.pre_vad is False
        assert cfg.audio.denoise.enabled is True


class TestUnknownKeysIgnored:
    """알 수 없는 설정 키가 무시되는지 테스트."""

    def test_unknown_keys(self, tmp_path: Path) -> None:
        yaml_data = {
            "stt": {"model": "whisperx", "unknown_key": "value"},
            "unknown_section": {"foo": "bar"},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_data), encoding="utf-8")

        cfg = AppConfig.load(config_path=config_file)
        assert cfg.stt.model == "whisperx"
