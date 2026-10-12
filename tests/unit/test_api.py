"""Python API 단위 테스트."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.api import SubWyrhta, TranscribeConfig, save_result
from src.models.result import Metadata, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def sample_result() -> TranscriptionResult:
    return TranscriptionResult(
        metadata=Metadata(source_file="video.mp4", language="en"),
        segments=[
            TranscriptionSegment(
                1,
                "Hello.",
                0.5,
                1.5,
                0.9,
                words=[WordSegment("Hello.", 0.5, 1.4, 0.9)],
            ),
        ],
    )


class TestTranscribeConfig:
    def test_defaults_leave_config_file_values(self) -> None:
        # 지정하지 않은 값은 None: 설정 파일·코드 기본값을 덮어쓰지 않는다
        tc = TranscribeConfig()
        assert tc.source_lang is None
        assert tc.diarize is None
        assert tc.segmentation_mode is None
        assert tc.max_cps is None
        assert tc.output_format is None
        assert tc.denoise is None
        assert tc.writes_files is False

    def test_output_options_write_files(self) -> None:
        assert TranscribeConfig(output_format="vtt").writes_files is True
        assert TranscribeConfig(output_dir="./subs").writes_files is True

    def test_custom(self) -> None:
        tc = TranscribeConfig(
            source_lang="ko",
            diarize=True,
            hf_token="hf_xxx",
            max_cps=9.0,
            denoise=True,
            output_format="ass",
        )
        assert tc.source_lang == "ko"
        assert tc.diarize is True
        assert tc.max_cps == 9.0


class TestSubWyrhta:
    def test_init(self) -> None:
        stt = SubWyrhta(model="faster-whisper", model_size="small", device="cpu")
        assert stt._model == "faster-whisper"
        assert stt._model_size == "small"
        assert stt._device == "cpu"

    def test_build_config_defaults(self) -> None:
        stt = SubWyrhta(model="whisperx", model_size="large-v3", device="auto")
        cfg = stt._build_config(TranscribeConfig())
        assert cfg.stt.model == "whisperx"
        assert cfg.stt.model_size == "large-v3"
        assert cfg.stt.source_lang == "auto"
        assert cfg.segmentation.mode == "advanced"

    def test_build_config_overrides(self) -> None:
        stt = SubWyrhta(model="faster-whisper", model_size="small", device="cpu")
        tc = TranscribeConfig(
            source_lang="ko",
            max_cps=9.0,
            segmentation_mode="basic",
            denoise=True,
        )
        cfg = stt._build_config(tc)
        assert cfg.stt.model == "faster-whisper"
        assert cfg.stt.device == "cpu"
        assert cfg.stt.source_lang == "ko"
        assert cfg.segmentation.max_cps == 9.0
        assert cfg.segmentation.mode == "basic"
        assert cfg.audio.denoise.enabled is True

    def test_build_config_diarize(self) -> None:
        stt = SubWyrhta()
        tc = TranscribeConfig(diarize=True, hf_token="hf_test", num_speakers="3")
        cfg = stt._build_config(tc)
        assert cfg.diarization.enabled is True
        assert cfg.diarization.hf_token == "hf_test"
        assert cfg.diarization.num_speakers == "3"


class TestConfigFileIsNotOverridden:
    """SubWyrhta(config_path=…) 의 설정 파일 값은 지정한 인자만 덮어쓴다 (2026-10-09 결함)."""

    def test_yaml_values_survive_unspecified_args(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "stt:\n  model: faster-whisper\n  model_size: small\n  source_lang: ko\n"
            "segmentation:\n  mode: basic\n  min_duration_s: 1.5\n"
            "diarization:\n  num_speakers: '2'\n",
            encoding="utf-8",
        )
        cfg = SubWyrhta(config_path=config_file)._build_config(TranscribeConfig())
        assert (cfg.stt.model, cfg.stt.model_size, cfg.stt.source_lang) == (
            "faster-whisper",
            "small",
            "ko",
        )
        assert (cfg.segmentation.mode, cfg.segmentation.min_duration_s) == ("basic", 1.5)
        assert cfg.diarization.num_speakers == "2"

    def test_specified_args_win(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "stt:\n  model: faster-whisper\n  source_lang: ko\n", encoding="utf-8"
        )
        cfg = SubWyrhta(model="whisperx", config_path=config_file)._build_config(
            TranscribeConfig(source_lang="en")
        )
        assert (cfg.stt.model, cfg.stt.source_lang) == ("whisperx", "en")

    def test_without_config_file_uses_code_defaults(self) -> None:
        cfg = SubWyrhta()._build_config(TranscribeConfig())
        assert (cfg.stt.model, cfg.stt.model_size) == ("whisperx", "large-v3")
        assert cfg.diarization.enabled is False


class TestSaveResult:
    def test_save_srt(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        p = save_result(sample_result, tmp_path / "out.srt")
        assert p.exists()
        assert "Hello." in p.read_text()

    def test_save_vtt(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        p = save_result(sample_result, tmp_path / "out.vtt")
        assert "WEBVTT" in p.read_text()

    def test_save_json(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        p = save_result(sample_result, tmp_path / "out.json")
        data = json.loads(p.read_text())
        assert data["segments"][0]["text"] == "Hello."

    def test_save_ass(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        p = save_result(sample_result, tmp_path / "out.ass")
        assert "[Script Info]" in p.read_text()

    def test_save_unknown_ext(self, sample_result: TranscriptionResult, tmp_path: Path) -> None:
        p = save_result(sample_result, tmp_path / "out.xyz")
        assert p.exists()  # defaults to srt
