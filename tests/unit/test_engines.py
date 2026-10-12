"""엔진 팩토리 및 기반 클래스 단위 테스트."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.config import STTConfig
from src.core.exceptions import ConfigError, TranscriptionError
from src.engines.base import (
    STTEngine,
    build_load_attempts,
    get_device_info,
    resolve_compute_type,
    resolve_device,
)
from src.engines.engine_factory import create_engine, get_available_engines
from src.engines.faster_whisper_engine import (
    FasterWhisperEngine,
    _logprob_to_confidence,
    _resolve_model_name,
)
from src.engines.whisperx_engine import WhisperXEngine, _extract_words


class TestResolveDevice:
    """디바이스 감지 테스트."""

    def test_explicit_device(self) -> None:
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("mps") == "mps"

    def test_auto_returns_valid(self) -> None:
        result = resolve_device("auto")
        assert result in ("cpu", "cuda", "mps")


class TestResolveComputeType:
    """compute_type 해석 테스트."""

    def test_explicit(self) -> None:
        assert resolve_compute_type("float32", "cpu") == "float32"
        assert resolve_compute_type("int8", "cuda") == "int8"

    def test_auto_gpu(self) -> None:
        assert resolve_compute_type("auto", "cuda") == "float16"
        assert resolve_compute_type("auto", "mps") == "float16"

    def test_auto_cpu(self) -> None:
        assert resolve_compute_type("auto", "cpu") == "float32"


class TestBuildLoadAttempts:
    """폴백 체인 구성 테스트."""

    def test_cuda_float16(self) -> None:
        attempts = build_load_attempts("cuda", "float16")
        assert attempts == [("cuda", "float16"), ("cuda", "int8"), ("cpu", "float32")]

    def test_cuda_int8(self) -> None:
        attempts = build_load_attempts("cuda", "int8")
        assert attempts == [("cuda", "int8"), ("cpu", "float32")]

    def test_mps(self) -> None:
        attempts = build_load_attempts("mps", "float16")
        assert attempts == [("mps", "float16"), ("cpu", "float32")]

    def test_cpu_float32(self) -> None:
        attempts = build_load_attempts("cpu", "float32")
        assert attempts == [("cpu", "float32")]

    def test_cpu_float16_fallback(self) -> None:
        attempts = build_load_attempts("cpu", "float16")
        assert attempts == [("cpu", "float16"), ("cpu", "float32")]


class TestGetDeviceInfo:
    """디바이스 정보 테스트."""

    def test_returns_dict(self) -> None:
        info = get_device_info()
        assert isinstance(info, dict)
        assert "cpu" in info["available_devices"]

    def test_cuda_vram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch = pytest.importorskip("torch")
        props = MagicMock(spec=["total_memory"], total_memory=8 * 1024**3)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: props)

        info = get_device_info()

        assert "cuda" in info["available_devices"]
        assert info["cuda_vram_gb"] == 8.0

    def test_cuda_vram_failure_keeps_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        torch = pytest.importorskip("torch")

        def fail(_index: int) -> None:
            raise RuntimeError("CUDA error")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")
        monkeypatch.setattr(torch.cuda, "get_device_properties", fail)

        info = get_device_info()

        assert "cuda" in info["available_devices"]
        assert "cuda_vram_gb" not in info


class TestSTTEngineABC:
    """STTEngine ABC 테스트."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            STTEngine(STTConfig())  # type: ignore[abstract]


class TestEngineFactory:
    """엔진 팩토리 테스트."""

    def test_available_engines(self) -> None:
        engines = get_available_engines()
        assert "whisperx" in engines
        assert "faster-whisper" in engines
        assert "distil-whisper" in engines

    def test_create_whisperx(self) -> None:
        engine = create_engine(STTConfig(model="whisperx"))
        assert isinstance(engine, WhisperXEngine)
        assert engine.engine_name == "whisperx"

    def test_create_faster_whisper(self) -> None:
        engine = create_engine(STTConfig(model="faster-whisper"))
        assert isinstance(engine, FasterWhisperEngine)

    def test_distil_whisper_alias(self) -> None:
        engine = create_engine(STTConfig(model="distil-whisper"))
        assert isinstance(engine, FasterWhisperEngine)

    def test_unknown_engine(self) -> None:
        with pytest.raises(ConfigError, match="알 수 없는 STT 엔진"):
            create_engine(STTConfig(model="nonexistent"))

    def test_get_model_info(self) -> None:
        engine = create_engine(STTConfig(model="whisperx"))
        info = engine.get_model_info()
        assert info["engine"] == "whisperx"
        assert info["is_loaded"] is False


class TestWhisperXEngine:
    """WhisperXEngine 테스트."""

    def test_transcribe_before_load(self) -> None:
        engine = WhisperXEngine(STTConfig())
        with pytest.raises(TranscriptionError, match="로드되지 않았습니다"):
            engine.transcribe("test.wav")

    def test_extract_words(self) -> None:
        seg = {
            "words": [
                {"word": "Hello", "start": 0.5, "end": 0.9, "score": 0.98},
                {"word": "world", "start": 1.0, "end": 1.4, "score": 0.96},
            ]
        }
        words = _extract_words(seg)
        assert len(words) == 2
        assert words[0].word == "Hello"
        assert words[0].confidence == 0.98

    def test_extract_words_skip_no_timestamp(self) -> None:
        seg = {"words": [{"word": "orphan"}]}
        words = _extract_words(seg)
        assert len(words) == 0

    def test_extract_words_empty(self) -> None:
        assert _extract_words({}) == []
        assert _extract_words({"words": []}) == []

    def test_convert_result(self) -> None:
        engine = WhisperXEngine(STTConfig(model_size="small"))
        raw = {
            "segments": [
                {
                    "text": " Hello world. ",
                    "start": 0.5,
                    "end": 1.5,
                    "words": [
                        {"word": "Hello", "start": 0.5, "end": 0.9, "score": 0.9},
                        {"word": "world.", "start": 1.0, "end": 1.4, "score": 0.8},
                    ],
                }
            ]
        }
        result = engine._convert_result(raw, Path("test.wav"), "en")
        assert result.segment_count == 1
        assert result.segments[0].text == "Hello world."
        assert result.metadata.model == "whisperx/small"
        assert len(result.segments[0].words) == 2


class TestFasterWhisperEngine:
    """FasterWhisperEngine 테스트."""

    def test_transcribe_before_load(self) -> None:
        engine = FasterWhisperEngine(STTConfig())
        with pytest.raises(TranscriptionError, match="로드되지 않았습니다"):
            engine.transcribe("test.wav")

    def test_resolve_model_name(self) -> None:
        assert _resolve_model_name("large-v3") == "large-v3"
        assert "Systran" in _resolve_model_name("distil-large-v3")

    def test_logprob_to_confidence(self) -> None:
        assert _logprob_to_confidence(0.0) == 1.0
        assert 0.0 < _logprob_to_confidence(-0.5) < 1.0
        assert _logprob_to_confidence(-100.0) < 0.01

    def test_convert_result_model_label(self) -> None:
        engine = FasterWhisperEngine(STTConfig(model_size="distil-large-v3"))
        mock_seg = MagicMock()
        mock_seg.text = " Test "
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.avg_log_prob = -0.3
        mock_seg.words = None

        result = engine._convert_result(
            segments_iter=[mock_seg],
            audio_path=Path("test.wav"),
            language="en",
            duration=10.0,
        )
        assert result.metadata.model == "distil-whisper/distil-large-v3"

    def test_convert_result_regular_model(self) -> None:
        engine = FasterWhisperEngine(STTConfig(model_size="large-v3"))
        mock_seg = MagicMock()
        mock_seg.text = " Test "
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.avg_log_prob = -0.2
        mock_seg.words = None

        result = engine._convert_result(
            segments_iter=[mock_seg],
            audio_path=Path("test.wav"),
            language="ko",
            duration=30.0,
        )
        assert result.metadata.model == "faster-whisper/large-v3"
        assert result.metadata.language == "ko"
