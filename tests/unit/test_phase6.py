"""Phase 6 테스트: LLM 클라이언트, llm-assisted 세그멘테이션, 챕터, 엔진."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import LLMConfig, SegmentationConfig, STTConfig
from src.core.exceptions import LLMSegmentationError, ModelLoadError
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment


def _words(texts: list[str], gap: float = 0.05) -> list[WordSegment]:
    result: list[WordSegment] = []
    t = 0.0
    for text in texts:
        result.append(WordSegment(text, t, t + 0.3, 0.9))
        t += 0.3 + gap
    return result


# ══════════════════════════════════════════════
# LLM 클라이언트
# ══════════════════════════════════════════════


class TestLLMProviderFactory:
    def test_create_subscription(self) -> None:
        from src.segmentation.llm_client import (
            ClaudeCodeProvider,
            CodexProvider,
            create_llm_provider,
        )

        assert isinstance(
            create_llm_provider(LLMConfig(provider="claude-code")), ClaudeCodeProvider
        )
        assert isinstance(create_llm_provider(LLMConfig(provider="codex")), CodexProvider)

    def test_create_local(self) -> None:
        from src.segmentation.llm_client import OpenAICompatibleProvider, create_llm_provider

        cfg = LLMConfig(provider="local", model="m")
        p = create_llm_provider(cfg)
        assert isinstance(p, OpenAICompatibleProvider)

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "unknown"])
    def test_unknown_provider(self, provider: str) -> None:
        # OpenAI·Anthropic 은 구독(codex, claude-code)만 허용한다 (API 키 방식 없음)
        from src.segmentation.llm_client import create_llm_provider

        with pytest.raises(LLMSegmentationError, match="알 수 없는"):
            create_llm_provider(LLMConfig(provider=provider, model="m"))

    def test_provider_name(self) -> None:
        from src.segmentation.llm_client import create_llm_provider

        assert (
            create_llm_provider(LLMConfig(provider="gemini", model="m")).provider_name() == "gemini"
        )
        assert (
            create_llm_provider(LLMConfig(provider="local", model="m")).provider_name() == "local"
        )


class TestLLMResponse:
    def test_dataclass(self) -> None:
        from src.segmentation.llm_client import LLMResponse

        r = LLMResponse(text="hello", input_tokens=10, output_tokens=5, model="gpt-4o")
        assert r.text == "hello"
        assert r.input_tokens == 10


class TestResolveApiKey:
    def test_fixed_env_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.segmentation.llm_client import _resolve_api_key

        monkeypatch.setenv("GEMINI_API_KEY", "default_key")
        assert _resolve_api_key("gemini") == "default_key"

    def test_missing_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.segmentation.llm_client import _resolve_api_key

        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        assert _resolve_api_key("nvidia-nim") == ""

    def test_local_no_env(self) -> None:
        from src.segmentation.llm_client import _resolve_api_key

        assert _resolve_api_key("local") == ""
        assert _resolve_api_key("claude-code") == ""


class TestCompleteWithRetry:
    def test_success_first_try(self) -> None:
        from src.segmentation.llm_client import LLMResponse, complete_with_retry

        provider = MagicMock()
        provider.complete.return_value = LLMResponse(text="ok", input_tokens=1, output_tokens=1)
        result = complete_with_retry(provider, "sys", "usr", max_retries=2)
        assert result.text == "ok"
        assert provider.complete.call_count == 1

    def test_retry_on_generic_error(self) -> None:
        from src.segmentation.llm_client import LLMResponse, complete_with_retry

        provider = MagicMock()
        provider.complete.side_effect = [
            RuntimeError("fail"),
            LLMResponse(text="ok", input_tokens=1, output_tokens=1),
        ]
        result = complete_with_retry(provider, "sys", "usr", max_retries=2)
        assert result.text == "ok"
        assert provider.complete.call_count == 2

    def test_no_retry_on_llm_error(self) -> None:
        from src.segmentation.llm_client import complete_with_retry

        provider = MagicMock()
        provider.complete.side_effect = LLMSegmentationError("bad")
        with pytest.raises(LLMSegmentationError):
            complete_with_retry(provider, "sys", "usr", max_retries=3)
        assert provider.complete.call_count == 1

    def test_all_retries_fail(self) -> None:
        from src.segmentation.llm_client import complete_with_retry

        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("fail")
        with pytest.raises(LLMSegmentationError, match="모두 실패"):
            complete_with_retry(provider, "sys", "usr", max_retries=0)


class TestAvailability:
    def test_openai_available(self) -> None:
        from src.segmentation.llm_client import is_openai_available

        assert isinstance(is_openai_available(), bool)


# ══════════════════════════════════════════════
# llm-assisted 세그멘테이션
# ══════════════════════════════════════════════


class TestLLMAssistedSegmentation:
    def test_factory(self) -> None:
        from src.segmentation.llm_assisted import LLMAssistedSegmentation
        from src.segmentation.segmentation_factory import create_segmentation

        cfg = SegmentationConfig(mode="llm-assisted")
        s = create_segmentation(cfg, llm_config=LLMConfig())
        assert isinstance(s, LLMAssistedSegmentation)

    def test_empty(self) -> None:
        from src.segmentation.llm_assisted import LLMAssistedSegmentation

        cfg = SegmentationConfig(mode="llm-assisted")
        s = LLMAssistedSegmentation(cfg)
        assert s.segment([], "en") == []

    def test_fallback_on_llm_error(self) -> None:
        """LLM 실패 시 advanced 결과로 폴백."""
        from src.segmentation.llm_assisted import LLMAssistedSegmentation

        words = _words(["Hello,", "world.", "How", "are", "you?"])
        cfg = SegmentationConfig(mode="llm-assisted", min_gap_s=0.04)
        s = LLMAssistedSegmentation(cfg, llm_config=LLMConfig())

        # LLM 호출을 에러로 만듦
        with patch(
            "src.segmentation.llm_client.create_llm_provider",
            side_effect=LLMSegmentationError("no api key"),
        ):
            result = s.segment(words, "en")

        assert len(result) >= 1
        assert all(isinstance(seg, TranscriptionSegment) for seg in result)

    def test_mode_name(self) -> None:
        from src.segmentation.llm_assisted import LLMAssistedSegmentation

        s = LLMAssistedSegmentation(SegmentationConfig())
        assert s.mode_name == "llm-assisted"

    def test_token_usage(self) -> None:
        from src.segmentation.llm_assisted import LLMAssistedSegmentation

        s = LLMAssistedSegmentation(SegmentationConfig())
        assert s.token_usage == {"input_tokens": 0, "output_tokens": 0}


class TestLLMResponseParsing:
    def test_valid_array(self) -> None:
        from src.segmentation.llm_assisted import _parse_llm_response

        assert _parse_llm_response("[5, 12, 18]") == [5, 12, 18]

    def test_code_block(self) -> None:
        from src.segmentation.llm_assisted import _parse_llm_response

        assert _parse_llm_response("```json\n[3, 7]\n```") == [3, 7]

    def test_extra_text(self) -> None:
        from src.segmentation.llm_assisted import _parse_llm_response

        assert _parse_llm_response("Here are the splits: [1, 2, 3]") == [1, 2, 3]

    def test_empty_array(self) -> None:
        from src.segmentation.llm_assisted import _parse_llm_response

        assert _parse_llm_response("[]") == []

    def test_invalid(self) -> None:
        from src.segmentation.llm_assisted import _parse_llm_response

        assert _parse_llm_response("no numbers here") == []

    def test_dedup_and_sort(self) -> None:
        from src.segmentation.llm_assisted import _parse_llm_response

        assert _parse_llm_response("[5, 3, 5, 1]") == [1, 3, 5]


class TestWindowing:
    def test_small_input(self) -> None:
        from src.segmentation.llm_assisted import _create_windows

        segs = [MagicMock() for _ in range(5)]
        windows = _create_windows(segs, window_size=10, overlap=2)
        assert len(windows) == 1
        assert len(windows[0]) == 5

    def test_large_input(self) -> None:
        from src.segmentation.llm_assisted import _create_windows

        segs = [MagicMock() for _ in range(25)]
        windows = _create_windows(segs, window_size=10, overlap=2)
        assert len(windows) >= 3
        # All segments covered
        all_items = set()
        for w in windows:
            all_items.update(id(s) for s in w)
        assert len(all_items) == 25


class TestBuildSegmentsFromSplits:
    def test_basic(self) -> None:
        from src.segmentation.llm_assisted import _build_segments_from_splits

        words = _words(["a", "b", "c", "d", "e"])
        segs = _build_segments_from_splits(words, [2, 4])
        assert len(segs) == 3
        assert segs[0].text == "a b"
        assert segs[1].text == "c d"
        assert segs[2].text == "e"

    def test_no_splits(self) -> None:
        from src.segmentation.llm_assisted import _build_segments_from_splits

        words = _words(["a", "b"])
        segs = _build_segments_from_splits(words, [])
        assert len(segs) == 1
        assert segs[0].text == "a b"


class TestForceConstraints:
    def test_max_duration_split(self) -> None:
        from src.segmentation.llm_assisted import _force_split_segment

        words = _words(["a", "b", "c", "d"])
        # Make duration exceed max
        words[0].start = 0.0
        words[-1].end = 10.0
        seg = TranscriptionSegment(1, "a b c d", 0.0, 10.0, 0.9, words=words)
        result = _force_split_segment(seg, max_duration_s=7.0)
        assert len(result) >= 2


# ══════════════════════════════════════════════
# 챕터 처리
# ══════════════════════════════════════════════


class TestChapterExtraction:
    def test_chapter_dataclass(self) -> None:
        from src.audio.extractor import Chapter

        ch = Chapter(index=0, title="Intro", start_s=0.0, end_s=60.0)
        assert ch.duration_s == 60.0
        assert ch.start_tc == "00:00:00.000"
        assert ch.end_tc == "00:01:00.000"

    def test_timecode_format(self) -> None:
        from src.audio.extractor import Chapter

        ch = Chapter(index=1, title="Ch2", start_s=3661.5, end_s=7200.0)
        assert ch.start_tc == "01:01:01.500"

    def test_get_chapters_no_ffprobe(self) -> None:
        from src.audio.extractor import get_chapters

        # Non-existent file → empty list (ffprobe fails gracefully)
        result = get_chapters("/nonexistent/file.mkv")
        assert result == []

    def test_get_chapters_mock(self) -> None:
        from src.audio.extractor import get_chapters

        ffprobe_json = json.dumps(
            {
                "chapters": [
                    {
                        "start_time": "0.000",
                        "end_time": "120.500",
                        "tags": {"title": "Opening"},
                    },
                    {
                        "start_time": "120.500",
                        "end_time": "300.000",
                        "tags": {"title": "Main"},
                    },
                ]
            }
        )

        mock_result = MagicMock()
        mock_result.stdout = ffprobe_json

        with (
            patch("src.audio.extractor._find_ffprobe", return_value="/usr/bin/ffprobe"),
            patch("subprocess.run", return_value=mock_result),
        ):
            chapters = get_chapters("/fake/video.mkv")

        assert len(chapters) == 2
        assert chapters[0].title == "Opening"
        assert chapters[0].start_s == 0.0
        assert chapters[0].end_s == 120.5
        assert chapters[1].title == "Main"

    def test_get_chapters_no_chapters(self) -> None:
        from src.audio.extractor import get_chapters

        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"chapters": []})

        with (
            patch("src.audio.extractor._find_ffprobe", return_value="/usr/bin/ffprobe"),
            patch("subprocess.run", return_value=mock_result),
        ):
            assert get_chapters("/fake/video.mkv") == []


# ══════════════════════════════════════════════
# 장면 감지
# ══════════════════════════════════════════════


class TestSceneDetector:
    def _segs(self, times: list[tuple[float, float]]) -> list[TranscriptionSegment]:
        return [TranscriptionSegment(i + 1, f"seg{i}", s, e, 0.9) for i, (s, e) in enumerate(times)]

    def test_gap_detection(self) -> None:
        from src.quality.scene_detector import detect_scenes_by_gap

        segs = self._segs([(0, 1), (1.5, 2.5), (10, 11), (11.5, 12.5)])
        boundaries = detect_scenes_by_gap(segs, gap_threshold_s=6.0)
        assert len(boundaries) == 1
        assert boundaries[0].source == "gap"

    def test_no_gaps(self) -> None:
        from src.quality.scene_detector import detect_scenes_by_gap

        segs = self._segs([(0, 1), (1.1, 2), (2.1, 3)])
        assert detect_scenes_by_gap(segs, gap_threshold_s=6.0) == []

    def test_empty(self) -> None:
        from src.quality.scene_detector import detect_scenes_by_gap

        assert detect_scenes_by_gap([], gap_threshold_s=6.0) == []

    def test_single_segment(self) -> None:
        from src.quality.scene_detector import detect_scenes_by_gap

        assert detect_scenes_by_gap(self._segs([(0, 1)])) == []

    def test_boundary_gap_enforcement(self) -> None:
        from src.quality.scene_detector import (
            SceneBoundary,
            ensure_scene_boundary_gaps,
        )

        segs = self._segs([(0, 5.0), (5.1, 10)])
        boundaries = [SceneBoundary(time_s=5.05)]
        result = ensure_scene_boundary_gaps(segs, boundaries, boundary_gap_s=1.0)
        # First seg end should be pulled back
        assert result[0].end <= 5.05 - 0.5 + 0.01
        # Second seg start should be pushed forward
        assert result[1].start >= 5.05 + 0.5 - 0.01

    def test_build_scenes(self) -> None:
        from src.quality.scene_detector import SceneBoundary, build_scenes

        segs = self._segs([(0, 1), (1.5, 2.5), (10, 11), (11.5, 12.5)])
        boundaries = [SceneBoundary(time_s=5.0)]
        scenes = build_scenes(segs, boundaries)
        assert len(scenes) == 2
        assert scenes[0].segment_count == 2
        assert scenes[1].segment_count == 2

    def test_pyscenedetect_available(self) -> None:
        from src.quality.scene_detector import is_pyscenedetect_available

        assert isinstance(is_pyscenedetect_available(), bool)

    def test_pyscenedetect_not_installed(self) -> None:
        from src.quality.scene_detector import (
            detect_scenes_by_video,
            is_pyscenedetect_available,
        )

        if is_pyscenedetect_available():
            pytest.skip("PySceneDetect is installed")
        assert detect_scenes_by_video("/fake.mp4") == []

    def test_min_scene_merge(self) -> None:
        from src.quality.scene_detector import detect_scenes_by_gap

        # Single segment between gaps → too small scene → merged
        segs = self._segs([(0, 1), (8, 9), (16, 17), (17.5, 18.5)])
        boundaries = detect_scenes_by_gap(segs, gap_threshold_s=6.0, min_scene_segments=2)
        # The middle single-segment scene boundary should be removed
        assert len(boundaries) <= 2


# ══════════════════════════════════════════════
# whisper-turbo 엔진
# ══════════════════════════════════════════════


class TestWhisperTurboEngine:
    def test_engine_name(self) -> None:
        from src.engines.whisper_turbo_engine import WhisperTurboEngine

        cfg = STTConfig(model="whisper-turbo")
        engine = WhisperTurboEngine(cfg)
        assert engine.engine_name == "whisper-turbo"

    def test_not_loaded_transcribe(self) -> None:
        from src.core.exceptions import TranscriptionError
        from src.engines.whisper_turbo_engine import WhisperTurboEngine

        engine = WhisperTurboEngine(STTConfig())
        with pytest.raises(TranscriptionError, match="로드되지 않았습니다"):
            engine.transcribe("/fake.wav")

    def test_factory_registration(self) -> None:
        from src.engines.engine_factory import _ENGINE_REGISTRY

        assert "whisper-turbo" in _ENGINE_REGISTRY

    def test_load_requires_faster_whisper(self) -> None:
        from src.engines.whisper_turbo_engine import WhisperTurboEngine

        engine = WhisperTurboEngine(STTConfig(device="cpu"))
        with (
            patch.dict("sys.modules", {"faster_whisper": None}),
            pytest.raises(ModelLoadError, match="faster-whisper"),
        ):
            engine.load_model()

    def test_unload_noop(self) -> None:
        from src.engines.whisper_turbo_engine import WhisperTurboEngine

        engine = WhisperTurboEngine(STTConfig())
        engine.unload_model()  # Should not raise


# ══════════════════════════════════════════════
# Parakeet 엔진
# ══════════════════════════════════════════════


class TestParakeetEngine:
    def test_engine_name(self) -> None:
        from src.engines.parakeet_engine import ParakeetEngine

        engine = ParakeetEngine(STTConfig(model="parakeet"))
        assert engine.engine_name == "parakeet"

    def test_not_loaded_transcribe(self) -> None:
        from src.core.exceptions import TranscriptionError
        from src.engines.parakeet_engine import ParakeetEngine

        engine = ParakeetEngine(STTConfig())
        with pytest.raises(TranscriptionError, match="로드되지 않았습니다"):
            engine.transcribe("/fake.wav")

    def test_factory_registration(self) -> None:
        from src.engines.engine_factory import _ENGINE_REGISTRY

        assert "parakeet" in _ENGINE_REGISTRY

    def test_nemo_not_available(self) -> None:
        from src.engines.parakeet_engine import ParakeetEngine, is_nemo_available

        if is_nemo_available():
            pytest.skip("nemo_toolkit is installed")
        engine = ParakeetEngine(STTConfig())
        with pytest.raises(ModelLoadError, match="nemo_toolkit"):
            engine.load_model()

    def test_is_nemo_available(self) -> None:
        from src.engines.parakeet_engine import is_nemo_available

        assert isinstance(is_nemo_available(), bool)

    def test_resolve_model(self) -> None:
        from src.engines.parakeet_engine import _resolve_parakeet_model

        assert _resolve_parakeet_model("parakeet-tdt-1.1b") == "nvidia/parakeet-tdt-1.1b"
        assert _resolve_parakeet_model("large-v3") == "nvidia/parakeet-tdt-1.1b"
        assert _resolve_parakeet_model("nvidia/parakeet-ctc-1.1b") == "nvidia/parakeet-ctc-1.1b"

    def test_unload_noop(self) -> None:
        from src.engines.parakeet_engine import ParakeetEngine

        engine = ParakeetEngine(STTConfig())
        engine.unload_model()  # Should not raise


# ══════════════════════════════════════════════
# LLMConfig 통합
# ══════════════════════════════════════════════


class TestLLMConfig:
    def test_defaults(self) -> None:
        cfg = LLMConfig()
        # OpenAI·Claude 는 구독만 허용: 기본은 Claude 구독, 모델은 CLI 기본값
        assert cfg.provider == "claude-code"
        assert cfg.model == ""
        assert cfg.rpm == 0
        assert cfg.timeout_s == 30.0
        assert cfg.max_retries == 3

    def test_config_load(self) -> None:
        from src.core.config import AppConfig

        config = AppConfig.load()
        assert hasattr(config, "llm")
        assert config.llm.provider == "claude-code"
