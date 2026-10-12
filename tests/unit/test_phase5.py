"""Phase 5 테스트: demucs, enhancer, 구두점 복원, extreme 세그멘테이션."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from src.audio.enhancer import (
    enhance_audio,
    is_noisereduce_available,
    normalize_loudness,
    reduce_noise_safe,
)
from src.audio.vocal_separator import (
    is_demucs_available,
    separate_vocals,
    separate_vocals_safe,
)
from src.core.config import SegmentationConfig
from src.core.exceptions import AudioEnhancementError
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.postprocess.punctuation import (
    _add_commas_en,
    _add_commas_ja,
    _add_commas_ko,
    _add_sentence_end,
    is_punctuation_model_available,
    restore_punctuation,
)
from src.segmentation.extreme import ExtremeSegmentation, is_sentence_transformers_available

if TYPE_CHECKING:
    from pathlib import Path


def _seg(text: str, idx: int = 1) -> TranscriptionSegment:
    return TranscriptionSegment(idx, text, 0.0, 1.0, 0.9)


def _words(texts: list[str], gap: float = 0.05) -> list[WordSegment]:
    result: list[WordSegment] = []
    t = 0.0
    for text in texts:
        result.append(WordSegment(text, t, t + 0.3, 0.9))
        t += 0.3 + gap
    return result


# ──────────────────────────────────────────────
# demucs 래퍼
# ──────────────────────────────────────────────


class TestDemucsWrapper:
    def test_is_available(self) -> None:
        # Just check it returns bool without error
        assert isinstance(is_demucs_available(), bool)

    def test_separate_vocals_not_installed(self) -> None:
        if is_demucs_available():
            pytest.skip("demucs is installed")
        with pytest.raises(AudioEnhancementError, match="설치되지 않았습니다"):
            separate_vocals("/nonexistent.wav")

    def test_safe_returns_none(self) -> None:
        if is_demucs_available():
            pytest.skip("demucs is installed")
        assert separate_vocals_safe("/nonexistent.wav") is None

    def test_nonexistent_file(self) -> None:
        if not is_demucs_available():
            pytest.skip("demucs not installed")
        with pytest.raises(AudioEnhancementError, match="존재하지 않습니다"):
            separate_vocals("/nonexistent.wav")


# ──────────────────────────────────────────────
# 음량 정규화 + 노이즈 억제
# ──────────────────────────────────────────────


class TestNormalizeLoudness:
    def test_nonexistent_file(self) -> None:
        with pytest.raises(AudioEnhancementError, match="존재하지 않습니다"):
            normalize_loudness("/nonexistent.wav")


class TestReduceNoise:
    def test_is_available(self) -> None:
        assert isinstance(is_noisereduce_available(), bool)

    def test_safe_returns_none_if_not_installed(self) -> None:
        if is_noisereduce_available():
            pytest.skip("noisereduce is installed")
        assert reduce_noise_safe("/dummy.wav") is None


# ──────────────────────────────────────────────
# 전처리 순서 검증
# ──────────────────────────────────────────────


class TestEnhancePipelineOrder:
    """enhance_audio가 올바른 순서로 단계를 실행하는지 검증."""

    def test_no_processing(self, tmp_path: Path) -> None:
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"\x00" * 100)
        result = enhance_audio(dummy, normalize=False, denoise=False, separate_vocals=False)
        assert result == dummy  # no-op returns original

    def test_order_with_mocks(self, tmp_path: Path) -> None:
        """단계 호출 순서를 mock으로 검증한다."""
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"\x00" * 100)
        call_order: list[str] = []

        def mock_vocals(path, **kwargs):  # type: ignore
            call_order.append("demucs")
            return None  # skip

        def mock_normalize(path, **kwargs):  # type: ignore
            call_order.append("normalize")
            out = tmp_path / "norm.wav"
            out.write_bytes(b"\x00" * 100)
            return out

        def mock_denoise(path, **kwargs):  # type: ignore
            call_order.append("denoise")
            return None  # skip

        with (
            patch("src.audio.vocal_separator.separate_vocals_safe", mock_vocals),
            patch("src.audio.enhancer.normalize_loudness", mock_normalize),
            patch("src.audio.enhancer.reduce_noise_safe", mock_denoise),
        ):
            enhance_audio(
                dummy,
                normalize=True,
                denoise=True,
                separate_vocals=True,
            )

        assert call_order == ["demucs", "normalize", "denoise"]


# ──────────────────────────────────────────────
# 구두점 복원
# ──────────────────────────────────────────────


class TestSentenceEnd:
    def test_english(self) -> None:
        assert _add_sentence_end("Hello world", "en") == "Hello world."

    def test_already_has(self) -> None:
        assert _add_sentence_end("Hello!", "en") == "Hello!"

    def test_japanese(self) -> None:
        assert _add_sentence_end("こんにちは", "ja") == "こんにちは。"

    def test_chinese(self) -> None:
        assert _add_sentence_end("你好", "zh") == "你好。"

    def test_korean(self) -> None:
        assert _add_sentence_end("안녕", "ko") == "안녕."

    def test_empty(self) -> None:
        assert _add_sentence_end("", "en") == ""


class TestEnglishCommas:
    def test_conjunction(self) -> None:
        assert "world, but" in _add_commas_en("Hello world but goodbye")

    def test_short_no_change(self) -> None:
        assert _add_commas_en("go but") == "go but"

    def test_already_has_comma(self) -> None:
        result = _add_commas_en("Hello, but goodbye")
        assert result.count(",") == 1


class TestKoreanCommas:
    def test_ending(self) -> None:
        assert "먹고," in _add_commas_ko("밥을 먹고 학교에 갔다")

    def test_conjunction(self) -> None:
        result = _add_commas_ko("좋다 그리고 나쁘다")
        assert "좋다, 그리고" in result
        assert "그리고," not in result  # no double comma on conjunction


class TestJapaneseCommas:
    def test_particle(self) -> None:
        assert "、" in _add_commas_ja("食べたがおいしかった")


class TestRestorePunctuation:
    def test_english_pipeline(self) -> None:
        segs = [_seg("Hello world"), _seg("Already done.")]
        result = restore_punctuation(segs, "en")
        assert result[0].text.endswith(".")
        assert result[1].text == "Already done."

    def test_empty(self) -> None:
        assert restore_punctuation([], "en") == []

    def test_model_fallback(self) -> None:
        result = restore_punctuation([_seg("hello")], "en", use_model=True)
        assert result[0].text.endswith(".")

    def test_model_available(self) -> None:
        assert isinstance(is_punctuation_model_available(), bool)


# ──────────────────────────────────────────────
# extreme 세그멘테이션
# ──────────────────────────────────────────────


class TestExtremeSegmentation:
    def test_factory(self) -> None:
        from src.segmentation.segmentation_factory import create_segmentation

        s = create_segmentation(SegmentationConfig(mode="extreme"))
        assert isinstance(s, ExtremeSegmentation)

    def test_fallback_to_advanced(self) -> None:
        """sentence-transformers 미설치 시 advanced 결과 반환."""
        words = _words(["Hello,", "world."])
        words2 = _words(["Another", "sentence."])
        # Add gap between groups
        for w in words2:
            w.start += 2.0
            w.end += 2.0
        all_words = words + words2

        cfg = SegmentationConfig(mode="extreme", min_gap_s=0.04)
        extreme = ExtremeSegmentation(cfg)
        segments = extreme.segment(all_words, "en")
        assert len(segments) >= 1
        assert all(len(s.words) > 0 for s in segments if s.text.strip())

    def test_empty(self) -> None:
        cfg = SegmentationConfig(mode="extreme")
        assert ExtremeSegmentation(cfg).segment([], "en") == []

    def test_single_word(self) -> None:
        words = [WordSegment("Hi.", 0.0, 0.5, 0.9)]
        cfg = SegmentationConfig(mode="extreme", min_gap_s=0.04)
        segs = ExtremeSegmentation(cfg).segment(words, "en")
        assert len(segs) == 1

    def test_is_available(self) -> None:
        assert isinstance(is_sentence_transformers_available(), bool)


class TestSaveVocals:
    """demucs 보컬 저장: torchaudio.save 없이 16kHz mono 16-bit WAV (저장 백엔드 없음 대응)."""

    def test_writes_pipeline_wav_without_torchaudio_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import math
        import wave

        torch = pytest.importorskip("torch")
        torchaudio = pytest.importorskip("torchaudio")
        from src.audio.vocal_separator import _save_audio_tensor

        def no_backend(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Couldn't find appropriate backend")

        monkeypatch.setattr(torchaudio, "save", no_backend)
        rate = 44_100
        t = torch.arange(rate) / rate
        stereo = (
            torch.stack([torch.sin(2 * math.pi * 440 * t), torch.sin(2 * math.pi * 440 * t)]) * 0.5
        )
        out = tmp_path / "vocals.wav"

        _save_audio_tensor(stereo, out, rate)

        with wave.open(str(out)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16_000)
            assert abs(wav.getnframes() - 16_000) <= 1
            samples = wav.readframes(wav.getnframes())
        peak = max(
            abs(int.from_bytes(samples[i : i + 2], "little", signed=True))
            for i in range(0, 4000, 2)
        )
        assert 10_000 < peak < 20_000  # 0.5 진폭 ≈ 16383
