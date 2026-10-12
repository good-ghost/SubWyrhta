"""후처리 (노이즈 필터 + 텍스트 정규화) 단위 테스트."""

from __future__ import annotations

from src.core.config import NoiseFilterConfig
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.postprocess.noise_filter import (
    _build_filler_set,
    _compile_patterns,
    _filter_by_confidence,
    _remove_custom_patterns,
    _remove_fillers,
    _remove_repeats,
    filter_noise,
)
from src.postprocess.normalizer import _capitalize_sentence_start, normalize_segments


def _w(text: str, conf: float = 0.9) -> WordSegment:
    return WordSegment(text, 0.0, 0.5, conf)


def _seg(words: list[WordSegment], idx: int = 1) -> TranscriptionSegment:
    text = " ".join(w.word for w in words)
    start = words[0].start if words else 0.0
    end = words[-1].end if words else 0.0
    return TranscriptionSegment(idx, text, start, end, 0.9, words=words)


# ──────────────────────────────────────────────
# 노이즈 필터
# ──────────────────────────────────────────────


class TestFillerWords:
    def test_english_fillers(self) -> None:
        fillers = _build_filler_set(["en"])
        assert "uh" in fillers
        assert "um" in fillers

    def test_multi_language(self) -> None:
        fillers = _build_filler_set(["en", "ko", "ja"])
        assert "uh" in fillers
        assert "어" in fillers
        assert "あー" in fillers

    def test_remove_fillers(self) -> None:
        words = [_w("Hello"), _w("uh"), _w("world")]
        fillers = _build_filler_set(["en"])
        result, count = _remove_fillers(words, fillers)
        assert count == 1
        assert len(result) == 2

    def test_punctuated_filler(self) -> None:
        words = [_w("um,"), _w("hello")]
        fillers = _build_filler_set(["en"])
        _result, count = _remove_fillers(words, fillers)
        assert count == 1


class TestRepeatDetection:
    def test_three_repeats(self) -> None:
        words = [_w("the")] * 4 + [_w("dog")]
        result, count = _remove_repeats(words)
        assert count == 3
        assert len(result) == 2

    def test_two_repeats_ok(self) -> None:
        words = [_w("the"), _w("the"), _w("dog")]
        _result, count = _remove_repeats(words)
        assert count == 0

    def test_multiple_groups(self) -> None:
        words = [_w("a")] * 3 + [_w("b")] * 4
        _result, count = _remove_repeats(words)
        assert count == 5  # 2 from 'a', 3 from 'b'


class TestConfidenceFilter:
    def test_low_confidence_removed(self) -> None:
        words = [_w("good", 0.9), _w("bad", 0.3), _w("ok", 0.8)]
        result, count = _filter_by_confidence(words, 0.7)
        assert count == 1
        assert len(result) == 2

    def test_zero_confidence_kept(self) -> None:
        words = [_w("no_conf", 0.0)]
        _result, count = _filter_by_confidence(words, 0.7)
        assert count == 0


class TestCustomPatterns:
    def test_match_and_remove(self) -> None:
        patterns = _compile_patterns([r"\[.*\]", r"♪.*"])
        words = [_w("[Music]"), _w("Hello"), _w("♪")]
        _result, count = _remove_custom_patterns(words, patterns)
        assert count == 2

    def test_invalid_pattern_ignored(self) -> None:
        patterns = _compile_patterns(["[invalid"])
        assert len(patterns) == 0

    def test_no_patterns(self) -> None:
        words = [_w("Hello")]
        _result, count = _remove_custom_patterns(words, [])
        assert count == 0


class TestFilterNoisePipeline:
    def test_disabled(self) -> None:
        segs = [_seg([_w("uh"), _w("hello")])]
        result = filter_noise(segs, NoiseFilterConfig(enabled=False))
        assert len(result) == 1
        assert "uh" in result[0].text

    def test_empty_segment_removed(self) -> None:
        segs = [_seg([_w("um")])]
        result = filter_noise(segs, NoiseFilterConfig(enabled=True, languages=["en"]))
        assert len(result) == 0

    def test_reindex(self) -> None:
        segs = [
            _seg([_w("Hello")], 1),
            _seg([_w("um")], 2),
            _seg([_w("World")], 3),
        ]
        result = filter_noise(segs, NoiseFilterConfig(enabled=True, languages=["en"]))
        assert len(result) == 2
        assert result[0].index == 1
        assert result[1].index == 2

    def test_empty_input(self) -> None:
        assert filter_noise([], NoiseFilterConfig()) == []


# ──────────────────────────────────────────────
# 텍스트 정규화
# ──────────────────────────────────────────────


class TestNormalizer:
    def test_double_spaces(self) -> None:
        segs = [TranscriptionSegment(1, "hello  world", 0.0, 1.0)]
        result = normalize_segments(segs)
        assert result[0].text == "Hello world"

    def test_strip(self) -> None:
        segs = [TranscriptionSegment(1, "  hello  ", 0.0, 1.0)]
        result = normalize_segments(segs)
        assert result[0].text == "Hello"

    def test_capitalize(self) -> None:
        assert _capitalize_sentence_start("hello. world") == "Hello. World"
        assert _capitalize_sentence_start("") == ""

    def test_empty_removal(self) -> None:
        segs = [
            TranscriptionSegment(1, "keep", 0.0, 1.0),
            TranscriptionSegment(2, "", 2.0, 3.0),
            TranscriptionSegment(3, "   ", 4.0, 5.0),
        ]
        result = normalize_segments(segs)
        assert len(result) == 1
        assert result[0].index == 1

    def test_empty_input(self) -> None:
        assert normalize_segments([]) == []


class TestLowConfidenceSetting:
    """저신뢰도 단어: 기본은 남기고, remove_low_confidence 일 때만 지운다."""

    def _segment(self) -> TranscriptionSegment:
        words = [
            WordSegment(word="Welcome", start=0.0, end=0.5, confidence=0.4),
            WordSegment(word="home", start=0.6, end=1.0, confidence=0.95),
        ]
        return TranscriptionSegment(index=1, text="Welcome home", start=0.0, end=1.0, words=words)

    def test_default_keeps_low_confidence_words(self) -> None:
        result = filter_noise([self._segment()], NoiseFilterConfig())
        assert result[0].text == "Welcome home"

    def test_remove_flag_drops_low_confidence_words(self) -> None:
        result = filter_noise([self._segment()], NoiseFilterConfig(remove_low_confidence=True))
        assert result[0].text == "home"
