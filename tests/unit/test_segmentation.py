"""세그멘테이션 단위 테스트 (basic, advanced, CPS, min_display_guard)."""

from __future__ import annotations

import pytest

from src.core.config import SegmentationConfig
from src.core.exceptions import ConfigError
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.segmentation.advanced import (
    _BASE_SPLIT_THRESHOLD,
    AdvancedSegmentation,
    _clause_score,
    _dynamic_threshold,
    _pause_score,
)
from src.segmentation.base import SegmentationStrategy
from src.segmentation.basic import BasicSegmentation
from src.segmentation.min_display_guard import enforce_min_duration
from src.segmentation.segmentation_factory import create_segmentation, get_available_modes
from src.segmentation.speed_optimizer import (
    get_cps_for_language,
    optimize_cps,
)
from src.segmentation.word_boundary_snapper import (
    snap_segments_to_word_boundaries,
    verify_no_overlap,
)


def _make_words(texts: list[str], gap: float = 0.05, word_dur: float = 0.3) -> list[WordSegment]:
    """테스트용 단어 목록 생성."""
    words: list[WordSegment] = []
    t = 0.0
    for text in texts:
        words.append(WordSegment(text, t, t + word_dur, 0.9))
        t += word_dur + gap
    return words


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────


class TestSegmentationFactory:
    def test_available_modes(self) -> None:
        modes = get_available_modes()
        assert "basic" in modes
        assert "advanced" in modes

    def test_create_basic(self) -> None:
        s = create_segmentation(SegmentationConfig(mode="basic"))
        assert isinstance(s, BasicSegmentation)

    def test_create_advanced(self) -> None:
        s = create_segmentation(SegmentationConfig(mode="advanced"))
        assert isinstance(s, AdvancedSegmentation)

    def test_unknown_mode(self) -> None:
        with pytest.raises(ConfigError):
            create_segmentation(SegmentationConfig(mode="nonexistent"))

    def test_abc_not_instantiable(self) -> None:
        with pytest.raises(TypeError):
            SegmentationStrategy(SegmentationConfig())  # type: ignore[abstract]


# ──────────────────────────────────────────────
# Basic segmentation
# ──────────────────────────────────────────────


class TestBasicSegmentation:
    def test_simple_sentences(self) -> None:
        words = _make_words(["Hello,", "world.", "This", "is", "a", "test."])
        cfg = SegmentationConfig(mode="basic", min_gap_s=0.04)
        segs = BasicSegmentation(cfg).segment(words, "en")
        assert len(segs) >= 2
        assert all(len(s.words) > 0 for s in segs)

    def test_empty_input(self) -> None:
        cfg = SegmentationConfig(mode="basic")
        assert BasicSegmentation(cfg).segment([], "en") == []

    def test_single_word(self) -> None:
        words = _make_words(["Hello."])
        cfg = SegmentationConfig(mode="basic", min_gap_s=0.04)
        segs = BasicSegmentation(cfg).segment(words, "en")
        assert len(segs) == 1

    def test_no_overlap(self) -> None:
        words = _make_words(["First.", "Second.", "Third.", "Fourth."])
        cfg = SegmentationConfig(mode="basic", min_gap_s=0.04)
        segs = BasicSegmentation(cfg).segment(words, "en")
        violations = verify_no_overlap(segs, 0.04)
        assert len(violations) == 0


# ──────────────────────────────────────────────
# Advanced segmentation
# ──────────────────────────────────────────────


class TestAdvancedScoring:
    def test_pause_score_strong(self) -> None:
        w1 = WordSegment("a", 0.0, 0.5, 0.9)
        w2 = WordSegment("b", 1.5, 2.0, 0.9)  # gap=1.0
        assert _pause_score(w1, w2) == 3.0

    def test_pause_score_medium(self) -> None:
        w1 = WordSegment("a", 0.0, 0.5, 0.9)
        w2 = WordSegment("b", 1.1, 1.5, 0.9)  # gap=0.6
        assert _pause_score(w1, w2) == 1.0

    def test_pause_score_none(self) -> None:
        w1 = WordSegment("a", 0.0, 0.5, 0.9)
        w2 = WordSegment("b", 0.55, 1.0, 0.9)  # gap=0.05
        assert _pause_score(w1, w2) == 0.0

    def test_clause_score_sentence_end(self) -> None:
        assert _clause_score(WordSegment("hello.", 0, 1, 0.9)) == 2.5

    def test_clause_score_comma(self) -> None:
        assert _clause_score(WordSegment("hello,", 0, 1, 0.9)) == 1.0

    def test_clause_score_none(self) -> None:
        assert _clause_score(WordSegment("hello", 0, 1, 0.9)) == 0.0

    def test_clause_score_cjk(self) -> None:
        assert _clause_score(WordSegment("。", 0, 1, 0.9)) == 2.5


class TestDynamicThreshold:
    def test_before_start(self) -> None:
        cfg = SegmentationConfig(
            dynamic_threshold=True, dynamic_threshold_start_s=3.0, dynamic_threshold_min_ratio=0.4
        )
        assert _dynamic_threshold(1.0, cfg) == _BASE_SPLIT_THRESHOLD

    def test_at_start(self) -> None:
        cfg = SegmentationConfig(
            dynamic_threshold=True, dynamic_threshold_start_s=3.0, dynamic_threshold_min_ratio=0.4
        )
        assert _dynamic_threshold(3.0, cfg) == _BASE_SPLIT_THRESHOLD

    def test_at_end(self) -> None:
        cfg = SegmentationConfig(
            dynamic_threshold=True, dynamic_threshold_start_s=3.0, dynamic_threshold_min_ratio=0.4
        )
        t = _dynamic_threshold(13.0, cfg)
        assert t == pytest.approx(_BASE_SPLIT_THRESHOLD * 0.4)

    def test_monotonic_decrease(self) -> None:
        cfg = SegmentationConfig(
            dynamic_threshold=True, dynamic_threshold_start_s=3.0, dynamic_threshold_min_ratio=0.4
        )
        values = [_dynamic_threshold(t, cfg) for t in range(0, 20)]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]

    def test_disabled(self) -> None:
        cfg = SegmentationConfig(dynamic_threshold=False)
        assert _dynamic_threshold(20.0, cfg) == _BASE_SPLIT_THRESHOLD


class TestAdvancedSegmentation:
    def test_natural_splits(self) -> None:
        # Words with a strong pause between sentences
        words: list[WordSegment] = []
        t = 0.0
        for w in ["Hello,", "world."]:
            words.append(WordSegment(w, t, t + 0.3, 0.9))
            t += 0.35
        t += 0.9  # strong pause
        for w in ["Another", "sentence."]:
            words.append(WordSegment(w, t, t + 0.3, 0.9))
            t += 0.35

        cfg = SegmentationConfig(mode="advanced", min_gap_s=0.04)
        segs = AdvancedSegmentation(cfg).segment(words, "en")
        assert len(segs) >= 2

    def test_hard_cap(self) -> None:
        words = _make_words([f"w{i}" for i in range(50)], gap=0.01)
        cfg = SegmentationConfig(mode="advanced", max_duration_s=2.0, min_gap_s=0.04)
        segs = AdvancedSegmentation(cfg).segment(words, "en")
        for s in segs:
            assert s.duration <= 2.2  # small tolerance

    def test_empty(self) -> None:
        cfg = SegmentationConfig(mode="advanced")
        assert AdvancedSegmentation(cfg).segment([], "en") == []


# ──────────────────────────────────────────────
# CPS optimizer
# ──────────────────────────────────────────────


class TestCPSOptimizer:
    def test_language_defaults(self) -> None:
        cfg = SegmentationConfig(max_cps=None)
        assert get_cps_for_language("en", cfg) == 17.0
        assert get_cps_for_language("ko", cfg) == 9.0
        assert get_cps_for_language("ja", cfg) == 7.0
        assert get_cps_for_language("zh", cfg) == 5.0
        assert get_cps_for_language("fr", cfg) == 17.0

    def test_explicit_override(self) -> None:
        cfg = SegmentationConfig(max_cps=12.0)
        assert get_cps_for_language("en", cfg) == 12.0

    def test_no_split_within_limit(self) -> None:
        words = _make_words(["Short", "text."], word_dur=0.5)
        seg = TranscriptionSegment(
            1, "Short text.", words[0].start, words[-1].end, 0.9, words=words
        )
        cfg = SegmentationConfig(max_cps=None, min_gap_s=0.04, min_duration_s=0.5)
        result = optimize_cps([seg], "en", cfg)
        assert len(result) == 1

    def test_split_when_exceeded(self) -> None:
        # Very fast: many chars in short time, 문장 부호 자리에서 나눈다
        texts = ["word"] * 9 + ["word."] + ["word"] * 10
        words = _make_words(texts, gap=0.01, word_dur=0.02)
        text = " ".join(w.word for w in words)
        seg = TranscriptionSegment(1, text, words[0].start, words[-1].end, 0.9, words=words)
        cfg = SegmentationConfig(max_cps=None, min_gap_s=0.04, min_duration_s=0.01)
        result = optimize_cps([seg], "en", cfg)
        assert len(result) >= 2
        assert result[0].text.endswith("word.")

    def test_no_split_without_punctuation(self) -> None:
        """2026-10-10 변경 (REQUIREMENT 6.5): 문장 부호가 없는 자리는 나누지 않는다 (읽기 시간 확보가 늘린다)."""
        words = _make_words(["word"] * 20, gap=0.01, word_dur=0.02)
        text = " ".join(w.word for w in words)
        seg = TranscriptionSegment(1, text, words[0].start, words[-1].end, 0.9, words=words)
        cfg = SegmentationConfig(max_cps=None, min_gap_s=0.04, min_duration_s=0.01)
        assert len(optimize_cps([seg], "en", cfg)) == 1

    def test_empty(self) -> None:
        cfg = SegmentationConfig()
        assert optimize_cps([], "en", cfg) == []


# ──────────────────────────────────────────────
# Min display guard
# ──────────────────────────────────────────────


class TestMinDisplayGuard:
    def test_no_short_segments(self) -> None:
        segs = [
            TranscriptionSegment(1, "Long.", 0.0, 2.0),
            TranscriptionSegment(2, "Also long.", 3.0, 5.0),
        ]
        cfg = SegmentationConfig(min_duration_s=1.0)
        result = enforce_min_duration(segs, cfg)
        assert len(result) == 2

    def test_short_merged(self) -> None:
        segs = [
            TranscriptionSegment(1, "Long.", 0.0, 2.0),
            TranscriptionSegment(2, "Hi.", 2.5, 2.8),  # 0.3s
            TranscriptionSegment(3, "End.", 3.5, 5.5),
        ]
        cfg = SegmentationConfig(min_duration_s=1.0)
        result = enforce_min_duration(segs, cfg)
        assert len(result) == 2

    def test_same_speaker_priority(self) -> None:
        segs = [
            TranscriptionSegment(1, "A.", 0.0, 2.0, speaker="X"),
            TranscriptionSegment(2, "B.", 2.5, 2.8, speaker="X"),
            TranscriptionSegment(3, "C.", 3.5, 5.5, speaker="Y"),
        ]
        cfg = SegmentationConfig(min_duration_s=1.0)
        result = enforce_min_duration(segs, cfg)
        assert len(result) == 2
        # Should merge with same-speaker neighbor
        assert result[0].speaker == "X"

    def test_multi_speaker_separator(self) -> None:
        segs = [
            TranscriptionSegment(1, "A.", 0.0, 2.0, speaker="X"),
            TranscriptionSegment(2, "B.", 2.5, 2.8, speaker="Y"),
        ]
        cfg = SegmentationConfig(min_duration_s=1.0, overlap_text_separator=" / ")
        result = enforce_min_duration(segs, cfg)
        assert len(result) == 1
        assert " / " in result[0].text
        assert isinstance(result[0].speaker, list)

    def test_single_segment_unchanged(self) -> None:
        segs = [TranscriptionSegment(1, "Hi.", 0.0, 0.3)]
        cfg = SegmentationConfig(min_duration_s=1.0)
        result = enforce_min_duration(segs, cfg)
        assert len(result) == 1

    def test_empty(self) -> None:
        cfg = SegmentationConfig(min_duration_s=1.0)
        assert enforce_min_duration([], cfg) == []

    def test_reindex(self) -> None:
        segs = [
            TranscriptionSegment(1, "A.", 0.0, 0.3),
            TranscriptionSegment(2, "B.", 0.5, 0.8),
            TranscriptionSegment(3, "C.", 1.0, 3.0),
        ]
        cfg = SegmentationConfig(min_duration_s=1.0)
        result = enforce_min_duration(segs, cfg)
        for i, s in enumerate(result, 1):
            assert s.index == i


# ──────────────────────────────────────────────
# Word boundary snapper
# ──────────────────────────────────────────────


class TestWordBoundarySnapper:
    def test_snap_to_word_boundary(self) -> None:
        words1 = [WordSegment("Hello", 0.5, 0.9, 0.9), WordSegment("world", 1.0, 1.4, 0.9)]
        words2 = [WordSegment("Test", 2.0, 2.3, 0.9)]
        segs = [
            TranscriptionSegment(1, "Hello world", 0.0, 2.0, words=words1),
            TranscriptionSegment(2, "Test", 1.5, 3.0, words=words2),
        ]
        result = snap_segments_to_word_boundaries(segs, 0.04)
        assert result[0].start == 0.5
        assert result[0].end == 1.4
        assert result[1].start >= 1.44

    def test_verify_no_overlap_clean(self) -> None:
        segs = [
            TranscriptionSegment(1, "A", 0.0, 1.0),
            TranscriptionSegment(2, "B", 1.1, 2.0),
        ]
        assert verify_no_overlap(segs, 0.04) == []

    def test_verify_no_overlap_violation(self) -> None:
        segs = [
            TranscriptionSegment(1, "A", 0.0, 1.0),
            TranscriptionSegment(2, "B", 1.02, 2.0),  # gap=0.02 < 0.04
        ]
        violations = verify_no_overlap(segs, 0.04)
        assert len(violations) == 1

    def test_empty(self) -> None:
        assert snap_segments_to_word_boundaries([], 0.04) == []
