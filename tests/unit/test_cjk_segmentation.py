"""일본어(글자 단위 단어) 자막 과분할 회귀 테스트.

WhisperX 는 일본어·중국어를 글자 단위로 정렬해 "단어" 하나가 한 글자다. 2026-10-09 작업 #1
(2시간 일본어, 화자 분리)의 자막 2124개 중 81% 가 1초 미만이었고 글자마다 공백이 들어갔다.
"""

from __future__ import annotations

import pytest

from src.core.config import SegmentationConfig
from src.core.text import join_words
from src.diarization.diarizer import DiarizationResult, SpeakerTurn
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.segmentation.advanced import AdvancedSegmentation
from src.segmentation.min_display_guard import enforce_min_duration
from src.segmentation.speaker_splitter import split_by_speaker
from src.segmentation.speed_optimizer import optimize_cps

# 일본어 발화 (작업 #1 출력의 실제 문장), 초당 약 10글자 — ja 기본 max_cps 7 을 넘는다
_SENTENCE = "お気に召していただけると思います。本日はご予約ですね。"
_CHAR_SECONDS = 0.1


def _char_words(text: str, start: float = 0.0) -> list[WordSegment]:
    """글자마다 단어 하나 (WhisperX 일본어 정렬과 같은 모양)."""
    words = []
    t = start
    for ch in text:
        words.append(WordSegment(ch, t, t + _CHAR_SECONDS * 0.8, 0.9))
        t += _CHAR_SECONDS
    return words


def _segment(words: list[WordSegment], speaker: str | None = None) -> TranscriptionSegment:
    return TranscriptionSegment(
        1, join_words(w.word for w in words), words[0].start, words[-1].end, 0.9, speaker, words
    )


class TestJoinWords:
    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            (["も", "し", "も", "し"], "もしもし"),
            (["テ", "、", "レ", "ッ", "ド", "。"], "テ、レッド。"),
            (["本日", "は"], "本日は"),
            (["これは", "AI", "です"], "これはAIです"),
            (["Hello", "world."], "Hello world."),
            (["안녕", "하세요"], "안녕 하세요"),
            ([" hello ", "", "  ", "there"], "hello there"),
            # 전각 라틴 문자(U+FF21~)도 공백 없이 잇는다
            (["\uff21\uff22\uff23", "テスト"], "\uff21\uff22\uff23テスト"),
            ([], ""),
        ],
    )
    def test_joins_by_script(self, parts: list[str], expected: str) -> None:
        assert join_words(parts) == expected


class TestCpsOptimizerJapanese:
    def test_uniform_speech_is_not_shredded(self) -> None:
        """말 속도가 고르면 나눠도 CPS 가 그대로다: 한 글자 자막으로 쪼개지 않는다."""
        words = _char_words(_SENTENCE)
        seg = _segment(words)
        cfg = SegmentationConfig(min_duration_s=1.0)

        result = optimize_cps([seg], "ja", cfg)

        assert all(s.duration >= cfg.min_duration_s for s in result)
        assert all(len(s.words) > 1 for s in result)

    def test_never_splits_below_min_duration(self) -> None:
        """분할점이 없으면 min_duration_s 를 무시하고 강제로 나누지 않는다 (REQUIREMENT 6.5)."""
        words = _char_words("ありがとう")  # 0.5초, ja CPS 10
        seg = _segment(words)

        result = optimize_cps([seg], "ja", SegmentationConfig(min_duration_s=1.0))

        assert len(result) == 1
        assert result[0].text == "ありがとう"

    def test_text_has_no_spaces(self) -> None:
        words = _char_words(_SENTENCE * 2)
        result = optimize_cps([_segment(words)], "ja", SegmentationConfig(min_duration_s=1.0))
        assert all(" " not in s.text for s in result)


class TestMinDisplayGuardMany:
    def test_merges_every_short_segment(self) -> None:
        """짧은 자막이 많아도 모두 병합한다 (예전에는 50번에서 멈췄다)."""
        words = _char_words(_SENTENCE * 8)
        shredded = [
            TranscriptionSegment(i + 1, w.word, w.start, w.end, 0.9, words=[w])
            for i, w in enumerate(words)
        ]
        cfg = SegmentationConfig(min_duration_s=1.0)

        result = enforce_min_duration(shredded, cfg)

        assert len(shredded) > 200
        assert all(s.duration >= cfg.min_duration_s for s in result)
        # 같은 화자끼리 합친 텍스트에는 공백이 없다
        assert join_words(s.text for s in result) == _SENTENCE * 8

    def test_single_segment_stays(self) -> None:
        seg = TranscriptionSegment(1, "は", 0.0, 0.1, 0.9)
        assert enforce_min_duration([seg], SegmentationConfig(min_duration_s=1.0)) == [seg]


class TestSpeakerSplitJapanese:
    def test_words_between_turns_take_nearest_speaker(self) -> None:
        """화자 구간 사이 틈의 글자도 화자를 받아, 같은 화자 말이 갈라지거나 ` / ` 로 이어지지 않는다."""
        words = _char_words(_SENTENCE)
        # 같은 화자 A 의 구간이 0.6~0.9초 틈으로 끊겨 있다
        diar = DiarizationResult(
            turns=[SpeakerTurn("A", 0.0, 0.6), SpeakerTurn("A", 0.9, 3.0)], num_speakers=1
        )
        cfg = SegmentationConfig(speaker_split=True, overlap_text_separator=" / ")

        result = split_by_speaker([_segment(words)], diar, cfg)

        assert len(result) == 1
        assert result[0].speaker == "A"
        assert result[0].text == _SENTENCE

    def test_overlap_keeps_separator_only_between_speakers(self) -> None:
        """겹침 병합의 구분자는 다른 화자 사이에만 들어간다 (REQUIREMENT 6.6 케이스 3)."""
        a = [WordSegment("は", 0.0, 0.4, 0.9), WordSegment("い", 0.3, 0.7, 0.9)]
        b = [WordSegment("え", 0.6, 1.0, 0.9)]
        a2 = [WordSegment("そ", 0.9, 1.3, 0.9), WordSegment("う", 1.3, 1.6, 0.9)]
        diar = DiarizationResult(
            turns=[
                SpeakerTurn("A", 0.0, 0.75),
                SpeakerTurn("B", 0.75, 1.05),
                SpeakerTurn("A", 1.05, 2.0),
            ],
            num_speakers=2,
        )
        cfg = SegmentationConfig(speaker_split=True, overlap_text_separator=" / ")

        result = split_by_speaker([_segment(a + b + a2)], diar, cfg)

        assert len(result) == 1
        assert result[0].text == "はい / え / そう"
        assert result[0].speaker == ["A", "B"]


class TestEndToEndJapanese:
    def test_advanced_then_cps_then_guard(self) -> None:
        """파이프라인 순서(advanced → CPS → 최소 표시 시간)로 일본어 자막이 1초 미만으로 쪼개지지 않는다."""
        cfg = SegmentationConfig(min_duration_s=1.0)
        words = _char_words(_SENTENCE * 6)

        segments = AdvancedSegmentation(cfg).segment(words, "ja")
        segments = optimize_cps(segments, "ja", cfg)
        segments = enforce_min_duration(segments, cfg)

        assert all(s.duration >= cfg.min_duration_s for s in segments)
        assert all(" " not in s.text for s in segments)
        assert "".join(s.text for s in segments) == _SENTENCE * 6


class TestJapanesePhrases:
    def test_groups_characters_into_phrases(self) -> None:
        from src.postprocess.word_units import group_words_into_phrases

        words = _char_words("ご視聴ありがとうございました")
        phrases = group_words_into_phrases(words, "ja")

        assert [w.word for w in phrases] == ["ご視聴", "ありがとうございました"]
        assert phrases[0].start == words[0].start
        assert phrases[1].end == words[-1].end

    def test_particles_and_auxiliaries_attach_to_previous(self) -> None:
        from src.postprocess.word_units import group_words_into_phrases

        phrases = group_words_into_phrases(_char_words("指も入れてみましょうか"), "ja")
        assert [w.word for w in phrases] == ["指も", "入れてみましょうか"]

    def test_other_languages_unchanged(self) -> None:
        from src.postprocess.word_units import group_words_into_phrases

        words = [WordSegment("Hello", 0.0, 0.4, 0.9), WordSegment("world", 0.5, 0.9, 0.9)]
        assert group_words_into_phrases(words, "en") == words

    def test_without_janome_keeps_characters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.postprocess import word_units

        monkeypatch.setattr(word_units, "_japanese_tokenizer", lambda: None)
        words = _char_words("ありがとう")
        assert word_units.group_words_into_phrases(words, "ja") == words

    def test_phrases_prevent_mid_word_split(self) -> None:
        """문절로 묶으면 자막이 단어 중간(`ございまし` / `た`)에서 나뉘지 않는다."""
        from src.postprocess.word_units import group_words_into_phrases

        cfg = SegmentationConfig(min_duration_s=1.0, max_duration_s=2.0, force_split_after_s=1.5)
        text = "ご視聴ありがとうございました" * 3
        phrases = group_words_into_phrases(_char_words(text), "ja")

        segments = AdvancedSegmentation(cfg).segment(phrases, "ja")

        assert len(segments) > 1
        assert all(s.text.endswith(("視聴", "ました")) for s in segments)


class TestClampWordDurations:
    def test_stretched_character_is_cut(self) -> None:
        from src.postprocess.word_units import clamp_word_durations

        words = [WordSegment("き", 10.0, 18.9, 0.5), WordSegment("あー", 20.0, 20.8, 0.9)]
        assert clamp_word_durations(words) == 1
        assert words[0].start == 10.0
        assert words[0].end == pytest.approx(11.0)  # 한 글자 상한 1초
        assert words[1].end == pytest.approx(20.8)  # 상한 안이면 그대로

    def test_cap_grows_with_length(self) -> None:
        from src.postprocess.word_units import clamp_word_durations

        word = WordSegment("ありがとうございました", 0.0, 9.0, 0.9)  # 11글자 → 상한 5.5초
        clamp_word_durations([word])
        assert word.end == pytest.approx(5.5)


class TestMinDisplayGuardFarNeighbors:
    def test_does_not_merge_across_long_silence(self) -> None:
        """병합하면 max_duration_s 를 넘으면(10초 무음 너머) 병합하지 않고 빈 시간으로 늘린다."""
        segs = [
            TranscriptionSegment(1, "確認してみましょう", 33.49, 34.45, 0.9),
            TranscriptionSegment(2, "どうぞ", 44.03, 45.11, 0.9),
        ]
        cfg = SegmentationConfig(min_duration_s=1.0, max_duration_s=7.0)

        result = enforce_min_duration(segs, cfg)

        assert [s.text for s in result] == ["確認してみましょう", "どうぞ"]
        assert result[0].duration == pytest.approx(1.0)
        assert result[0].start == pytest.approx(33.49)  # 뒤쪽 빈 시간으로 늘렸다

    def test_extends_backwards_when_next_is_close(self) -> None:
        segs = [
            TranscriptionSegment(1, "前", 0.0, 1.5, 0.9),
            TranscriptionSegment(2, "は", 5.0, 5.2, 0.9),
            # 병합하면 5.0~12.5 = 7.5초로 max_duration_s 를 넘는다
            TranscriptionSegment(3, "次の長い字幕です", 5.3, 12.5, 0.9),
        ]
        cfg = SegmentationConfig(min_duration_s=1.0, max_duration_s=7.0, min_gap_s=0.04)

        result = enforce_min_duration(segs, cfg)

        assert len(result) == 3
        # 뒤로 0.06초(다음 자막과 min_gap_s 는 둔다), 모자란 만큼 앞으로
        assert result[1].end == pytest.approx(5.26)
        assert result[1].duration == pytest.approx(1.0)

    def test_merges_towards_shorter_gap(self) -> None:
        """같은 화자 양쪽이면 gap 이 짧은 쪽과 병합한다 (REQUIREMENT 6.7 케이스 A)."""
        segs = [
            TranscriptionSegment(1, "前の文", 0.0, 2.0, 0.9),
            TranscriptionSegment(2, "ね", 2.9, 3.1, 0.9),
            TranscriptionSegment(3, "次の文", 3.2, 4.0, 0.9),
        ]
        result = enforce_min_duration(segs, SegmentationConfig(min_duration_s=1.0))
        assert [s.text for s in result] == ["前の文", "ね次の文"]
