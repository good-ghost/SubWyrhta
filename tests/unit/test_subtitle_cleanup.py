"""자막 텍스트 정리 테스트: 지어낸 문구·신음 제거, 발화 구분, 앞에 붙은 부호, 줄바꿈.

2026-10-09 작업 #3 출력(2시간 일본어) 점검에서 나온 문제다: 신음만 있는 자막 31%,
`ご視聴ありがとうございました` 31회, 모든 자막이 한 줄(최대 36자), `?いい子です`, `初めてですあ、はい`.
"""

from __future__ import annotations

import pytest

from src.core.config import NoiseFilterConfig, SegmentationConfig
from src.core.text import join_utterances
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.postprocess.noise_filter import filter_noise, is_interjection
from src.postprocess.subtitle_text import (
    mark_utterance_breaks,
    move_leading_punctuation,
    wrap_lines,
)
from src.segmentation.speed_optimizer import get_chars_per_line


def _words(*texts: str, start: float = 0.0) -> list[WordSegment]:
    words = []
    t = start
    for text in texts:
        words.append(WordSegment(text, t, t + 0.4, 0.9))
        t += 0.5
    return words


def _seg(text: str, start: float = 0.0, end: float = 1.0, words: list[WordSegment] | None = None):
    return TranscriptionSegment(1, text, start, end, 0.9, words=words or [])


class TestInterjections:
    @pytest.mark.parametrize(
        "text", ["ああ", "んん", "はぁ", "おー", "ああ…", "ああ?", "ふぅ", "んんんん"]
    )
    def test_moans(self, text: str) -> None:
        assert is_interjection(text)

    @pytest.mark.parametrize("text", ["はい", "いい", "そう", "ありがとう", "?", "ー", "Hello"])
    def test_words_and_punctuation_are_not_moans(self, text: str) -> None:
        assert not is_interjection(text)

    def test_moan_words_removed_and_empty_segment_dropped(self) -> None:
        segs = [
            _seg("ああ素晴らしい", words=_words("ああ", "素晴らしい")),
            _seg("んんんん", 2.0, 3.0, words=_words("んんんん", start=2.0)),
        ]
        result = filter_noise(segs, NoiseFilterConfig())

        assert [s.text for s in result] == ["素晴らしい"]

    def test_can_be_turned_off(self) -> None:
        segs = [_seg("んんんん", words=_words("んんんん"))]
        result = filter_noise(segs, NoiseFilterConfig(remove_interjections=False))
        assert [s.text for s in result] == ["んんんん"]


class TestHallucinations:
    @pytest.mark.parametrize(
        "parts",
        [
            ["ご視聴", "ありがとうございました"],
            ["ご視聴", "ありがとうございました。"],
            ["ご視聴ありがとうございました", "ご視聴ありがとうございました"],
            ["Thank", "you", "for", "watching."],
        ],
    )
    def test_whole_segment_removed(self, parts: list[str]) -> None:
        segs = [_seg("x", words=_words(*parts))]
        assert filter_noise(segs, NoiseFilterConfig()) == []

    def test_phrase_inside_real_speech_is_kept(self) -> None:
        """대화 속 문구는 지우지 않는다 (세그먼트 전체가 그 문구일 때만)."""
        segs = [_seg("x", words=_words("本日は", "ご視聴", "ありがとうございました"))]
        result = filter_noise(segs, NoiseFilterConfig())
        assert [s.text for s in result] == ["本日はご視聴ありがとうございました"]

    def test_everyday_greetings_are_kept(self) -> None:
        segs = [_seg("x", words=_words("おやすみなさい"))]
        assert [s.text for s in filter_noise(segs, NoiseFilterConfig())] == ["おやすみなさい"]


class TestRepeatedPhrases:
    def test_multi_word_phrase_repeated_three_times(self) -> None:
        """`次回予告次回予告次回予告` 처럼 여러 단어 문구의 반복도 하나로 (TO-DO 8)."""
        segs = [_seg("x", words=_words("次回", "予告", "次回", "予告", "次回", "予告", "です"))]
        assert [s.text for s in filter_noise(segs, NoiseFilterConfig())] == ["次回予告です"]

    def test_two_repeats_are_kept(self) -> None:
        segs = [_seg("x", words=_words("よかった", "よかった"))]
        assert [s.text for s in filter_noise(segs, NoiseFilterConfig())] == ["よかったよかった"]


class TestUtteranceBreaks:
    def test_join_utterances(self) -> None:
        assert join_utterances(["初めてです", "あ、はい"]) == "初めてです　あ、はい"
        assert join_utterances(["わかりました。", "どうぞ"]) == "わかりました。どうぞ"
        assert join_utterances(["Hello.", "Bye"]) == "Hello. Bye"

    def test_marks_break_between_stt_segments(self) -> None:
        first = _words("この店に", "来るのは", "初めてです")
        second = _words("あ、", "はい", start=2.0)
        seg = _seg("この店に来るのは初めてですあ、はい", 0.0, 3.0, words=first + second)

        mark_utterance_breaks([seg], {id(first[-1]), id(second[-1])})

        assert seg.text == "この店に来るのは初めてです　あ、はい"

    def test_multi_speaker_text_untouched(self) -> None:
        words = _words("はい", "え")
        seg = _seg("はい / え", words=words)
        seg.speaker = ["A", "B"]
        mark_utterance_breaks([seg], {id(words[0])})
        assert seg.text == "はい / え"


class TestLeadingPunctuation:
    def test_moves_to_previous(self) -> None:
        segs = [_seg("どうですか"), _seg("?いい子です"), _seg("ー。そうです")]
        result = move_leading_punctuation(segs)
        assert [s.text for s in result] == ["どうですか?", "いい子ですー。", "そうです"]

    @pytest.mark.parametrize(
        ("previous", "following", "expected"),
        [
            ("いかがですか?", "?はい", "いかがですか?"),
            ("すみません。", "?はい", "すみません?"),
            ("白金!", "??はい", "白金!?"),
            ("安いのかな?", "?はい", "安いのかな?"),
            ("そうか、", "。はい", "そうか。"),
        ],
    )
    def test_merges_with_existing_punctuation(
        self, previous: str, following: str, expected: str
    ) -> None:
        """앞 자막이 부호로 끝나면 옮긴 부호와 합친다 (TO-DO 3: `いかがですか??`, `すみません。?`)."""
        result = move_leading_punctuation([_seg(previous), _seg(following)])
        assert [s.text for s in result] == [expected, "はい"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("もう8時!??", "もう8時!?"),
            ("早く!!", "早く!"),
            ("はい。?", "はい?"),
            ("そう…", "そう…"),
        ],
    )
    def test_runs_inside_subtitle_are_merged(self, text: str, expected: str) -> None:
        """Whisper 원문에 이어진 부호도 합친다 (작업 #12 `もう8時!??`)."""
        assert [s.text for s in move_leading_punctuation([_seg(text)])] == [expected]

    def test_punctuation_only_segment_removed(self) -> None:
        segs = [_seg("もしもし"), _seg("?"), _seg("はい")]
        result = move_leading_punctuation(segs)
        assert [s.text for s in result] == ["もしもし?", "はい"]
        assert [s.index for s in result] == [1, 2]

    def test_first_segment_drops_punctuation(self) -> None:
        assert [s.text for s in move_leading_punctuation([_seg("、はい")])] == ["はい"]

    def test_ellipsis_is_kept(self) -> None:
        assert [s.text for s in move_leading_punctuation([_seg("a"), _seg("…そうですね")])] == [
            "a",
            "…そうですね",
        ]


class TestWrapLines:
    def test_language_defaults(self) -> None:
        cfg = SegmentationConfig()
        assert get_chars_per_line("ja", cfg) == 16
        assert get_chars_per_line("en", cfg) == 42
        assert get_chars_per_line("ja", SegmentationConfig(max_chars_per_line=20)) == 20

    def test_japanese_breaks_at_phrase_boundary(self) -> None:
        seg = _seg("なんか駅前にあるブルースカイっていうところなんだけど")
        assert wrap_lines([seg], "ja", 16, 2) == 1
        first, second = seg.text.split("\n")
        assert len(first) <= 16 and len(second) <= 16
        assert first + second == "なんか駅前にあるブルースカイっていうところなんだけど"
        # 문절 중간에서 나누지 않는다
        assert not first.endswith(("ブルース", "ところな"))

    def test_prefers_utterance_break(self) -> None:
        seg = _seg("この店に来るのは初めてです　あ、はい")
        wrap_lines([seg], "ja", 16, 2)
        assert seg.text == "この店に来るのは初めてです\nあ、はい"

    def test_english_breaks_at_space(self) -> None:
        seg = _seg("This is a fairly long English subtitle line for the test")
        wrap_lines([seg], "en", 30, 2)
        assert all(len(line) <= 30 for line in seg.text.split("\n"))
        assert (
            seg.text.replace("\n", " ")
            == "This is a fairly long English subtitle line for the test"
        )

    def test_short_and_single_line_settings_untouched(self) -> None:
        short = _seg("短い字幕")
        long = _seg("なんか駅前にあるブルースカイっていうところなんだけど")
        assert wrap_lines([short], "ja", 16, 2) == 0
        assert wrap_lines([long], "ja", 16, 1) == 0
        assert "\n" not in short.text and "\n" not in long.text
