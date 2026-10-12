"""자막 규칙 테스트 (2026-10-10, 영상 3개 점검 뒤 수정 — docs/TO-DO.md 4·2·1·5·7번).

- 4: 오디오 트랙 자동 선택 (예: 기본 eng, 두 번째 jpn)
- 2: advanced 글자 수 상한, 최소 표시 시간 병합의 글자 수 예외
- 1: 복합 명사·복합 동사 문절, 쉼 단위 분석, 작은 가나 앞 경계 없음, 단어 경계 줄바꿈
- 5: 쉼으로 발화 구분
- 7: 원문 공백(영어 단어) 보존, 라틴 문장 부호 앞 공백 없음
"""

from __future__ import annotations

import subprocess
from itertools import pairwise
from unittest.mock import patch

import pytest

from src.audio.extractor import find_audio_track_for_language, get_audio_track_languages
from src.core.config import AppConfig, SegmentationConfig
from src.core.text import join_words
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.pipeline import PipelineOptions, TranscriptionPipeline, _utterance_ends
from src.postprocess.subtitle_text import mark_utterance_breaks, wrap_lines
from src.postprocess.word_units import group_words_into_phrases
from src.segmentation.advanced import AdvancedSegmentation
from src.segmentation.min_display_guard import enforce_min_duration
from src.segmentation.speed_optimizer import lines_needed
from src.web.job_options import JobOptions


def _chars(text: str, start: float = 0.0, step: float = 0.1) -> list[WordSegment]:
    words, t = [], start
    for ch in text:
        words.append(WordSegment(ch, t, t + step * 0.8, 0.9))
        t += step
    return words


def _probe(stdout: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TestAudioTrack:
    def test_languages(self) -> None:
        with (
            patch("src.audio.extractor._find_ffprobe", return_value="ffprobe"),
            patch("subprocess.run", return_value=_probe("1,eng\n2,jpn\n")),
        ):
            assert get_audio_track_languages("x.mkv") == ["eng", "jpn"]

    @pytest.mark.parametrize(
        ("stdout", "expected"),
        [("1,eng\n2,jpn\n", 1), ("1,jpn\n", None), ("1,eng\n2,fre\n", None), ("1\n2\n", None)],
    )
    def test_find_for_language(self, stdout: str, expected: int | None) -> None:
        with (
            patch("src.audio.extractor._find_ffprobe", return_value="ffprobe"),
            patch("subprocess.run", return_value=_probe(stdout)),
        ):
            assert find_audio_track_for_language("x.mkv", "ja") == expected

    def test_pipeline_uses_matching_track(self) -> None:
        config = AppConfig()
        config.stt.source_lang = "ja"
        with patch("src.audio.extractor.find_audio_track_for_language", return_value=1) as find:
            opts = TranscriptionPipeline(config)._resolve_audio_track("x.mkv", PipelineOptions())
        find.assert_called_once_with("x.mkv", "ja")
        assert opts.audio_track == 1

    def test_explicit_track_and_auto_language_are_kept(self) -> None:
        config = AppConfig()
        with patch("src.audio.extractor.find_audio_track_for_language") as find:
            pipeline = TranscriptionPipeline(config)
            assert pipeline._resolve_audio_track("x", PipelineOptions()).audio_track is None
            config.stt.source_lang = "ja"
            assert (
                pipeline._resolve_audio_track("x", PipelineOptions(audio_track=0)).audio_track == 0
            )
        find.assert_not_called()

    def test_job_option(self) -> None:
        base = {"engine": "whisperx", "model_size": "large-v3", "formats": ["srt"]}
        assert JobOptions.model_validate({**base, "audio_track": 1}).audio_track == 1
        assert JobOptions.model_validate(base).audio_track is None
        with pytest.raises(ValueError, match="greater than or equal"):
            JobOptions.model_validate({**base, "audio_track": -1})


class TestCharacterLimit:
    def test_advanced_never_exceeds_two_lines(self) -> None:
        """두 줄(16 x 2)을 넘지 않게 나눈다. 예전에는 1.2배(38자)까지, 긴 문절이 붙으면 그 너머까지 갔다."""
        cfg = SegmentationConfig(min_duration_s=1.0)
        text = "私はただ古い部品でC型の機械を直しただけだだから彼女を追い出したわけではないと"
        phrases = group_words_into_phrases(_chars(text, step=0.15), "ja")

        segments = AdvancedSegmentation(cfg).segment(phrases, "ja")

        assert len(segments) > 1
        assert all(len(s.text) <= 32 for s in segments)
        assert "".join(s.text for s in segments) == text

    def test_guard_does_not_merge_past_char_limit(self) -> None:
        segs = [
            TranscriptionSegment(1, "あ" * 30, 0.0, 3.0, 0.9),
            TranscriptionSegment(2, "でした", 3.2, 3.6, 0.9),
        ]
        result = enforce_min_duration(segs, SegmentationConfig(min_duration_s=1.0), max_chars=32)
        assert len(result) == 2
        assert result[1].duration == pytest.approx(1.0)


class TestLineFit:
    """글자 수 합이 두 줄 안이어도 단어 길이 때문에 두 줄로 나뉘지 않으면 나눈다 (재테스트)."""

    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            (["早く", "出てきなさいよ", "チョコレートパフェと", "バニラアイスクリーム"], 3),
            (["これは", "一体", "我々は"], 1),
            (["あ" * 20, "です"], 2),
            (["This", "is", "a", "test"], 1),
            ([], 0),
        ],
    )
    def test_lines_needed(self, parts: list[str], expected: int) -> None:
        assert lines_needed(parts, 16) == expected

    def test_advanced_splits_unwrappable_text(self) -> None:
        phrases = [
            WordSegment(text, i * 0.8, i * 0.8 + 0.7, 0.9)
            for i, text in enumerate(
                ["早く", "出てきなさいよ", "チョコレートパフェと", "バニラアイスクリーム"]
            )
        ]
        segments = AdvancedSegmentation(SegmentationConfig(min_duration_s=1.0)).segment(
            phrases, "ja"
        )
        assert len(segments) == 2
        assert all(lines_needed([w.word for w in s.words], 16) <= 2 for s in segments)

    def test_guard_does_not_merge_into_unwrappable_text(self) -> None:
        first = [WordSegment("早く", 0.0, 0.4, 0.9), WordSegment("出てきなさいよ", 0.4, 0.8, 0.9)]
        second = [
            WordSegment("チョコレートパフェと", 1.0, 2.5, 0.9),
            WordSegment("バニラアイスクリーム", 2.5, 4.0, 0.9),
        ]
        segs = [
            TranscriptionSegment(1, "早く出てきなさいよ", 0.0, 0.8, 0.9, words=first),
            TranscriptionSegment(
                2, "チョコレートパフェとバニラアイスクリーム", 1.0, 4.0, 0.9, words=second
            ),
        ]
        result = enforce_min_duration(segs, SegmentationConfig(min_duration_s=1.0), 32, 16)
        assert len(result) == 2

    def test_short_segment_borrows_from_neighbor(self) -> None:
        """병합도 빈 시간도 없으면 이웃에게서 시간을 받는다 (`これも好きでしょ?` 0.28초)."""
        segs = [
            TranscriptionSegment(1, "でもこのくらいはさせてよ", 391.28, 393.19, 0.9),
            TranscriptionSegment(2, "これも好きでしょ?", 393.23, 393.51, 0.9),
            TranscriptionSegment(3, "いいって言ったけど!", 393.55, 394.84, 0.9),
        ]
        cfg = SegmentationConfig(min_duration_s=1.0)
        result = enforce_min_duration(segs, cfg, max_chars=20)

        assert [s.text for s in result] == [s.text for s in segs]
        assert all(s.duration >= 1.0 - 1e-6 for s in result)
        assert result[0].end < result[1].start < result[1].end < result[2].start

    def test_single_phrase_wraps_at_morpheme(self) -> None:
        text = "東京都立中央図書館地域資料閲覧室利用案内"
        seg = TranscriptionSegment(1, text, 0.0, 4.0, 0.9, words=[WordSegment(text, 0.0, 4.0, 0.9)])
        wrap_lines([seg], "ja", 16, 2)
        assert all(len(line) <= 16 for line in seg.text.split("\n"))
        assert seg.text.replace("\n", "") == text

    def test_separator_dropped_when_nothing_fits(self) -> None:
        """발화 구분 때문에 어디서도 두 줄에 들어가지 않으면 구분을 빼고 나눈다 (작업 #9 #177)."""
        text = "東京都立中央図書館地域資料展示会場入口案内\u3000第12会議室予約受付"
        words = [
            WordSegment(w, 0.0, 1.0, 0.9)
            for w in ["東京都立中央図書館地域資料展示会場入口案内", "第12", "会議室予約受付"]
        ]
        seg = TranscriptionSegment(1, text, 0.0, 4.0, 0.9, words=words)
        wrap_lines([seg], "ja", 16, 2)
        # 발화 구분을 빼면 15 / 16 으로 들어간다 (TO-DO 11)
        assert seg.text == "東京都立中央図書館地域資料展示\n会場入口案内第12会議室予約受付"

    def test_unknown_word_reanalysed(self) -> None:
        """janome 미등록어(`ゃいけないのにどうしてこんなに` 16자)를 다시 분석해 긴 문절을 만들지 않는다."""
        text = "行かなきゃいけないのにどうしてこんなに遅いんだよこの電車"
        phrases = [w.word for w in group_words_into_phrases(_chars(text), "ja")]
        assert "".join(phrases) == text
        assert max(len(p) for p in phrases) <= 10

        seg = TranscriptionSegment(1, text, 0.0, 4.0, 0.9)
        wrap_lines([seg], "ja", 16, 2)
        assert all(len(line) <= 16 for line in seg.text.split("\n"))


class TestCollapsedAlignment:
    def test_collapsed_run_takes_spare_time_from_neighbours(self) -> None:
        """실제 작업 값: 50글자가 1.64초에 몰리고 뒤 `お家だよ` 1.44초, `!` 0.92초 (TO-DO 9)."""
        from src.postprocess.word_units import spread_collapsed_words

        texts = [
            ("アヤカちゃん!ナカムラさんたら", 220.43, 221.7),
            ("新世代で", 221.7, 221.82),
            ("ヒロシちゃんを", 221.82, 221.96),
            ("改造", 221.96, 222.0),
            ("しようと", 222.0, 222.08),
            ("したんだよ", 222.08, 222.18),
            ("!委員長って", 222.18, 222.32),
            ("こんなに", 222.32, 222.4),
            ("女の子っぽい", 222.4, 222.52),
            ("雰囲気だったんだよ", 222.52, 222.84),
            ("ヒロシちゃんの", 222.84, 223.34),
            ("お家だよ", 223.34, 224.78),
            ("!", 224.78, 225.7),
        ]
        words = [WordSegment(t, s, e, 0.9) for t, s, e in texts]

        assert spread_collapsed_words(words, 7.0) == 1

        run = words[1:11]
        rate = sum(len(w.word) for w in run) / (run[-1].end - run[0].start)
        assert rate < 20  # 초당 30자 → 20자 미만
        assert words[0].end == pytest.approx(221.7)  # 앞 단어는 정상 속도라 그대로
        assert words[11].duration == pytest.approx(4 / 7)  # 4글자를 CPS 7 로 (`!` 의 시간도 받았다)
        assert all(a.end <= b.start + 1e-9 for a, b in pairwise(words))

    def test_normal_english_speech_untouched(self) -> None:
        """영어는 글자 밀도가 높다: 초당 15자 정상 발화를 몰림으로 보지 않는다 (CPS 17 기준)."""
        from src.postprocess.word_units import spread_collapsed_words

        texts = ["this", "is", "a", "perfectly", "normal", "english", "sentence", "here", "today"]
        words, t = [], 0.0
        for text in texts:
            words.append(WordSegment(text, t, t + len(text) / 15, 0.9))
            t += len(text) / 15
        words.append(WordSegment("okay", t, t + 1.5, 0.9))
        before = [(w.start, w.end) for w in words]
        assert spread_collapsed_words(words, 17.0) == 0
        assert [(w.start, w.end) for w in words] == before

    def test_normal_speech_untouched(self) -> None:
        from src.postprocess.word_units import spread_collapsed_words

        words = _chars("これは普通の速さで話している文です", step=0.12)
        before = [(w.start, w.end) for w in words]
        assert spread_collapsed_words(words, 7.0) == 0
        assert [(w.start, w.end) for w in words] == before


class TestInnerPunctuation:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("そうか?そうだよ、", ["そうか?", "そうだよ、"]),
            ("しただけだ。だから", ["しただけだ。", "だから"]),
            ("変!?そうか", ["変!?", "そうか"]),
            ("1,000", ["1,000"]),
            ("U.S.", ["U.S."]),
        ],
    )
    def test_split_after_inner_mark(self, text: str, expected: list[str]) -> None:
        """단어 가운데 부호 뒤에서 나눈다 (경계 옮기기·절 경계가 쓴다, 작업 #12 재테스트)."""
        from src.postprocess.word_units import split_at_inner_punctuation

        words = split_at_inner_punctuation([WordSegment(text, 1.0, 2.0, 0.9)])
        assert [w.word for w in words] == expected
        assert words[0].start == 1.0 and words[-1].end == 2.0


class TestReadingTime:
    def test_fast_subtitle_extends_into_gap(self) -> None:
        """CPS 상한(ja 7)이 되도록 뒤쪽 빈 시간으로 끝을 늘린다 (TO-DO 10)."""
        from src.segmentation.speed_optimizer import extend_for_reading

        segs = [
            TranscriptionSegment(1, "そういうわけではないんだ。", 152.05, 153.13, 0.9),
            TranscriptionSegment(2, "無論だ。", 155.0, 156.5, 0.9),
        ]
        assert extend_for_reading(segs, "ja", SegmentationConfig()) == 1
        assert segs[0].end == pytest.approx(152.05 + 13 / 7)
        assert segs[0].start == pytest.approx(152.05)

    def test_limited_by_next_subtitle(self) -> None:
        from src.segmentation.speed_optimizer import extend_for_reading

        segs = [
            TranscriptionSegment(1, "そういうわけではないんだ。", 152.05, 153.13, 0.9),
            TranscriptionSegment(2, "無論だ。", 153.5, 155.0, 0.9),
        ]
        extend_for_reading(segs, "ja", SegmentationConfig(min_gap_s=0.04))
        assert segs[0].end == pytest.approx(153.46)


class TestSpeakerSeparatorLineBreak:
    def test_break_at_speaker_separator_drops_it(self) -> None:
        """화자 구분자 ` / ` 자리에서 나누면 구분자를 지운다 (TO-DO 12: `それじゃあ /`)."""
        seg = TranscriptionSegment(1, "それじゃあ / どうもありがとうございました。", 0.0, 4.0, 0.9)
        wrap_lines([seg], "ja", 16, 2, " / ")
        assert seg.text == "それじゃあ\nどうもありがとうございました。"


class TestMergeGap:
    def test_far_neighbour_is_not_merged(self) -> None:
        """쉼이 1초 이상이면 병합 대신 빈 시간으로 늘린다 (TO-DO 6: `おはよう` + 2.3초 뒤 대사)."""
        segs = [
            TranscriptionSegment(1, "おはよう", 10.0, 10.75, 0.9),
            TranscriptionSegment(2, "あなたたち今日も遅刻ギリギリね", 13.05, 16.15, 0.9),
        ]
        result = enforce_min_duration(segs, SegmentationConfig(min_duration_s=1.0))
        assert [s.text for s in result] == ["おはよう", "あなたたち今日も遅刻ギリギリね"]
        assert result[0].duration == pytest.approx(1.0)

    def test_close_neighbour_is_merged(self) -> None:
        segs = [
            TranscriptionSegment(1, "はい", 10.0, 10.5, 0.9),
            TranscriptionSegment(2, "そうです", 10.7, 12.0, 0.9),
        ]
        result = enforce_min_duration(segs, SegmentationConfig(min_duration_s=1.0))
        assert len(result) == 1


class TestPhrases:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("三大発明を選ぶ", ["三大発明を", "選ぶ"]),
            ("最後のラストチャンスだ", ["最後の", "ラストチャンスだ"]),
            ("負担は計り知れない", ["負担は", "計り知れない"]),
            ("私何もできない", ["私", "何も", "できない"]),
        ],
    )
    def test_compounds(self, text: str, expected: list[str]) -> None:
        assert [w.word for w in group_words_into_phrases(_chars(text), "ja")] == expected

    def test_pause_splits_analysis(self) -> None:
        """쉼(0.2초 이상) 앞뒤를 따로 분석한다: 긴 무부호 문장에서 `まし|ょう` 로 무너지지 않는다."""
        first = _chars("してあげましょう")
        second = _chars("どうですかどうしたらいいんですか", start=first[-1].end + 0.4)

        phrases = [w.word for w in group_words_into_phrases(first + second, "ja")]

        assert "してあげましょう" in phrases
        assert not any(p.startswith("ょ") for p in phrases)

    def test_original_spaces_kept_for_english(self) -> None:
        words = _chars("これはsmartwatchです")
        phrases = group_words_into_phrases(words, "ja", "これは smart watch です")
        assert join_words(w.word for w in phrases) == "これはsmart watchです"


class TestLineBreaks:
    def test_breaks_at_word_boundary(self) -> None:
        phrases = group_words_into_phrases(
            _chars("20世紀における三大発明を選ぶならこの携帯電話端末"), "ja"
        )
        seg = TranscriptionSegment(
            1, join_words(w.word for w in phrases), 0.0, 5.0, 0.9, words=phrases
        )
        wrap_lines([seg], "ja", 16, 2)
        assert "\n" in seg.text
        assert "三\n大" not in seg.text and "携帯\n" not in seg.text

    def test_long_phrase_falls_back_to_morphemes(self) -> None:
        seg = TranscriptionSegment(
            1, "なんか駅前にあるブルースカイっていうところなんだけど", 0.0, 4.0, 0.9
        )
        wrap_lines([seg], "ja", 16, 2)
        first, second = seg.text.split("\n")
        assert len(first) <= 16 and len(second) <= 16
        assert not second.startswith(("っ", "ー"))


class TestPauseUtterances:
    def test_long_pause_marks_utterance(self) -> None:
        words = [
            WordSegment("いや", 0.0, 0.4, 0.9),
            WordSegment("何でもない", 0.4, 1.0, 0.9),
            WordSegment("何がだよ", 1.8, 2.4, 0.9),
        ]
        seg = TranscriptionSegment(1, "いや何でもない何がだよ", 0.0, 2.4, 0.9, words=words)

        ends = _utterance_ends([seg])
        mark_utterance_breaks([seg], ends)

        assert id(words[1]) in ends and id(words[0]) not in ends
        assert seg.text == "いや何でもない　何がだよ"


class TestLatinSpacing:
    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            (["She", "was", "crying", "."], "She was crying."),
            (["Can", "'", "t"], "Can't"),
            (["If", "so", ",", "school"], "If so, school"),
            (["(", "note", ")"], "(note)"),
        ],
    )
    def test_punctuation(self, parts: list[str], expected: str) -> None:
        assert join_words(parts) == expected
