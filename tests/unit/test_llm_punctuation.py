"""LLM 부호 넣기 테스트 (세그멘테이션 전, 2026-10-10).

LLM 응답은 가짜 프로바이더로 만든다. 부호만 더한 응답을 받아 단어(문절)에 넣고, 시각은 바꾸지 않는다.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from src.core.config import AppConfig, LLMConfig
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.pipeline import PipelineOptions, TranscriptionPipeline
from src.postprocess.llm_punctuation import punctuate_transcript
from src.segmentation.llm_client import LLMProvider, LLMResponse
from src.stage_cache import StageCache

_CONFIG = LLMConfig(provider="fake", max_retries=0)


class _FakeProvider(LLMProvider):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[dict[str, object]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.prompts.append(json.loads(user_prompt))
        return LLMResponse(self.replies.pop(0), input_tokens=10, output_tokens=5)

    def provider_name(self) -> str:
        return "fake"


def _changes(*items: tuple[int, str]) -> str:
    return json.dumps({"changes": [{"id": i, "text": t} for i, t in items]}, ensure_ascii=False)


def _segment(*phrases: str, start: float = 0.0) -> TranscriptionSegment:
    words, t = [], start
    for phrase in phrases:
        words.append(WordSegment(phrase, t, t + 0.5, 0.9))
        t += 0.6
    return TranscriptionSegment(1, "".join(phrases), start, t, 0.9, words=words)


class TestPunctuateTranscript:
    def test_punctuation_goes_into_words(self) -> None:
        seg = _segment("本当に", "よかった", "田中さん", "どうかしたの?", "いや", "何でもない")
        times = [(w.start, w.end) for w in seg.words]
        reply = _changes((1, "本当によかった。田中さん、どうかしたの?いや、何でもない。"))

        result = punctuate_transcript([seg], "ja", _CONFIG, provider=_FakeProvider([reply]))

        assert [w.word for w in seg.words] == [
            "本当に",
            "よかった。",
            "田中さん、",
            "どうかしたの?",
            "いや、",
            "何でもない。",
        ]
        assert seg.text == "本当によかった。田中さん、どうかしたの?いや、何でもない。"
        assert [(w.start, w.end) for w in seg.words] == times
        assert result.stats.changed == 1

    def test_punctuation_inside_a_phrase_splits_the_word(self) -> None:
        """단어 가운데 넣은 부호 뒤에서 단어를 나눈다 (`しただけだ。だから`, 경계 옮기기·절 경계가 쓴다)."""
        seg = _segment("そうかそうか", "わかった")
        reply = _changes((1, "そうか、そうか。わかった"))
        punctuate_transcript([seg], "ja", _CONFIG, provider=_FakeProvider([reply]))
        assert [w.word for w in seg.words] == ["そうか、", "そうか。", "わかった"]
        first, second = seg.words[0], seg.words[1]
        assert first.start == 0.0 and second.end == 0.5
        assert first.end == second.start == 0.25

    def test_full_stop_after_wo_is_dropped(self) -> None:
        """`を` 뒤 마침표는 빼고 다른 부호는 받는다 (TO-DO 13: `同じであることを。`)."""
        seg = _segment("正しさと", "同じである", "ことを")
        reply = _changes((1, "正しさと、同じであることを。"))
        punctuate_transcript([seg], "ja", _CONFIG, provider=_FakeProvider([reply]))
        assert seg.text == "正しさと、同じであることを"

    def test_changed_characters_are_rejected(self) -> None:
        seg = _segment("カナを", "呼んだ")
        reply = _changes((1, "ハナを呼んだ。"))

        result = punctuate_transcript([seg], "ja", _CONFIG, provider=_FakeProvider([reply]))

        assert seg.text == "カナを呼んだ"
        assert (result.stats.changed, result.stats.rejected) == (0, 1)
        assert result.texts == {"カナを呼んだ": "カナを呼んだ"}

    def test_known_texts_are_not_sent_again(self) -> None:
        first = _segment("終わったよ", "そうか")
        second = _segment("いや", "何でもない", start=5.0)
        provider = _FakeProvider([_changes((1, "いや、何でもない"))])
        known = {"終わったよそうか": "終わったよ。そうか"}

        result = punctuate_transcript([first, second], "ja", _CONFIG, known, provider)

        assert provider.prompts[0]["lines"] == [{"id": 1, "text": "いや何でもない"}]
        assert first.text == "終わったよ。そうか" and second.text == "いや、何でもない"
        assert result.stats.cached == 1
        assert result.texts["いや何でもない"] == "いや、何でもない"

    def test_unanswered_window_is_not_recorded(self) -> None:
        seg = _segment("終わったよ", "そうか")
        result = punctuate_transcript([seg], "ja", _CONFIG, provider=_FakeProvider(["not json"]))
        assert result.texts == {}
        assert seg.text == "終わったよそうか"

    def test_english_comma_after_word(self) -> None:
        seg = TranscriptionSegment(
            1,
            "If so school is out",
            0.0,
            2.0,
            0.9,
            words=[
                WordSegment(w, i * 0.4, i * 0.4 + 0.3, 0.9)
                for i, w in enumerate(["If", "so", "school", "is", "out"])
            ],
        )
        reply = _changes((1, "If so, school is out."))
        punctuate_transcript([seg], "en", _CONFIG, provider=_FakeProvider([reply]))
        assert [w.word for w in seg.words] == ["If", "so,", "school", "is", "out."]


class TestOtherLanguages:
    def test_korean_keeps_spaces(self) -> None:
        """다른 언어도 같은 과정: 한국어는 공백 단어 뒤에 부호를 붙이고 공백은 바꾸지 않는다."""
        words = [
            WordSegment(w, i * 0.5, i * 0.5 + 0.4, 0.9)
            for i, w in enumerate(["그래", "알았어", "내일", "보자"])
        ]
        seg = TranscriptionSegment(1, "그래 알았어 내일 보자", 0.0, 2.0, 0.9, words=words)
        reply = _changes((1, "그래, 알았어. 내일 보자."))
        punctuate_transcript([seg], "ko", _CONFIG, provider=_FakeProvider([reply]))
        assert seg.text == "그래, 알았어. 내일 보자."

    def test_korean_spacing_change_is_rejected(self) -> None:
        words = [
            WordSegment(w, i * 0.5, i * 0.5 + 0.4, 0.9) for i, w in enumerate(["내일", "보자"])
        ]
        seg = TranscriptionSegment(1, "내일 보자", 0.0, 1.0, 0.9, words=words)
        reply = _changes((1, "내일보자."))
        result = punctuate_transcript([seg], "ko", _CONFIG, provider=_FakeProvider([reply]))
        assert seg.text == "내일 보자"
        assert result.stats.rejected == 1


class TestPipelineStage:
    def _pipeline(self) -> TranscriptionPipeline:
        config = AppConfig()
        config.postprocess.llm_punctuation = True
        config.noise_filter.enabled = False
        return TranscriptionPipeline(config)

    def test_runs_before_segmentation_and_saves_cache(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        seg = _segment("終わったよ", "そうか")
        cache = StageCache(tmp_path / "work")
        reply = _changes((1, "終わったよ。そうか"))
        with patch(
            "src.segmentation.llm_client.create_llm_provider",
            return_value=_FakeProvider([reply]),
        ):
            result = self._pipeline()._postprocess_stt(
                [seg], "ja", cache, PipelineOptions(resume=True)
            )

        assert result[0].text == "終わったよ。そうか"
        saved = json.loads((tmp_path / "work" / "stages.json").read_text(encoding="utf-8"))
        assert saved["stages"]["punctuation"]["texts"] == {"終わったよそうか": "終わったよ。そうか"}

    def test_resume_uses_cache_without_calling(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        cache = StageCache(tmp_path / "work")
        reply = _changes((1, "終わったよ。そうか"))
        with patch(
            "src.segmentation.llm_client.create_llm_provider",
            return_value=_FakeProvider([reply]),
        ):
            self._pipeline()._postprocess_stt(
                [_segment("終わったよ", "そうか")], "ja", cache, PipelineOptions(resume=True)
            )
        with patch("src.segmentation.llm_client.create_llm_provider") as create:
            result = self._pipeline()._postprocess_stt(
                [_segment("終わったよ", "そうか")], "ja", cache, PipelineOptions(resume=True)
            )
        create.assert_not_called()
        assert result[0].text == "終わったよ。そうか"

    def test_off_by_default(self) -> None:
        with patch("src.postprocess.llm_punctuation.punctuate_transcript") as punctuate:
            TranscriptionPipeline(AppConfig())._postprocess_stt([_segment("あ", "い")], "ja")
        punctuate.assert_not_called()
