"""검수 메모 언어 테스트: 메모·교정 이유를 자막 언어가 아니라 화면 언어로 남긴다 (2026-10-12).

LLM 응답은 가짜 프로바이더로 만든다.
"""

from __future__ import annotations

import json

import pytest

from src.core.config import AppConfig, LLMConfig, SegmentationConfig
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.postprocess.llm_correction import correct_transcript
from src.postprocess.llm_finalize import finalize_subtitles
from src.postprocess.llm_text import (
    LLMTextStats,
    is_written_in,
    localize_notes,
    note_language_rule,
)
from src.segmentation.llm_client import LLMProvider, LLMResponse
from src.web.job_options import JobOptions

_LLM = LLMConfig(provider="fake", max_retries=0)
_CFG = SegmentationConfig(min_duration_s=1.0, max_duration_s=7.0, min_gap_s=0.04)
_JA_NOTE = "「機会」は「機械」の誤認識の可能性"
_KO_NOTE = "'機会'는 '機械'의 오인식으로 보임"


class _FakeProvider(LLMProvider):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        return LLMResponse(self.replies.pop(0), input_tokens=10, output_tokens=5)

    def provider_name(self) -> str:
        return "fake"


def _notes(*items: tuple[int, str]) -> str:
    return json.dumps({"notes": [{"id": i, "text": t} for i, t in items]}, ensure_ascii=False)


def _seg(*phrases: str) -> TranscriptionSegment:
    words = [WordSegment(p, i * 0.6, i * 0.6 + 0.5, 0.9) for i, p in enumerate(phrases)]
    return TranscriptionSegment(1, "".join(phrases), 0.0, words[-1].end, 0.9, words=words)


class TestWrittenIn:
    @pytest.mark.parametrize(
        ("text", "language", "expected"),
        [
            (_JA_NOTE, "ko", False),
            (_KO_NOTE, "ko", True),
            (
                "カナ → ハナ 로 보임",
                "ko",
                True,
            ),  # 인용하지 않은 자막 낱말이 섞여도 한글이 있으면 된다
            ("「機会」", "ko", False),  # 인용만 있다
            ("'機会' is probably '機械'", "en", True),
            (_JA_NOTE, "en", False),
            (_KO_NOTE, "en", False),
            (_JA_NOTE, "ja", True),
            (_KO_NOTE, "ja", False),
            ("“機会”应为“機械”", "zh", True),
            (_JA_NOTE, "zh", False),
            (_JA_NOTE, "fr", True),  # 판단할 수 없는 언어는 맞는 것으로 본다
        ],
    )
    def test_script_decides(self, text: str, language: str, expected: bool) -> None:
        assert is_written_in(text, language) is expected

    def test_rule_names_the_language_and_field(self) -> None:
        rule = note_language_rule("ko", "note", "subtitles")
        assert "Korean (한국어)" in rule and '"note"' in rule and "subtitles" in rule
        assert "English" in note_language_rule("en", "reason", "lines")
        # 모르는 코드는 기본 언어
        assert "Korean" in note_language_rule("xx", "note", "subtitles")


class TestLocalizeNotes:
    def test_only_mismatched_notes_are_sent(self) -> None:
        provider = _FakeProvider([_notes((1, _KO_NOTE))])
        stats = LLMTextStats()

        notes, records = localize_notes(
            ["문장 끝에 마침표가 없음", _JA_NOTE, ""], "ko", _LLM, stats, provider
        )

        assert notes == ["문장 끝에 마침표가 없음", _KO_NOTE, ""]
        sent = json.loads(provider.user_prompts[0])
        assert sent["notes"] == [{"id": 1, "text": _JA_NOTE}] and sent["language"] == "ko"
        assert "Korean (한국어)" in provider.system_prompts[0]
        assert stats.windows == 1 and len(records) == 1

    def test_no_call_when_every_note_matches(self) -> None:
        provider = _FakeProvider([])
        notes, records = localize_notes([_KO_NOTE], "ko", _LLM, LLMTextStats(), provider)
        assert notes == [_KO_NOTE] and records == {} and provider.user_prompts == []

    @pytest.mark.parametrize(
        "translated",
        [
            "'기회'는 '기계'의 오인식으로 보임",  # 인용한 자막 낱말을 옮겨 버렸다
            "「機会」は「機械」の誤りです",  # 여전히 메모 언어가 아니다
        ],
    )
    def test_bad_translation_keeps_the_original(self, translated: str) -> None:
        provider = _FakeProvider([_notes((1, translated), (9, "묶음 밖"))])
        notes, _ = localize_notes([_JA_NOTE], "ko", _LLM, LLMTextStats(), provider)
        assert notes == [_JA_NOTE]

    def test_known_answer_is_not_requested_again(self) -> None:
        first = _FakeProvider([_notes((1, _KO_NOTE))])
        _, records = localize_notes([_JA_NOTE], "ko", _LLM, LLMTextStats(), first)
        again = _FakeProvider([])
        notes, _ = localize_notes([_JA_NOTE], "ko", _LLM, LLMTextStats(), again, known=records)
        assert notes == [_KO_NOTE] and again.user_prompts == []


class TestFinalizeNotes:
    def _issue_reply(self, note: str) -> str:
        return json.dumps(
            {"issues": [{"id": 1, "type": "misrecognition", "note": note}]}, ensure_ascii=False
        )

    def test_prompt_and_request_carry_the_note_language(self) -> None:
        provider = _FakeProvider([self._issue_reply(_KO_NOTE)])
        result = finalize_subtitles(
            [_seg("この", "機会は", "古い")], "ja", 16, _CFG, _LLM, provider=provider
        )
        assert "Korean (한국어)" in provider.system_prompts[0]
        assert "in Korean, " not in provider.system_prompts[0]  # 예전의 고정 문구가 남지 않았다
        assert json.loads(provider.user_prompts[0])["note_language"] == "Korean (한국어)"
        assert result.issues[0]["note"] == _KO_NOTE and len(provider.user_prompts) == 1

    def test_subtitle_language_note_is_translated_and_cached(self) -> None:
        provider = _FakeProvider([self._issue_reply(_JA_NOTE), _notes((1, _KO_NOTE))])
        segs = [_seg("この", "機会は", "古い")]

        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=provider)

        assert result.issues[0]["note"] == _KO_NOTE
        assert len(provider.user_prompts) == 2 and len(result.records) == 2
        # 단계 캐시가 있으면 검토도 옮기기도 다시 보내지 않는다
        again = _FakeProvider([])
        cached = finalize_subtitles(
            [_seg("この", "機会は", "古い")], "ja", 16, _CFG, _LLM, result.records, again
        )
        assert cached.issues[0]["note"] == _KO_NOTE and again.user_prompts == []

    def test_english_screen_gets_english_notes(self) -> None:
        english = "'機会' is probably '機械'"
        provider = _FakeProvider([self._issue_reply(_JA_NOTE), _notes((1, english))])
        result = finalize_subtitles(
            [_seg("この", "機会は", "古い")],
            "ja",
            16,
            _CFG,
            _LLM,
            provider=provider,
            note_language="en",
        )
        assert result.issues[0]["note"] == english
        assert "English" in provider.system_prompts[0]


class TestCorrectionReason:
    def test_reason_is_translated_before_it_is_cached(self) -> None:
        correction = json.dumps(
            {
                "corrections": [
                    {"id": 1, "from": "以外な", "to": "意外な", "reason": "同じ読みです"}
                ]
            },
            ensure_ascii=False,
        )
        provider = _FakeProvider([correction, _notes((1, "읽기가 같음"))])
        seg = _seg("とても", "以外", "な")

        result = correct_transcript([seg], "ja", _LLM, provider=provider)

        assert "Korean (한국어)" in provider.system_prompts[0]
        assert result.edits["とても以外な"] == [["以外な", "意外な", "읽기가 같음"]]
        assert result.suggested[0]["reason"] == "읽기가 같음"


_JOB = {"engine": "faster-whisper", "model_size": "tiny", "formats": ["srt"]}


class TestReviewLanguageOption:
    def test_default_is_korean_and_job_option_overrides(self) -> None:
        assert AppConfig().postprocess.review_language == "ko"
        options = JobOptions.model_validate({**_JOB, "review_language": "en"})
        assert options.to_overrides()["postprocess.review_language"] == "en"
        # 화면이 보내지 않으면 설정 파일 값을 쓴다 (None 은 덮어쓰지 않는다)
        assert JobOptions.model_validate(_JOB).to_overrides()["postprocess.review_language"] is None

    def test_unknown_language_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="review_language"):
            JobOptions.model_validate({**_JOB, "review_language": "fr"})
