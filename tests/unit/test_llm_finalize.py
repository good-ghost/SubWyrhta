"""LLM 마지막 검토 테스트: 경계 옮기기, 줄바꿈, 검수 리포트, 단계 캐시 (2026-10-10).

LLM 응답은 가짜 프로바이더로 만든다.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from src.core.config import AppConfig, LLMConfig, SegmentationConfig
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.postprocess.llm_finalize import finalize_subtitles
from src.segmentation.llm_client import LLMProvider, LLMResponse

if TYPE_CHECKING:
    from pathlib import Path

_LLM = LLMConfig(provider="fake", max_retries=0)
_CFG = SegmentationConfig(min_duration_s=1.0, max_duration_s=7.0, min_gap_s=0.04)


class _FakeProvider(LLMProvider):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        return LLMResponse(self.replies.pop(0), input_tokens=10, output_tokens=5)

    def provider_name(self) -> str:
        return "fake"


def _reply(boundaries=(), line_breaks=(), issues=()) -> str:  # type: ignore[no-untyped-def]
    return json.dumps(
        {"boundaries": list(boundaries), "line_breaks": list(line_breaks), "issues": list(issues)},
        ensure_ascii=False,
    )


def _seg(words: list[tuple[str, float, float]], speaker: str = "A") -> TranscriptionSegment:
    ws = [WordSegment(w, s, e, 0.9) for w, s, e in words]
    text = "".join(w for w, _, _ in words)
    return TranscriptionSegment(1, text, ws[0].start, ws[-1].end, 0.9, speaker=speaker, words=ws)


def _split_sentence() -> list[TranscriptionSegment]:
    """작업 #12 실제 시각: `駅の近くの店に` / `いなくて本当によかった。田中さん、どうかしたの?`."""
    return [
        _seg([("駅の", 166.84, 167.22), ("近くの", 167.22, 167.6), ("店に", 167.6, 167.96)]),
        _seg(
            [
                ("いなくて", 168.0, 168.76),
                ("本当に", 168.76, 169.22),
                ("よかった。", 169.22, 169.5),
                ("田中さん、", 169.5, 170.44),
                ("どうかしたの?", 170.44, 172.21),
            ]
        ),
    ]


class TestBoundaries:
    def test_moves_words_to_sentence_end(self) -> None:
        segs = _split_sentence()
        reply = _reply(
            boundaries=[
                {
                    "id": 1,
                    "text": "駅の近くの店にいなくて本当によかった。",
                    "next": "田中さん、どうかしたの?",
                }
            ]
        )

        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_FakeProvider([reply]))

        assert result.boundaries == 1
        assert segs[0].text.replace("\n", "") == "駅の近くの店にいなくて本当によかった。"
        assert segs[1].text == "田中さん、どうかしたの?"
        assert segs[0].end == pytest.approx(169.46)  # 다음 자막 시작 - min_gap
        assert segs[1].start == pytest.approx(169.5)
        assert [w.word for w in segs[1].words] == ["田中さん、", "どうかしたの?"]

    @pytest.mark.parametrize(
        ("text", "next_text"),
        [
            (
                "駅の近くの店にいなくて本当に良かった。",
                "田中さん、どうかしたの?",
            ),  # 글자 바꿈
            (
                "駅の近くの店にいな",
                "くて本当によかった。田中さん、どうかしたの?",
            ),  # 단어 중간
        ],
    )
    def test_rejects_invalid(self, text: str, next_text: str) -> None:
        segs = _split_sentence()
        before = [s.text for s in segs]
        reply = _reply(boundaries=[{"id": 1, "text": text, "next": next_text}])

        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_FakeProvider([reply]))

        assert result.boundaries == 0
        assert [s.text for s in segs] == before

    def test_next_text_may_omit_its_tail(self) -> None:
        """LLM 이 뒤 자막 끝을 빼고 보내도 앞 텍스트로 경계를 정한다 (빠진 끝은 그대로 남는다)."""
        segs = _split_sentence()
        reply = _reply(
            boundaries=[
                {
                    "id": 1,
                    "text": "駅の近くの店にいなくて本当によかった。",
                    "next": "田中さん、",
                }
            ]
        )
        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_FakeProvider([reply]))
        assert result.boundaries == 1
        assert segs[1].text == "田中さん、どうかしたの?"

    def test_rejects_too_short_result(self) -> None:
        segs = _split_sentence()
        segs[1].words = segs[1].words[:4]  # `どうかしたの?` 없이 `田中さん、` 0.94초가 남는다
        segs[1].text = "いなくて本当によかった。田中さん、"
        segs[1].end = 170.44
        reply = _reply(
            boundaries=[
                {
                    "id": 1,
                    "text": "駅の近くの店にいなくて本当によかった。",
                    "next": "田中さん、",
                }
            ]
        )
        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_FakeProvider([reply]))
        assert result.boundaries == 0

    def test_rejects_different_speakers(self) -> None:
        segs = _split_sentence()
        segs[1].speaker = "B"
        reply = _reply(
            boundaries=[
                {
                    "id": 1,
                    "text": "駅の近くの店にいなくて本当によかった。",
                    "next": "田中さん、どうかしたの?",
                }
            ]
        )
        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_FakeProvider([reply]))
        assert result.boundaries == 0


class TestSplits:
    def _two_speakers(self) -> list[TranscriptionSegment]:
        """재테스트 실제 사례: 한 자막에 두 사람 대사 `君の顔を見たくなってな。` / `そんな理由で?`."""
        seg = _seg(
            [
                ("君の", 10.0, 10.5),
                ("顔を", 10.5, 11.0),
                ("見たくなってな。", 11.0, 12.0),
                ("そんな", 12.3, 12.8),
                ("理由で?", 12.8, 13.6),
            ]
        )
        seg.text = "君の顔を見たくなってな。\nそんな理由で?"
        return [seg]

    def test_splits_into_two_subtitles(self) -> None:
        segs = self._two_speakers()
        reply = json.dumps(
            {"splits": [{"id": 1, "parts": ["君の顔を見たくなってな。", "そんな理由で?"]}]},
            ensure_ascii=False,
        )

        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_FakeProvider([reply]))

        assert result.splits == 1
        assert [s.text for s in segs] == ["君の顔を見たくなってな。", "そんな理由で?"]
        assert segs[0].end == pytest.approx(12.0) and segs[1].start == pytest.approx(12.3)
        assert segs[1].end == pytest.approx(13.6)

    def test_rejects_split_with_changed_characters_or_short_part(self) -> None:
        segs = self._two_speakers()
        reply = json.dumps(
            {
                "splits": [
                    {"id": 1, "parts": ["君の顔を見たくなってな。", "そんな理由で"]},
                ]
            },
            ensure_ascii=False,
        )
        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_FakeProvider([reply]))
        assert result.splits == 0 and len(segs) == 1

        segs = self._two_speakers()
        segs[0].words[-1].end = 12.9  # 뒤 조각 0.6초
        segs[0].end = 12.9
        reply = json.dumps(
            {"splits": [{"id": 1, "parts": ["君の顔を見たくなってな。", "そんな理由で?"]}]},
            ensure_ascii=False,
        )
        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_FakeProvider([reply]))
        assert result.splits == 0

    def test_ids_stay_bound_after_split(self) -> None:
        """나누기로 자막이 늘어도 뒤 자막의 번호가 어긋나지 않는다."""
        segs = [
            *self._two_speakers(),
            _seg([("東京都立中央図書館", 20.0, 21.5), ("地域資料閲覧室利用案内", 21.5, 23.0)]),
        ]
        segs[1].text = "東京都立中央図書館\n地域資料閲覧室利用案内"
        reply = json.dumps(
            {
                "splits": [{"id": 1, "parts": ["君の顔を見たくなってな。", "そんな理由で?"]}],
                "line_breaks": [{"id": 2, "text": "東京都立中央図書館地域\n資料閲覧室利用案内"}],
            },
            ensure_ascii=False,
        )
        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_FakeProvider([reply]))
        assert (result.splits, result.line_breaks) == (1, 1)
        assert segs[2].text == "東京都立中央図書館地域\n資料閲覧室利用案内"


class TestEnglish:
    def test_moves_boundary_between_english_words(self) -> None:
        """다른 언어도 같은 과정: 영어는 공백 단어 경계로 옮긴다."""
        first = _seg([("I", 0.0, 0.2), ("told", 0.2, 0.5), ("you", 0.5, 0.8), ("that", 0.8, 1.2)])
        first.text = "I told you that"
        second = _seg(
            [
                ("yesterday.", 1.3, 2.0),
                ("Where", 2.2, 2.5),
                ("were", 2.5, 2.8),
                ("you?", 2.8, 3.5),
            ]
        )
        second.text = "yesterday. Where were you?"
        reply = _reply(
            boundaries=[{"id": 1, "text": "I told you that yesterday.", "next": "Where were you?"}]
        )
        cfg = SegmentationConfig(min_duration_s=1.0, max_duration_s=7.0, min_gap_s=0.04)

        result = finalize_subtitles(
            [first, second], "en", 42, cfg, _LLM, provider=_FakeProvider([reply])
        )

        assert result.boundaries == 1
        assert (first.text, second.text) == ("I told you that yesterday.", "Where were you?")
        assert second.start == pytest.approx(2.2)


class TestLineBreaksAndIssues:
    def test_line_break_and_issue(self) -> None:
        seg = _seg([("東京都立中央図書館", 0.0, 1.5), ("地域資料閲覧室利用案内", 1.5, 3.0)])
        seg.text = "東京都立中央図書館\n地域資料閲覧室利用案内"
        reply = _reply(
            line_breaks=[{"id": 1, "text": "東京都立中央図書館地域\n資料閲覧室利用案内"}],
            issues=[
                {"id": 1, "type": "misrecognition", "note": "カナ → ハナ 로 보임"},
                {"id": 1, "type": "weird", "note": "x"},
                {"id": 9, "type": "other", "note": "묶음 밖"},
            ],
        )

        result = finalize_subtitles([seg], "ja", 16, _CFG, _LLM, provider=_FakeProvider([reply]))

        assert seg.text == "東京都立中央図書館地域\n資料閲覧室利用案内"
        assert result.line_breaks == 1
        assert [i["type"] for i in result.issues] == ["misrecognition", "other"]
        assert result.issues[0]["start"] == 0.0 and result.issues[0]["note"].startswith("カナ")

    def test_known_window_is_not_sent_again(self) -> None:
        first = finalize_subtitles(
            _split_sentence(), "ja", 16, _CFG, _LLM, provider=_FakeProvider([_reply()])
        )
        provider = _FakeProvider([])
        again = finalize_subtitles(
            _split_sentence(), "ja", 16, _CFG, _LLM, known=first.records, provider=provider
        )
        assert provider.calls == 0
        assert again.stats.cached == 2


class TestFailures:
    def test_provider_failure_keeps_rules_result(self) -> None:
        from src.core.exceptions import LLMSegmentationError

        class _Failing(_FakeProvider):
            def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
                raise LLMSegmentationError("인증 실패")

        segs = _split_sentence()
        before = [s.text for s in segs]
        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_Failing([]))
        assert [s.text for s in segs] == before
        assert result.stats.failed == 1 and result.records == {}

    def test_unreadable_reply_is_skipped(self) -> None:
        segs = _split_sentence()
        result = finalize_subtitles(segs, "ja", 16, _CFG, _LLM, provider=_FakeProvider(["no json"]))
        assert result.stats.failed == 1 and result.boundaries == 0


class TestPipeline:
    def test_runs_only_when_enabled_and_saves_review(self, tmp_path: Path) -> None:
        from src.models.result import TranscriptionResult
        from src.pipeline import TranscriptionPipeline
        from src.postprocess.llm_finalize import FinalizeResult

        words = [WordSegment("こんにちは", 0.0, 1.5, 0.9)]
        segments = [TranscriptionSegment(1, "こんにちは", 0.0, 1.5, 0.9, words=words)]
        config = AppConfig()
        config.output.dir = str(tmp_path)
        issue = {"start": 0.0, "end": 1.5, "text": "こんにちは", "type": "other", "note": "메모"}
        with patch(
            "src.postprocess.llm_finalize.finalize_subtitles",
            return_value=FinalizeResult(issues=[issue]),
        ) as finalize:
            pipeline = TranscriptionPipeline(config)
            pipeline._segment(segments, "ja", None)
            finalize.assert_not_called()
            config.postprocess.llm_review = True
            pipeline._segment(segments, "ja", None)
            finalize.assert_called_once()

        result = TranscriptionResult(segments=segments)
        result.metadata.source_file = "/x/video.mkv"
        path = pipeline._save_review(result, None)
        assert path == tmp_path / "video.review.json"
        assert json.loads(path.read_text(encoding="utf-8"))["issues"] == [issue]
