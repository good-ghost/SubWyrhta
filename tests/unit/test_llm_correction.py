"""LLM 오인식 교정·다시 정렬·용어 목록·읽기 비교 테스트 (WI-9.018, 2026-10-11).

LLM 응답은 가짜 프로바이더로, whisperx 정렬은 가짜 모듈로 만든다. 사례는 검수 리포트에서 본 유형을 일반 예문으로 옮겼다.
"""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from src.core.config import LLMConfig
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.postprocess import reading
from src.postprocess.glossary import GLOSSARY_ENV, load_glossary
from src.postprocess.llm_correction import (
    apply_correction,
    correct_transcript,
    is_acceptable_correction,
)
from src.postprocess.realign import realign_segments
from src.segmentation.llm_client import LLMProvider, LLMResponse

if TYPE_CHECKING:
    from pathlib import Path

_LLM = LLMConfig(provider="fake", max_retries=0)
# 반영 모드를 켜는 용어 목록 (교정 대상과 상관없는 용어)
_GLOSSARY = ["サトシ"]


class _FakeProvider(LLMProvider):
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.prompts.append(system_prompt + user_prompt)
        return LLMResponse(self.replies.pop(0), input_tokens=10, output_tokens=5)

    def provider_name(self) -> str:
        return "fake"


def _reply(*items: tuple[int, str, str]) -> str:
    return json.dumps(
        {
            "corrections": [
                {"id": i, "from": f, "to": t, "reason": "읽기 같음"} for i, f, t in items
            ]
        },
        ensure_ascii=False,
    )


def _seg(*phrases: str, start: float = 0.0) -> TranscriptionSegment:
    words, t = [], start
    for phrase in phrases:
        words.append(WordSegment(phrase, t, t + 0.5, 0.9))
        t += 0.6
    return TranscriptionSegment(1, "".join(phrases), start, t, 0.9, words=words)


class TestReading:
    @pytest.mark.parametrize(
        ("a", "b", "language", "close"),
        [
            ("以外な", "意外な", "ja", True),  # 읽기 같음 (イガイナ)
            ("サトミ", "サトシ", "ja", True),  # 1음 차이
            ("コンピューダー", "コンピューター", "ja", True),  # 탁점 차이
            ("テレビ", "ラジオ", "ja", False),  # 소리가 다르다 → 리포트에만
            ("smartphone", "smart phone", "en", True),
            ("recieve", "receive", "en", True),
            ("table", "chair", "en", False),
            ("사토미", "사토시", "ko", True),
            ("사과", "바나나", "ko", False),
        ],
    )
    def test_distance(self, a: str, b: str, language: str, close: bool) -> None:
        ratio = reading.reading_distance_ratio(a, b, language)
        assert ratio is not None
        assert (ratio <= 0.34) is close

    def test_other_languages_are_extensible(self) -> None:
        assert not reading.is_supported("xx")
        assert reading.reading_distance_ratio("a", "b", "xx") is None
        reading.register_reading("xx", str.lower)
        try:
            assert reading.reading_distance_ratio("AB", "ab", "xx") == 0.0
        finally:
            reading._READERS.pop("xx")


class TestAcceptable:
    def test_accepts_same_reading(self) -> None:
        assert is_acceptable_correction("とても以外な", "以外な", "意外な", "ja")

    @pytest.mark.parametrize(
        ("line", "frm", "to"),
        [
            ("家でテレビを見る", "テレビ", "ラジオ"),  # 소리가 다르다
            ("サトミとサトミ", "サトミ", "サトシ"),  # 두 번 나온다
            ("第11回", "第11回", "第12回"),  # 숫자가 바뀐다
            ("いいね", "いいね", "いいね。"),  # 부호를 더한다
            ("とても以外な", "以外", ""),  # 지운다
        ],
    )
    def test_rejects(self, line: str, frm: str, to: str) -> None:
        assert not is_acceptable_correction(line, frm, to, "ja")

    @pytest.mark.parametrize(
        ("line", "frm", "to"),
        [
            # 실제로 있던 응답 유형: 앞뒤 문장을 붙여 거리 비율을 낮췄다 (はじめ / タナカ 는 소리가 다르다)
            (
                "今日はじめと行ける僕は",
                "今日はじめと行ける僕は",
                "今日タナカと行ける僕は",
            ),
            ("電車が本当に遅れるなら", "電車が本当に遅れるなら", "台風が本当に遅れるなら"),
        ],
    )
    def test_long_context_does_not_dilute_distance(self, line: str, frm: str, to: str) -> None:
        assert not is_acceptable_correction(line, frm, to, "ja")

    def test_word_bounds_of_corrected_text(self) -> None:
        """바뀌기 전 표기가 잘게 나뉘어도 고친 표기의 단어로 비교한다 (`図書間` / `図書館`, 둘 다 トショカン)."""
        line = "新しい図書間に行く"
        assert is_acceptable_correction(line, "図書間に", "図書館に", "ja")

    def test_context_is_fine_when_sound_is_same(self) -> None:
        line = "防犯カメラの視覚に入らない"
        assert is_acceptable_correction(line, "防犯カメラの視覚", "防犯カメラの死角", "ja")

    @pytest.mark.parametrize(
        ("line", "frm", "to"),
        [
            ("さくらが", "さくら", "桜"),  # 같은 읽기로 표기만 바꾼다 (가나 → 한자)
            ("構わせて", "構わせて", "かまわせて"),  # 표기만
            ("たつやなんだ", "たつや", "達也"),  # 이름 표기 (용어 목록에 없으면)
        ],
    )
    def test_script_only_change_is_rejected(self, line: str, frm: str, to: str) -> None:
        assert not is_acceptable_correction(line, frm, to, "ja")

    def test_script_only_change_allowed_for_glossary_term(self) -> None:
        assert is_acceptable_correction("たつやなんだ", "たつや", "達也", "ja", ["達也"])

    def test_glossary_widens_limit(self) -> None:
        line = "ルミナリーダーが人々の生活に"
        assert not is_acceptable_correction(line, "ルミナリーダー", "ルミナリンク", "ja")
        assert is_acceptable_correction(
            line, "ルミナリーダー", "ルミナリンク", "ja", ["ルミナリンク"]
        )


class TestApply:
    def test_inside_one_word(self) -> None:
        seg = _seg("だからサトミを", "呼んだ")
        assert apply_correction(seg, "サトミ", "サトシ")
        assert [w.word for w in seg.words] == ["だからサトシを", "呼んだ"]
        assert seg.text == "だからサトシを呼んだ"

    def test_across_words_merges(self) -> None:
        seg = _seg("とても", "以外", "な")
        assert apply_correction(seg, "以外な", "意外な")
        assert [w.word for w in seg.words] == ["とても", "意外な"]
        assert (seg.words[1].start, seg.words[1].end) == (0.6, 1.7)

    def test_english_spaces(self) -> None:
        seg = TranscriptionSegment(
            1,
            "I recieve it",
            0.0,
            1.0,
            0.9,
            words=[
                WordSegment(w, i * 0.3, i * 0.3 + 0.2, 0.9)
                for i, w in enumerate(["I", "recieve", "it"])
            ],
        )
        assert apply_correction(seg, "recieve", "receive")
        assert seg.text == "I receive it"


class TestCorrectTranscript:
    def test_applies_valid_and_caches(self) -> None:
        segs = [_seg("とても", "以外", "な"), _seg("家で", "テレビを", "見る", start=5.0)]
        provider = _FakeProvider([_reply((1, "以外な", "意外な"), (2, "テレビ", "ラジオ"))])

        result = correct_transcript(segs, "ja", _LLM, _GLOSSARY, provider=provider)

        assert segs[0].text == "とても意外な" and segs[1].text == "家でテレビを見る"
        assert result.corrected == [segs[0]]
        assert (result.stats.changed, result.stats.rejected) == (1, 1)
        assert result.applied[0]["from"] == "以外な" and result.applied[0]["to"] == "意外な"
        # 단계 캐시에는 검증 전 제안을 모두 남긴다
        assert result.edits["家でテレビを見る"] == [["テレビ", "ラジオ", "읽기 같음"]]
        assert result.suggested[0]["to"] == "ラジオ" and result.suggested[0]["verified"] is False

    def test_without_glossary_only_reports(self) -> None:
        """용어 목록이 없으면 반영하지 않고 모든 제안을 리포트에 남긴다 (TO-DO 19, 수동 교정)."""
        segs = [_seg("とても", "以外", "な"), _seg("家で", "テレビを", "見る", start=5.0)]
        provider = _FakeProvider([_reply((1, "以外な", "意外な"), (2, "テレビ", "ラジオ"))])

        result = correct_transcript(segs, "ja", _LLM, provider=provider)

        assert segs[0].text == "とても以外な" and result.corrected == [] and result.applied == []
        assert [(s["from"], s["verified"]) for s in result.suggested] == [
            ("以外な", True),
            ("テレビ", False),
        ]
        assert result.suggested[0]["text"] == "とても以外な"

    def test_known_lines_are_not_sent(self) -> None:
        segs = [_seg("とても", "以外", "な")]
        provider = _FakeProvider([])
        known = {"とても以外な": [["以外な", "意外な", "x"]]}
        result = correct_transcript(segs, "ja", _LLM, _GLOSSARY, known=known, provider=provider)
        assert provider.prompts == []
        assert segs[0].text == "とても意外な" and result.stats.cached == 1

    def test_cached_edit_is_revalidated(self) -> None:
        """예전 기준으로 받은 교정이 캐시에 있어도 지금 기준에 맞지 않으면 넣지 않는다."""
        segs = [_seg("今日", "はじめと", "行ける")]
        known = {"今日はじめと行ける": [["今日はじめと行ける", "今日タナカと行ける", "x"]]}
        result = correct_transcript(
            segs, "ja", _LLM, _GLOSSARY, known=known, provider=_FakeProvider([])
        )
        assert segs[0].text == "今日はじめと行ける" and result.stats.changed == 0

    def test_glossary_in_prompt(self) -> None:
        provider = _FakeProvider([_reply()])
        correct_transcript(
            [_seg("こんにちは")], "ja", _LLM, ["サトシ", "ルミナリンク"], provider=provider
        )
        assert "サトシ, ルミナリンク" in provider.prompts[0]

    def test_unsupported_language_is_skipped(self) -> None:
        provider = _FakeProvider([])
        result = correct_transcript([_seg("你好")], "zh", _LLM, provider=provider)
        assert provider.prompts == [] and result.stats.windows == 0


class TestGlossary:
    def test_unset_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(GLOSSARY_ENV, raising=False)
        assert load_glossary("x.mkv") == []

    def test_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "terms.txt"
        path.write_text("# 인물\nサトシ\n\nルミナリンク\nサトシ\n", encoding="utf-8")
        monkeypatch.setenv(GLOSSARY_ENV, str(path))
        assert load_glossary() == ["サトシ", "ルミナリンク"]

    def test_directory_common_and_per_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "common.txt").write_text("サトシ\n", encoding="utf-8")
        (tmp_path / "My Show - S01E01.txt").write_text("ルミナリンク\n", encoding="utf-8")
        monkeypatch.setenv(GLOSSARY_ENV, str(tmp_path))
        assert load_glossary("/v/My Show - S01E01.mkv") == ["サトシ", "ルミナリンク"]
        assert load_glossary("/v/other.mkv") == ["サトシ"]


class TestRealign:
    def _fake_whisperx(self, words: list[dict[str, Any]]) -> types.ModuleType:
        module = types.ModuleType("whisperx")
        module.load_align_model = lambda language_code, device: (
            "model",
            {"language": language_code},
        )  # type: ignore[attr-defined]
        module.load_audio = lambda path: "audio"  # type: ignore[attr-defined]
        module.align = lambda segs, model, meta, audio, device, return_char_alignments: {  # type: ignore[attr-defined]
            "segments": [{"words": words}]
        }
        return module

    def test_replaces_words_with_aligned_times(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seg = _seg("とても", "意外な")
        aligned = [
            {"word": ch, "start": 1.0 + i * 0.1, "end": 1.08 + i * 0.1, "score": 0.9}
            for i, ch in enumerate("とても意外な")
        ]
        aligned[3].pop("start")  # 시각이 없는 글자는 앞 글자 끝에 붙인다
        monkeypatch.setitem(sys.modules, "whisperx", self._fake_whisperx(aligned))

        assert realign_segments([seg], tmp_path / "a.wav", "ja", "cpu") == 1
        assert "".join(w.word for w in seg.words) == "とても意外な"
        assert seg.start == pytest.approx(1.0)
        assert seg.words[3].start == seg.words[2].end

    def test_mismatched_text_keeps_old_words(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seg = _seg("とても", "意外な")
        before = [(w.word, w.start, w.end) for w in seg.words]
        aligned = [{"word": "とても", "start": 1.0, "end": 1.2}]
        monkeypatch.setitem(sys.modules, "whisperx", self._fake_whisperx(aligned))
        assert realign_segments([seg], tmp_path / "a.wav", "ja", "cpu") == 0
        assert [(w.word, w.start, w.end) for w in seg.words] == before


class TestPipeline:
    def test_correction_runs_and_realigns(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from src.core.config import AppConfig
        from src.pipeline import TranscriptionPipeline
        from src.postprocess.llm_correction import CorrectionResult

        seg = _seg("とても", "以外", "な")
        config = AppConfig()
        config.postprocess.llm_correction = True
        config.noise_filter.enabled = False
        applied = {"start": 0.0, "end": 1.7, "from": "以外な", "to": "意外な", "reason": ""}
        suggested = {
            "start": 0.0,
            "end": 1.7,
            "text": "x",
            "from": "a",
            "to": "b",
            "reason": "",
            "verified": False,
        }
        with (
            patch(
                "src.postprocess.llm_correction.correct_transcript",
                return_value=CorrectionResult(
                    corrected=[seg], applied=[applied], suggested=[suggested]
                ),
            ) as correct,
            patch("src.postprocess.realign.realign_segments", return_value=1) as realign,
        ):
            pipeline = TranscriptionPipeline(config)
            pipeline._postprocess_stt([seg], "ja", None, None, "x.mkv", tmp_path / "a.wav")
        correct.assert_called_once()
        realign.assert_called_once()
        assert pipeline._review_corrections == [applied]
        assert pipeline._review_suggestions == [suggested]


class TestBreaker:
    """작업 단위 LLM 차단기 (TO-DO 20): 한 단계가 재시도까지 실패하면 이후 단계는 호출하지 않는다."""

    def test_failure_skips_later_stages_but_uses_cache(self) -> None:
        from src.core.exceptions import LLMSegmentationError
        from src.postprocess.llm_punctuation import punctuate_transcript
        from src.postprocess.llm_text import LLMBreaker

        class _Failing(_FakeProvider):
            def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
                self.prompts.append(user_prompt)
                raise LLMSegmentationError("timed out")

        breaker = LLMBreaker()
        failing = _Failing([])
        correct_transcript(
            [_seg("とても", "以外", "な")], "ja", _LLM, provider=failing, breaker=breaker
        )
        assert breaker.is_open and "timed out" in breaker.reason
        assert len(failing.prompts) == 1

        # 다음 단계는 호출하지 않는다. 단계 캐시에 있는 문장은 쓴다
        later = _FakeProvider([])
        segs = [_seg("終わったよ", "そうか"), _seg("いや", "何でもない", start=5.0)]
        known = {"終わったよそうか": "終わったよ。そうか"}
        result = punctuate_transcript(segs, "ja", _LLM, known, later, breaker=breaker)
        assert later.prompts == []
        assert segs[0].text == "終わったよ。そうか" and segs[1].text == "いや何でもない"
        assert result.stats.failed == 1

    def test_pipeline_writes_error_to_review(self, tmp_path: Path) -> None:
        from src.core.config import AppConfig
        from src.models.result import TranscriptionResult
        from src.pipeline import TranscriptionPipeline

        config = AppConfig()
        config.output.dir = str(tmp_path)
        pipeline = TranscriptionPipeline(config)
        pipeline._llm_breaker.trip("Request timed out.")
        result = TranscriptionResult(segments=[])
        result.metadata.source_file = "/x/video.mkv"
        data = json.loads(pipeline._save_review(result, None).read_text(encoding="utf-8"))
        assert data["llm_error"] == "Request timed out."
