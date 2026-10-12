"""LLM 부호 넣기: 세그멘테이션 전에 STT 문장에 문장 부호를 넣는다 (선택 단계, 기본 꺼짐).

Whisper 는 빠른 다화자 대사를 부호 없이 13~17초짜리 한 덩어리로 낸다
(`…駅に着いたら電話してわかったすぐ行くよ…`). advanced 분할의 절 경계 점수와 CPS 분할의
우선순위가 문장 부호라, 부호가 없으면 쉼·글자 수로만 엉뚱한 곳에서 나뉜다. 줄바꿈 뒤의 LLM 검토는 이미
정한 자막 경계 안에서만 부호를 넣을 수 있어, 경계를 정하기 전(``TranscriptionPipeline._postprocess_stt``,
노이즈 필터 뒤)에 넣는다.

LLM 은 부호만 더하고, 코드가 원문 글자 보존·부호 겹침 없음을 확인한 뒤 받는다. 더한 부호는 바로 앞
글자가 있는 단어(일본어는 문절)에 넣는다. 시각은 바꾸지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import structlog

from src.core.text import join_words
from src.models.word import WordSegment
from src.postprocess.llm_text import (
    NO_SPACE_LANGUAGES,
    LLMBreaker,
    LLMTextStats,
    addable_punctuation,
    added_punctuation,
    collect_changes,
    comparable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.config import LLMConfig
    from src.models.segment import TranscriptionSegment
    from src.segmentation.llm_client import LLMProvider

log = structlog.get_logger(__name__)

# 이 글자 바로 뒤에는 마침표를 받지 않는다 (TO-DO 13). 목적격 조사 `を` 로 문장이 끝나는 일은 거의 없는데
# LLM 이 다음 자막으로 이어지는 문장에 넣었다 (`考えていることを。`)
_NO_SENTENCE_END = frozenset("を")
_FULL_STOPS = frozenset("。.")

PUNCTUATION_PROMPT = """\
You add punctuation to a speech recognition transcript. The lines are consecutive utterances of \
one video, in order. A line may contain several sentences or several speakers run together \
without punctuation.
Add punctuation ({punctuation}): a full stop where a sentence ends (also at the end of a line \
when the sentence ends there), a comma after an interjection or a name being called or between \
clauses, a question mark after a question. Do not put a full stop where the sentence continues \
into the next line (for example after を or a continuing te-form).
Never change, delete, add or reorder any other character. Never remove existing punctuation. \
Never add punctuation next to existing punctuation.
Example (Japanese): "駅に着いたら電話してわかったすぐ行くよ田中さんどうかしたの?いや何でもない" -> \
"駅に着いたら電話して。わかった、すぐ行くよ。田中さん、どうかしたの?いや、何でもない。"
Reply with JSON only: {{"changes": [{{"id": <id>, "text": "<new text>"}}]}}. \
Include only lines you changed. If nothing needs to change, reply {{"changes": []}}.
"""


@dataclass
class PunctuationResult:
    """LLM 부호 넣기 결과.

    Attributes:
        stats: 집계.
        texts: 원문 → 부호를 넣은 문장 (받지 않았거나 고칠 것이 없으면 원문). 단계 캐시에 남겨
            다시 시도하거나 세그멘테이션만 다시 돌릴 때 같은 문장을 다시 보내지 않는다.
            응답을 받지 못한 묶음의 문장은 넣지 않는다 (다음에 다시 보낸다).
    """

    stats: LLMTextStats = field(default_factory=LLMTextStats)
    texts: dict[str, str] = field(default_factory=dict)


def punctuate_transcript(
    segments: Sequence[TranscriptionSegment],
    language: str,
    llm_config: LLMConfig,
    known: dict[str, str] | None = None,
    provider: LLMProvider | None = None,
    breaker: LLMBreaker | None = None,
) -> PunctuationResult:
    """STT 세그먼트에 LLM 으로 문장 부호를 넣는다 (in-place: 단어와 텍스트).

    Args:
        segments: STT 세그먼트 (단어가 있어야 부호를 넣는다).
        language: 언어 코드.
        llm_config: LLM 프로바이더 설정.
        known: 이전에 받은 결과 (원문 → 부호를 넣은 문장). 여기 있는 문장은 보내지 않는다.
        provider: LLM 프로바이더 (테스트용). None 이면 설정으로 만든다.
        breaker: 작업 단위 LLM 차단기 (열려 있으면 호출하지 않는다).

    Returns:
        결과 (집계, 원문 → 부호를 넣은 문장).
    """
    result = PunctuationResult(texts=dict(known or {}))
    targets = [seg for seg in segments if seg.words]
    originals = [join_words(w.word for w in seg.words) for seg in targets]
    pending = sorted({text for text in originals if text not in result.texts})
    result.stats.cached = len(originals) - sum(1 for text in originals if text in pending)

    if pending:
        system_prompt = PUNCTUATION_PROMPT.format(
            punctuation=" ".join(addable_punctuation(language))
        )
        changes, answered = collect_changes(
            pending,
            system_prompt,
            "lines",
            language,
            llm_config,
            result.stats,
            provider,
            breaker=breaker,
        )
        for index in answered:
            original = pending[index]
            proposed = _tidy(original, changes.get(index, original), language)
            if proposed != original and _added(original, proposed, language) is None:
                result.stats.rejected += 1
                log.debug("LLM 부호 거부", original=original, proposed=proposed)
                proposed = original
            result.texts[original] = proposed

    for seg, original in zip(targets, originals, strict=True):
        punctuated = result.texts.get(original, original)
        if punctuated != original and _apply(seg, original, punctuated, language):
            result.stats.changed += 1

    log.info(
        "LLM 부호 넣기 완료",
        segments=len(targets),
        changed=result.stats.changed,
        rejected=result.stats.rejected,
        cached=result.stats.cached,
        windows=result.stats.windows,
        failed=result.stats.failed,
        input_tokens=result.stats.input_tokens,
        output_tokens=result.stats.output_tokens,
    )
    return result


def _tidy(original: str, proposed: str, language: str) -> str:
    """공백 없는 언어에 넣은 반각 공백을 뗀다."""
    text = proposed.strip()
    if language in NO_SPACE_LANGUAGES and " " not in original:
        text = text.replace(" ", "")
    return text


def _added(original: str, proposed: str, language: str) -> list[tuple[int, str]] | None:
    """더한 부호 목록 (부호만 더한 게 아니면 None). 문장이 끝날 수 없는 자리의 마침표는 뺀다."""
    compact = comparable(original, language)
    added = added_punctuation(compact, comparable(proposed, language), language)
    if added is None:
        return None
    return [
        (after, mark)
        for after, mark in added
        if not (mark in _FULL_STOPS and compact[after - 1 : after] in _NO_SENTENCE_END)
    ]


def _apply(seg: TranscriptionSegment, original: str, punctuated: str, language: str) -> bool:
    """더한 부호를 단어에 넣고 텍스트를 다시 만든다. 단어와 원문이 맞지 않으면 False (그대로 둔다)."""
    added = _added(original, punctuated, language)
    if not added:
        return False
    # 비교용 글자(공백 제외)마다 그 글자가 있는 단어와 단어 안 위치
    owners: list[tuple[int, int]] = []
    for word_index, word in enumerate(seg.words):
        owners.extend((word_index, offset) for offset in range(len(word.word.strip())))
    if len(owners) != len(comparable(original, language).replace(" ", "")):
        log.warning("단어와 원문이 맞지 않아 LLM 부호를 넣지 않음", text=original[:40])
        return False

    inserts: dict[int, list[tuple[int, str]]] = {}
    compact = comparable(original, language)
    for after, mark in added:
        # added 는 비교용 원문(언어에 따라 공백 포함)의 위치라, 공백을 뺀 글자 수로 바꾼다
        position = len(compact[:after].replace(" ", ""))
        if position == 0:
            continue  # 맨 앞에 붙은 부호는 붙일 단어가 없다
        word_index, offset = owners[position - 1]
        inserts.setdefault(word_index, []).append((offset + 1, mark))
    words: list[WordSegment] = []
    for word_index, word in enumerate(seg.words):
        marks = inserts.get(word_index)
        words.extend(_insert_marks(word, marks) if marks else [word])
    seg.words = words
    seg.text = join_words(w.word for w in seg.words)
    return bool(inserts)


def _insert_marks(word: WordSegment, marks: list[tuple[int, str]]) -> list[WordSegment]:
    """단어 안 위치(앞에서 몇 글자 뒤)에 부호를 넣는다. 단어 가운데 넣은 부호 뒤에서는 단어를 나눈다.

    문절 묶기가 부호 없는 문장을 한 문절로 묶으면(`見ただけだだから`) 부호가 단어 가운데 들어간다.
    그대로 두면 절 경계 점수(단어 끝 부호)와 경계 옮기기가 그 자리를 쓰지 못한다. 나눈 단어의 시각은
    글자 수 비율로 나눈다.
    """
    text = word.word.strip()
    for offset, mark in sorted(marks, reverse=True):
        text = text[:offset] + mark + text[offset:]
    cuts = sorted(offset + i + 1 for i, (offset, _) in enumerate(sorted(marks)))
    pieces = [text[a:b] for a, b in pairwise([0, *(c for c in cuts if c < len(text)), len(text)])]
    pieces = [piece for piece in pieces if piece]
    if len(pieces) == 1:
        word.word = text
        return [word]
    result: list[WordSegment] = []
    position = word.start
    for piece in pieces:
        end = position + word.duration * len(piece) / len(text)
        result.append(WordSegment(piece, position, end, word.confidence))
        position = end
    result[-1].end = word.end
    return result
