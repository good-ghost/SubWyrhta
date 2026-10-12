"""LLM 오인식 교정: 소리는 맞고 표기가 틀린 말을 세그멘테이션 전에 고친다 (선택 단계, 기본 꺼짐).

WhisperX 가 소리는 맞게 들었지만 표기를 틀린 말(`機会` → `機械`, `以外` → `意外`,
`コンピューダー` → `コンピューター`)을 LLM 이 찾아 고친다 (WI-9.018, REQUIREMENT 6.2.3). 순서는 노이즈 필터 →
오인식 교정 → 다시 정렬 → LLM 부호 넣기.

LLM 은 바꿀 구간만 돌려주고(`from` → `to`), 코드가 검증한 교정만 받는다:
- `from` 이 그 문장에 한 번만 나온다
- 읽기(``reading``)가 같거나 가깝다: 편집 거리 비율 ≤ 0.34, 용어 목록의 표기로 바꾸면 ≤ 0.5
- 숫자·문장 부호는 그대로, 길이 변화는 max(3, 0.34 x 원래 길이) 이하, 한 문장에 교정 2개까지
- 일본어에서 읽기가 같고 한쪽이 가나뿐인 교정(표기만 바꿈)은 용어 목록에 있을 때만 받는다
받은 교정은 단어 글자를 바꾸고(여러 단어에 걸치면 합친다), 파이프라인이 그 문장을 다시 정렬한다.
용어 목록이 없으면 반영하지 않고 모든 제안을 검수 리포트(``suggested``)에 남긴다 (수동 교정, TO-DO 19).
읽기 함수를 등록하지 않은 언어는 교정하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from src.core.text import join_words
from src.models.word import WordSegment
from src.postprocess.llm_text import (
    DEFAULT_NOTE_LANGUAGE,
    LLMBreaker,
    LLMTextStats,
    localize_notes,
    note_language_rule,
    request_windows,
)
from src.postprocess.reading import is_supported, reading_distance_ratio

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.config import LLMConfig
    from src.models.segment import TranscriptionSegment
    from src.segmentation.llm_client import LLMProvider

log = structlog.get_logger(__name__)

# 읽기 편집 거리 비율 상한. 용어 목록의 표기로 바꾸는 교정은 넓힌다 (고유명사는 소리가 꽤 달라도 맞다)
MAX_READING_RATIO = 0.34
MAX_READING_RATIO_GLOSSARY = 0.5
# 길이 변화 상한 (글자) 과 비율
_MAX_LENGTH_CHANGE = 3
_MAX_LENGTH_CHANGE_RATIO = 0.34
# 한 문장에 받는 교정 수
_MAX_EDITS_PER_LINE = 2
# 교정 전후로 같아야 하는 글자 (숫자·문장 부호)
_FIXED_CHARS = frozenset("0123456789、。，,.!?！？…")

CORRECTION_PROMPT = """\
You check a speech recognition transcript for words that were heard correctly but written \
wrongly: a wrong kanji or katakana for the same sound, a misspelled name or term, a word that \
does not fit the context but sounds almost the same as the right word. The lines are \
consecutive utterances of one video, in order.
Only fix words whose correct form sounds the same or almost the same. Do not rephrase, do not fix \
grammar or style, do not add or remove words, do not change punctuation or numbers.
{glossary}Reply with JSON only: {{"corrections": [{{"id": <line id>, "from": "<exact text in the \
line>", "to": "<corrected text>", "reason": "<short reason>"}}]}}. "from" must be copied \
exactly from the line. If nothing needs to change, reply {{"corrections": []}}.
{note_language}"""
_GLOSSARY_SECTION = "Names and terms used in this video (prefer these spellings): {terms}\n"


@dataclass
class CorrectionResult:
    """LLM 오인식 교정 결과.

    Attributes:
        stats: 호출·캐시 집계 (``changed`` 는 반영한 교정 수, ``rejected`` 는 검증에서 떨어진 제안 수).
        edits: 원문 → LLM 제안 [[from, to, reason], ...] (검증 전). 단계 캐시에 남긴다. 반영·보고할 때
            지금 기준으로 검증한다 (응답을 받지 못한 묶음의 문장은 넣지 않아 다음에 다시 보낸다).
        corrected: 교정한 세그먼트 (파이프라인이 다시 정렬한다). 용어 목록이 없으면 비어 있다.
        applied: 반영한 교정 (검수 리포트 ``corrected``: 세그먼트 시각, from, to, 이유).
        suggested: 반영하지 않은 제안 (검수 리포트 ``suggested``: 문장, from, to, 이유, 검증 통과 여부).
            용어 목록이 없으면 모든 제안이 여기로 간다 (수동 교정용).
    """

    stats: LLMTextStats = field(default_factory=LLMTextStats)
    edits: dict[str, list[list[str]]] = field(default_factory=dict)
    corrected: list[TranscriptionSegment] = field(default_factory=list)
    applied: list[dict[str, Any]] = field(default_factory=list)
    suggested: list[dict[str, Any]] = field(default_factory=list)


def correct_transcript(
    segments: Sequence[TranscriptionSegment],
    language: str,
    llm_config: LLMConfig,
    glossary: Sequence[str] = (),
    known: dict[str, list[list[str]]] | None = None,
    provider: LLMProvider | None = None,
    breaker: LLMBreaker | None = None,
    note_language: str = DEFAULT_NOTE_LANGUAGE,
) -> CorrectionResult:
    """STT 세그먼트의 오인식을 LLM 으로 찾는다. 용어 목록이 있으면 검증한 교정을 반영한다 (in-place).

    2026-10-11 사용자 결정 (TO-DO 19): 용어 목록이 없으면 교정을 반영하지 않고 모든 제안을 검수 리포트에
    남긴다 (수동 교정). 작품 고유 용어를 LLM 이 모르면 소리가 비슷한 일반어로 잘못 고쳤다
    (작품 속 가상의 용어를 `総選挙` 같은 일반어로).

    Args:
        segments: STT 세그먼트 (단어가 있어야 교정한다).
        language: 언어 코드 (읽기 함수가 없으면 교정하지 않는다).
        llm_config: LLM 프로바이더 설정.
        glossary: 용어 목록 (``glossary.load_glossary``). 비어 있으면 보고만 한다.
        known: 이전에 받은 제안 (원문 → [[from, to, reason]]). 여기 있는 문장은 보내지 않는다.
        provider: LLM 프로바이더 (테스트용). None 이면 설정으로 만든다.
        breaker: 작업 단위 LLM 차단기 (열려 있으면 호출하지 않는다).
        note_language: 교정 이유를 쓸 언어 코드 (리포트를 읽는 화면 언어. 자막 언어가 아니다).

    Returns:
        결과 (집계, 단계 캐시에 남길 제안, 교정한 세그먼트, 리포트 항목).
    """
    result = CorrectionResult(edits=dict(known or {}))
    if not is_supported(language):
        log.info("읽기 비교를 지원하지 않는 언어라 오인식 교정을 건너뜀", language=language)
        return result
    targets = [seg for seg in segments if seg.words]
    originals = [join_words(w.word for w in seg.words) for seg in targets]
    pending = sorted({text for text in originals if text not in result.edits})
    result.stats.cached = len(originals) - sum(1 for text in originals if text in pending)
    terms = list(glossary)
    apply = bool(terms)

    if pending:
        glossary_text = _GLOSSARY_SECTION.format(terms=", ".join(terms)) if terms else ""
        prompt = CORRECTION_PROMPT.format(
            glossary=glossary_text,
            note_language=note_language_rule(note_language, "reason", "lines"),
        )
        answers, _ = request_windows(
            [{"text": t} for t in pending],
            prompt,
            "lines",
            language,
            llm_config,
            result.stats,
            provider,
            breaker=breaker,
        )
        for start, count, data in answers:
            raw: dict[int, list[list[str]]] = {i: [] for i in range(start, start + count)}
            for id_, frm, to, reason in _correction_items(data.get("corrections")):
                if id_ - 1 in raw:
                    raw[id_ - 1].append([frm, to, reason])
            for index, edits in raw.items():
                result.edits[pending[index]] = edits
        # 이유가 자막 언어로 온 제안은 메모 언어로 옮겨 단계 캐시에 남긴다
        fresh = [edit for text in pending for edit in result.edits.get(text, [])]
        reasons, _ = localize_notes(
            [edit[2] for edit in fresh],
            note_language,
            llm_config,
            result.stats,
            provider,
            breaker=breaker,
        )
        for edit, reason in zip(fresh, reasons, strict=True):
            edit[2] = reason

    for seg, original in zip(targets, originals, strict=True):
        accepted = 0
        for frm, to, reason in result.edits.get(original, []):
            # 캐시에서 꺼낸 제안도 지금 기준·용어 목록으로 검증한다
            verified = is_acceptable_correction(original, frm, to, language, terms)
            if not verified:
                result.stats.rejected += 1
            if (
                apply
                and verified
                and accepted < _MAX_EDITS_PER_LINE
                and apply_correction(seg, frm, to)
            ):
                accepted += 1
                result.stats.changed += 1
                result.applied.append(
                    {
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "from": frm,
                        "to": to,
                        "reason": reason[:300],
                    }
                )
                continue
            result.suggested.append(
                {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": original,
                    "from": frm,
                    "to": to,
                    "reason": reason[:300],
                    "verified": verified,
                }
            )
        if accepted:
            result.corrected.append(seg)

    log.info(
        "LLM 오인식 교정 완료"
        if apply
        else "LLM 오인식 교정 제안 (용어 목록이 없어 반영하지 않음)",
        segments=len(targets),
        corrected=result.stats.changed,
        suggested=len(result.suggested),
        rejected=result.stats.rejected,
        cached=result.stats.cached,
        windows=result.stats.windows,
        failed=result.stats.failed,
        glossary=len(terms),
        input_tokens=result.stats.input_tokens,
        output_tokens=result.stats.output_tokens,
    )
    return result


def is_acceptable_correction(
    line: str, frm: str, to: str, language: str, glossary: Sequence[str] = ()
) -> bool:
    """LLM 교정을 받아도 되는지 (위치가 분명하고, 읽기가 가깝고, 숫자·부호·길이가 그대로).

    Args:
        line: 교정할 문장.
        frm: LLM 이 고르라고 한 원래 구간.
        to: 고친 텍스트.
        language: 언어 코드.
        glossary: 용어 목록 (이 표기로 바꾸면 읽기 기준을 넓힌다).

    Returns:
        받아도 되면 True.
    """
    if not frm or not to or frm == to or line.count(frm) != 1:
        return False
    if sorted(ch for ch in frm if ch in _FIXED_CHARS) != sorted(
        ch for ch in to if ch in _FIXED_CHARS
    ):
        return False
    if abs(len(to) - len(frm)) > max(
        _MAX_LENGTH_CHANGE, round(_MAX_LENGTH_CHANGE_RATIO * len(frm))
    ):
        return False
    core_from, core_to = changed_words(frm, to, language)
    ratio = reading_distance_ratio(core_from, core_to, language)
    if ratio is None:
        return False
    in_glossary = any(term and (term in core_to or core_to in term) for term in glossary)
    if (
        language == "ja"
        and ratio == 0
        and not in_glossary
        and (_is_kana(core_from) or _is_kana(core_to))
    ):
        # 읽기가 같고 한쪽이 가나뿐이면 표기만 바꾼 것이다. 오인식이 아니고, LLM 이 같은 읽기의 다른 말로
        # 순화하기도 했다 (가나로 쓴 말을 같은 읽기의 인명 한자로). 이름·용어는 용어 목록에 있으면 받는다
        return False
    return ratio <= (MAX_READING_RATIO_GLOSSARY if in_glossary else MAX_READING_RATIO)


def _is_kana(text: str) -> bool:
    """히라가나·가타카나·장음 부호로만 된 텍스트."""
    return bool(text) and all("\u3041" <= ch <= "\u30ff" or ch == "\u30fc" for ch in text)


def changed_words(frm: str, to: str, language: str) -> tuple[str, str]:
    """교정 전후에서 실제로 바뀐 부분을 감싸는 단어 (일본어는 형태소, 그 밖은 공백 단어).

    읽기 비교는 이 부분으로만 한다. LLM 이 바뀐 부분 앞뒤의 문장을 길게 붙여 보내면 전체 길이로 나눈 거리
    비율이 작아져 소리가 다른 교정(`昨日はじめと…` → `昨日はるかと…`)이 통과했다.

    Args:
        frm: 교정 전 구간.
        to: 교정 후 텍스트.
        language: 언어 코드.

    Returns:
        (교정 전 단어, 교정 후 단어).
    """
    prefix = 0
    while prefix < min(len(frm), len(to)) and frm[prefix] == to[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(frm), len(to)) - prefix
        and frm[len(frm) - 1 - suffix] == to[len(to) - 1 - suffix]
    ):
        suffix += 1
    # 교정 전후 양쪽의 단어 경계를 보고 넓은 쪽을 쓴다. 바뀌기 전 표기가 잘게 나뉘면(`図書間` → `図書`|`間`)
    # 한 글자 읽기만 비교하게 된다 (바뀐 뒤 표기는 한 단어인데 바뀌기 전 표기가 사전에 없어 한 글자씩 나뉘는 경우)
    left = min(
        max(b for b in _word_bounds(frm, language) if b <= prefix),
        max(b for b in _word_bounds(to, language) if b <= prefix),
    )
    tail = min(
        len(frm) - min(b for b in _word_bounds(frm, language) if b >= len(frm) - suffix),
        len(to) - min(b for b in _word_bounds(to, language) if b >= len(to) - suffix),
    )
    return frm[left : len(frm) - tail], to[left : len(to) - tail]


def _word_bounds(text: str, language: str) -> set[int]:
    """단어 경계 위치 (양 끝 포함). 일본어는 형태소 경계, 그 밖은 공백."""
    bounds = {0, len(text)}
    if language == "ja":
        from src.postprocess.word_units import japanese_morpheme_boundaries

        bounds |= japanese_morpheme_boundaries(text)
    else:
        for i, ch in enumerate(text):
            if ch.isspace():
                bounds |= {i, i + 1}
    return bounds


def apply_correction(seg: TranscriptionSegment, frm: str, to: str) -> bool:
    """세그먼트 단어에 교정을 넣는다. 여러 단어에 걸치면 합친다 (시작·끝은 첫·마지막 단어).

    Args:
        seg: 교정할 세그먼트.
        frm: 원래 구간 (단어를 이은 문장에 한 번만 나와야 한다).
        to: 고친 텍스트.

    Returns:
        넣었으면 True. 구간이 문장에 없거나 여러 번 나오면 False.
    """
    spans, text = _word_spans(seg.words)
    if text.count(frm) != 1:
        return False
    start = text.index(frm)
    end = start + len(frm)
    touched = [i for i, (a, b) in enumerate(spans) if a < end and b > start]
    if not touched:
        return False
    first, last = touched[0], touched[-1]
    span_start, span_end = spans[first][0], spans[last][1]
    new_text = text[span_start:start] + to + text[end:span_end]
    merged = WordSegment(
        new_text,
        seg.words[first].start,
        seg.words[last].end,
        min(w.confidence for w in seg.words[first : last + 1]),
    )
    seg.words = [*seg.words[:first], merged, *seg.words[last + 1 :]]
    seg.text = join_words(w.word for w in seg.words)
    return True


def _word_spans(words: Sequence[WordSegment]) -> tuple[list[tuple[int, int]], str]:
    """단어를 ``join_words`` 로 이은 문장과, 단어마다 그 문장 안의 (시작, 끝) 위치."""
    spans: list[tuple[int, int]] = []
    text = ""
    for word in words:
        joined = join_words([text, word.word]) if text else word.word.strip()
        start = len(joined) - len(word.word.strip())
        spans.append((start, len(joined)))
        text = joined
    return spans, text


def _correction_items(items: Any) -> list[tuple[int, str, str, str]]:
    """``[{"id": int, "from": str, "to": str, "reason": str}]`` 에서 모양이 맞는 항목."""
    if not isinstance(items, list):
        return []
    return [
        (item["id"], item["from"], item["to"], str(item.get("reason", "")))
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("id"), int)
        and isinstance(item.get("from"), str)
        and isinstance(item.get("to"), str)
    ]
