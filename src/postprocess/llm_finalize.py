"""LLM 마지막 검토: 자막 경계 옮기기, 줄바꿈, 검수 리포트 (선택 단계, 기본 꺼짐).

줄바꿈까지 끝난 자막을 LLM 에 보여 주고 세 가지를 받는다 (``TranscriptionPipeline._segment`` 마지막).

1. 경계 옮기기: 문장이 이웃 자막 사이에서 어색하게 나뉜 곳(`駅の近くにある` / `店はもう閉まっていた。`)의
   경계 단어를 옮긴다. 같은 화자의 이웃 두 자막만, 단어 경계에서만 옮긴다. 시각은 옮긴 단어의 타임스탬프로
   다시 정한다
2. 줄바꿈: 두 줄 자막의 줄바꿈 위치를 의미 단위로 옮긴다 (9.14 의 줄바꿈 검토를 흡수)
3. 검수 리포트: 위로 고칠 수 없는 것(오인식 의심 `機会`/`機械`, 의미가 틀린 부호)을 표시만 한다.
   자막은 바꾸지 않고 출력 폴더의 ``<이름>.review.json`` 에 남긴다

글자 교정은 맡기지 않는다 (단어 시각과 맞지 않고 지어낼 위험이 있다). LLM 응답은 코드가 검증해 맞을 때만
받는다: 경계 옮기기는 두 자막을 이은 글자가 같고, 새 경계가 단어 경계이고, 각 자막이 두 줄·글자 수·최소/최대
표시 시간 안이어야 한다. 줄바꿈은 줄바꿈 위치 말고 바뀐 것이 없어야 한다.

2026-10-10 사용자 요청 ("제일 마지막에 LLM 이 전체 자막을 검수하는 프로세스", 검토 뒤 "진행").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from src.core.text import join_words
from src.postprocess.llm_review import is_acceptable_revision, tidy_revision
from src.postprocess.llm_text import (
    DEFAULT_NOTE_LANGUAGE,
    LLMBreaker,
    LLMTextStats,
    comparable,
    id_text_pairs,
    localize_notes,
    note_language_name,
    note_language_rule,
    request_windows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.config import LLMConfig, SegmentationConfig
    from src.models.segment import TranscriptionSegment
    from src.models.word import WordSegment
    from src.segmentation.llm_client import LLMProvider

log = structlog.get_logger(__name__)

# 검수 리포트 항목 종류
ISSUE_TYPES = frozenset({"misrecognition", "punctuation", "split", "other"})
# 표시 시간 비교 여유
_EPSILON_S = 1e-6
# 리포트에 남기는 메모 길이 상한
_MAX_NOTE_LENGTH = 300

FINALIZE_PROMPT = """\
You check the final subtitles of one video. The subtitles are consecutive, in order, with start \
and end times in seconds. Reply with four lists.
1. "boundaries": where a sentence is split between two adjacent subtitles in an unnatural place \
(subtitle id ends in the middle of a sentence and subtitle id+1 continues it), move whole words \
across that boundary so that subtitle id ends at a natural sentence or clause end: \
{{"id": <id>, "text": "<new text of id>", "next": "<new text of id+1>"}}. "text" + "next" must \
contain exactly the characters of subtitle id and id+1 together, in the same order.
2. "splits": where one subtitle contains two sentences of different speakers (or two unrelated \
sentences) and is long enough to show them separately, split it into two at a word boundary: \
{{"id": <id>, "parts": ["<first part>", "<second part>"]}}. The parts together must contain \
exactly the characters of subtitle id.
3. "line_breaks": for a subtitle with two lines, move the line break ("\\n") if a line does not end \
at a natural phrase boundary: {{"id": <id>, "text": "<new text>"}}. Never break inside a word, a \
name, a number with its counter or a compound noun. At most {max_lines} lines, each at most \
{chars_per_line} characters. Do not change any character.
4. "issues": problems you cannot fix by 1-3, for a human to check: a word that is probably a \
speech recognition error (does not fit the context), wrong punctuation, or other: \
{{"id": <id>, "type": "misrecognition" | "punctuation" | "other", "note": "<short note, with the \
likely correct word if any>"}}.
Never change, add or delete characters. Leave a list empty if there is nothing to report.
{note_language}Reply with JSON only: {{"boundaries": [], "splits": [], "line_breaks": [], "issues": []}}.
"""


@dataclass
class FinalizeResult:
    """LLM 마지막 검토 결과.

    Attributes:
        stats: 호출·캐시 집계 (``changed`` 는 경계 옮기기·나누기·줄바꿈을 합친 수).
        boundaries: 받아들인 경계 옮기기 수.
        splits: 받아들인 나누기 수 (자막이 그만큼 는다).
        line_breaks: 받아들인 줄바꿈 수.
        issues: 검수 리포트 항목 (고치기 전 자막의 시각·텍스트, 종류, 메모).
        records: 묶음 키 → LLM 응답 (단계 캐시에 남긴다).
    """

    stats: LLMTextStats = field(default_factory=LLMTextStats)
    boundaries: int = 0
    splits: int = 0
    line_breaks: int = 0
    issues: list[dict[str, Any]] = field(default_factory=list)
    records: dict[str, dict[str, Any]] = field(default_factory=dict)


def finalize_subtitles(
    segments: list[TranscriptionSegment],
    language: str,
    chars_per_line: int,
    config: SegmentationConfig,
    llm_config: LLMConfig,
    known: dict[str, dict[str, Any]] | None = None,
    provider: LLMProvider | None = None,
    breaker: LLMBreaker | None = None,
    note_language: str = DEFAULT_NOTE_LANGUAGE,
) -> FinalizeResult:
    """자막 경계·나누기·줄바꿈을 LLM 으로 고치고 검수 리포트 항목을 모은다 (in-place).

    LLM 의 번호는 보낸 순서의 자막 객체에 묶는다. 나누기로 자막이 늘어도 번호가 어긋나지 않는다.
    응답은 원래 텍스트를 보고 만든 것이라, 앞 제안으로 이미 바뀐 자막에 닿는 제안은 검증에서 떨어진다.

    Args:
        segments: 줄바꿈까지 끝난 자막 목록 (시간순). 나누면 이 목록에 끼워 넣는다.
        language: 언어 코드.
        chars_per_line: 줄당 최대 글자 수.
        config: 세그멘테이션 설정 (줄 수, 최소·최대 표시 시간, 간격, 화자 구분자).
        llm_config: LLM 프로바이더 설정.
        known: 이전에 받은 응답 (묶음 키 → 응답). 키가 같은 묶음은 보내지 않는다.
        provider: LLM 프로바이더 (테스트용). None 이면 설정으로 만든다.
        breaker: 작업 단위 LLM 차단기 (열려 있으면 호출하지 않는다).
        note_language: 검수 메모를 쓸 언어 코드 (리포트를 읽는 화면 언어. 자막 언어가 아니다).

    Returns:
        결과 (집계, 리포트 항목, 단계 캐시에 남길 응답).
    """
    result = FinalizeResult()
    items = [
        {"text": seg.text, "start": round(seg.start, 2), "end": round(seg.end, 2)}
        for seg in segments
    ]
    prompt = FINALIZE_PROMPT.format(
        chars_per_line=chars_per_line,
        max_lines=config.max_lines,
        note_language=note_language_rule(note_language, "note", "subtitles"),
    )
    answers, result.records = request_windows(
        items,
        prompt,
        "subtitles",
        language,
        llm_config,
        result.stats,
        provider,
        known,
        breaker,
        # 시스템 프롬프트의 규칙만으로는 모델이 자막 언어로 메모를 썼다: 자료 옆에서 한 번 더 알린다
        extra={"note_language": note_language_name(note_language)},
    )
    sent = list(segments)
    sent_texts = [seg.text for seg in sent]

    # 검수 리포트 (고치기 전 자막 기준)
    for start, count, data in answers:
        result.issues.extend(_issues(data.get("issues"), sent, start, count))
    # 메모가 자막 언어로 온 항목은 메모 언어로 옮긴다 (응답은 같은 단계 캐시에 남긴다)
    notes, note_records = localize_notes(
        [issue["note"] for issue in result.issues],
        note_language,
        llm_config,
        result.stats,
        provider,
        known,
        breaker,
    )
    for issue, note in zip(result.issues, notes, strict=True):
        issue["note"] = note[:_MAX_NOTE_LENGTH]
    result.records.update(note_records)

    def in_window(index: int, start: int, count: int) -> bool:
        return start <= index < start + count and index < len(sent)

    # 1. 경계 옮기기
    for start, count, data in answers:
        for id_, first_text, next_text in _boundary_items(data.get("boundaries")):
            index = id_ - 1
            if not in_window(index, start, count) or index + 1 >= len(sent):
                continue
            position = _position(segments, sent[index])
            if position is None or position + 1 >= len(segments):
                continue
            if segments[position + 1] is not sent[index + 1]:
                result.stats.rejected += 1
                continue
            before = sent[index].text
            if _move_boundary(
                segments, position, first_text, next_text, language, chars_per_line, config
            ):
                result.boundaries += sent[index].text != before
            else:
                result.stats.rejected += 1
                log.debug("LLM 경계 옮기기 거부", id=id_, text=first_text, next=next_text)

    # 2. 나누기
    for start, count, data in answers:
        for id_, parts in _split_items(data.get("splits")):
            index = id_ - 1
            if not in_window(index, start, count):
                continue
            seg = sent[index]
            position = _position(segments, seg)
            if position is None or seg.text != sent_texts[index]:
                result.stats.rejected += 1
                continue
            second = _split(seg, parts, language, chars_per_line, config)
            if second is None:
                result.stats.rejected += 1
                log.debug("LLM 나누기 거부", id=id_, parts=parts)
                continue
            segments.insert(position + 1, second)
            result.splits += 1

    # 3. 줄바꿈 (경계를 옮기거나 나눈 자막은 텍스트가 달라 받지 않는다)
    for start, count, data in answers:
        for id_, proposed in id_text_pairs(data.get("line_breaks")) or []:
            index = id_ - 1
            if not in_window(index, start, count):
                continue
            seg = sent[index]
            if seg.text != sent_texts[index] or "\n" not in seg.text:
                continue
            text = tidy_revision(seg.text, proposed, language)
            if text == seg.text:
                continue
            if is_acceptable_revision(seg.text, text, language, chars_per_line, config.max_lines):
                seg.text = text
                result.line_breaks += 1
            else:
                result.stats.rejected += 1

    result.stats.changed = result.boundaries + result.splits + result.line_breaks
    log.info(
        "LLM 마지막 검토 완료",
        subtitles=len(sent),
        boundaries=result.boundaries,
        splits=result.splits,
        line_breaks=result.line_breaks,
        issues=len(result.issues),
        rejected=result.stats.rejected,
        cached=result.stats.cached,
        windows=result.stats.windows,
        failed=result.stats.failed,
        input_tokens=result.stats.input_tokens,
        output_tokens=result.stats.output_tokens,
    )
    return result


def _position(segments: Sequence[TranscriptionSegment], seg: TranscriptionSegment) -> int | None:
    """목록에서 그 자막 객체의 위치 (같은 객체만 찾는다)."""
    return next((i for i, s in enumerate(segments) if s is seg), None)


def _split_items(items: Any) -> list[tuple[int, list[str]]]:
    """``[{"id": int, "parts": [str, str]}]`` 에서 모양이 맞는 항목 (두 조각만)."""
    if not isinstance(items, list):
        return []
    return [
        (item["id"], list(item["parts"]))
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("id"), int)
        and isinstance(item.get("parts"), list)
        and len(item["parts"]) == 2
        and all(isinstance(part, str) for part in item["parts"])
    ]


def _split(
    seg: TranscriptionSegment,
    parts: list[str],
    language: str,
    chars_per_line: int,
    config: SegmentationConfig,
) -> TranscriptionSegment | None:
    """자막을 단어 경계에서 둘로 나눈다 (앞 조각은 그 자막을 고친다). 검증에서 떨어지면 None.

    글자가 같고, 단어 경계이고, 두 조각 모두 최소·최대 표시 시간과 줄 제한 안이어야 한다. 화자가 여럿인
    자막(겹침 병합)은 나누지 않는다 (조각마다 화자를 정할 수 없다).
    """
    from src.models.segment import TranscriptionSegment

    if isinstance(seg.speaker, list) or len(seg.words) < 2:
        return None
    joined = _compact(seg.text, language)
    if _compact(join_words(parts), language) != joined:
        return None
    if _compact(join_words(w.word for w in seg.words), language) != joined:
        return None
    cut = _word_index_at(seg.words, len(_compact(parts[0], language)), language)
    if cut is None or cut in (0, len(seg.words)):
        return None
    left, right = seg.words[:cut], seg.words[cut:]
    first_end = min(left[-1].end, right[0].start - config.min_gap_s)
    durations = (first_end - seg.start, seg.end - right[0].start)
    if any(d < config.min_duration_s - _EPSILON_S or d > config.max_duration_s for d in durations):
        return None
    first_text = _wrapped(left, language, chars_per_line, config)
    second_text = _wrapped(right, language, chars_per_line, config)
    if first_text is None or second_text is None:
        return None
    second = TranscriptionSegment(
        seg.index,
        second_text,
        right[0].start,
        seg.end,
        seg.confidence,
        speaker=seg.speaker,
        words=list(right),
    )
    seg.words, seg.text, seg.end = list(left), first_text, first_end
    return second


def _boundary_items(items: Any) -> list[tuple[int, str, str]]:
    """``[{"id": int, "text": str, "next": str}]`` 에서 모양이 맞는 항목."""
    if not isinstance(items, list):
        return []
    return [
        (item["id"], item["text"], item["next"])
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("id"), int)
        and isinstance(item.get("text"), str)
        and isinstance(item.get("next"), str)
    ]


def _move_boundary(
    segments: Sequence[TranscriptionSegment],
    index: int,
    first_text: str,
    next_text: str,
    language: str,
    chars_per_line: int,
    config: SegmentationConfig,
) -> bool:
    """segments[index] 와 다음 자막 사이의 경계를 옮긴다. 검증에서 떨어지면 False (그대로 둔다)."""
    first, second = segments[index], segments[index + 1]
    if isinstance(first.speaker, list) or isinstance(second.speaker, list):
        return False
    if first.speaker != second.speaker or not first.words or not second.words:
        return False

    words = [*first.words, *second.words]
    joined = _compact(join_words([first.text, second.text]), language)
    head, tail = _compact(first_text, language), _compact(next_text, language)
    # 새 경계는 앞 텍스트로 정한다. LLM 이 세 자막에 걸친 재배치를 한 번에 제안하며 뒤 자막의 끝을 빼고
    # 보내는 경우가 많아(`少し遅れます。` / `了解です。では明日…`, 뒤의 `お願いします。` 를 뺌),
    # 뒤 텍스트는 나머지의 앞부분이면 된다. 빠진 끝은 원래 자리에 그대로 남는다
    if not head or not joined.startswith(head) or not joined[len(head) :].startswith(tail):
        return False
    if _compact(join_words(w.word for w in words), language) != joined:
        return False  # 텍스트와 단어가 맞지 않는다 (발화 구분·부호 옮기기 등으로 바뀐 자막)

    cut = _word_index_at(words, len(head), language)
    if cut == len(first.words):
        return True  # 지금 경계와 같다 (앞 제안을 받아들인 결과를 보고 만든 제안)
    if cut is None or cut in (0, len(words)):
        return False

    left, right = words[:cut], words[cut:]
    first_end = min(left[-1].end, right[0].start - config.min_gap_s)
    second_start = right[0].start
    durations = (first_end - first.start, second.end - second_start)
    if any(d < config.min_duration_s - _EPSILON_S or d > config.max_duration_s for d in durations):
        return False

    first_wrapped = _wrapped(left, language, chars_per_line, config)
    second_wrapped = _wrapped(right, language, chars_per_line, config)
    if first_wrapped is None or second_wrapped is None:
        return False

    first.words, first.text, first.end = left, first_wrapped, first_end
    second.words, second.text, second.start = right, second_wrapped, second_start
    return True


def _compact(text: str, language: str) -> str:
    """비교용 텍스트에서 공백까지 뺀 것 (단어 경계 위치를 글자 수로 센다)."""
    return comparable(text, language).replace(" ", "")


def _word_index_at(words: Sequence[WordSegment], chars: int, language: str) -> int | None:
    """앞에서 chars 글자가 끝나는 단어 경계 (그 뒤 단어의 인덱스). 단어 중간이면 None."""
    count = 0
    for index, word in enumerate(words):
        if count == chars:
            return index
        count += len(_compact(word.word, language))
    return len(words) if count == chars else None


def _wrapped(
    words: Sequence[WordSegment], language: str, chars_per_line: int, config: SegmentationConfig
) -> str | None:
    """단어로 텍스트를 만들고 규칙으로 줄을 나눈다. 줄 제한을 넘으면 None."""
    from src.models.segment import TranscriptionSegment
    from src.postprocess.subtitle_text import wrap_lines

    seg = TranscriptionSegment(
        0, join_words(w.word for w in words), 0.0, 0.0, 0.0, words=list(words)
    )
    wrap_lines([seg], language, chars_per_line, config.max_lines, config.overlap_text_separator)
    lines = seg.text.split("\n")
    if len(lines) > config.max_lines or any(len(line) > chars_per_line for line in lines):
        return None
    return seg.text


def _issues(
    items: Any, segments: Sequence[TranscriptionSegment], start: int, count: int
) -> list[dict[str, Any]]:
    """검수 리포트 항목. 번호가 묶음 밖이거나 모양이 다르면 뺀다. 자막 시각·텍스트를 함께 남긴다."""
    if not isinstance(items, list):
        return []
    issues: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            continue
        index = item["id"] - 1
        if not start <= index < start + count:
            continue
        kind = item.get("type") if item.get("type") in ISSUE_TYPES else "other"
        seg = segments[index]
        issues.append(
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
                "type": kind,
                "note": str(item.get("note", ""))[:_MAX_NOTE_LENGTH],
            }
        )
    return issues
