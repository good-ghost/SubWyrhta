"""텍스트 정규화.

문장 시작 대문자화, 이중 공백 정규화, 빈 항목 제거를 수행한다.
"""

from __future__ import annotations

import re

import structlog

from src.models.segment import TranscriptionSegment

log = structlog.get_logger(__name__)

_MULTI_SPACE = re.compile(r" {2,}")
_SENTENCE_START = re.compile(r"(^|[.!?。！？]\s+)(\w)")


def normalize_segments(
    segments: list[TranscriptionSegment],
) -> list[TranscriptionSegment]:
    """세그먼트 목록의 텍스트를 정규화한다.

    처리 순서:
        1. 이중 공백 정규화
        2. 양쪽 공백 제거
        3. 문장 시작 대문자화 (라틴 문자)
        4. 빈 항목 제거

    Args:
        segments: 입력 세그먼트 목록.

    Returns:
        정규화된 세그먼트 목록.
    """
    result: list[TranscriptionSegment] = []
    removed = 0

    for seg in segments:
        text = seg.text

        # 1. 이중 공백 → 단일 공백
        text = _MULTI_SPACE.sub(" ", text)

        # 2. 양쪽 공백 제거
        text = text.strip()

        # 3. 문장 시작 대문자화
        text = _capitalize_sentence_start(text)

        # 4. 빈 항목 제거
        if not text:
            removed += 1
            continue

        result.append(
            TranscriptionSegment(
                index=seg.index,
                text=text,
                start=seg.start,
                end=seg.end,
                confidence=seg.confidence,
                speaker=seg.speaker,
                words=list(seg.words),
            )
        )

    # 인덱스 재부여
    for i, seg in enumerate(result, start=1):
        seg.index = i

    if removed > 0:
        log.info("텍스트 정규화 완료", removed_empty=removed, remaining=len(result))

    return result


def _capitalize_sentence_start(text: str) -> str:
    """문장 시작을 대문자화한다 (라틴 문자만).

    - 텍스트 맨 앞 글자
    - `.!?` 뒤 공백 다음 글자
    """
    if not text:
        return text

    # 첫 글자 대문자화
    text = text[0].upper() + text[1:]

    # 문장 종결 후 대문자화
    def _upper(m: re.Match[str]) -> str:
        return m.group(1) + m.group(2).upper()

    text = _SENTENCE_START.sub(_upper, text)

    return text
