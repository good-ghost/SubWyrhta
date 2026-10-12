"""단어 타임스탬프 기반 경계 스냅 공통 유틸.

PySBD 등이 결정한 분할점을 가장 가까운 단어의 end 타임스탬프로 스냅하여
타임코드 겹침을 방지한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.models.segment import TranscriptionSegment

if TYPE_CHECKING:
    from src.models.word import WordSegment

log = structlog.get_logger(__name__)


def snap_segments_to_word_boundaries(
    segments: list[TranscriptionSegment],
    min_gap_s: float = 0.04,
) -> list[TranscriptionSegment]:
    """세그먼트 경계를 단어 타임스탬프에 스냅한다.

    각 세그먼트의 end를 마지막 단어의 end로,
    다음 세그먼트의 start를 prev.end + min_gap_s로 보정한다.

    Args:
        segments: 분할된 세그먼트 목록 (words 포함 필수).
        min_gap_s: 세그먼트 간 최소 갭(초).

    Returns:
        경계 스냅 적용된 세그먼트 목록.
    """
    if not segments:
        return []

    result: list[TranscriptionSegment] = []

    for seg in segments:
        snapped = _snap_single(seg)
        result.append(snapped)

    # 인접 세그먼트 간 start 보정 + 겹침 제거
    _fix_overlaps(result, min_gap_s)

    return result


def _snap_single(seg: TranscriptionSegment) -> TranscriptionSegment:
    """단일 세그먼트의 start/end를 단어 경계로 스냅한다."""
    if not seg.words:
        return seg

    # start: 첫 번째 단어의 start
    first_word = seg.words[0]
    # end: 마지막 단어의 end
    last_word = seg.words[-1]

    return TranscriptionSegment(
        index=seg.index,
        text=seg.text,
        start=first_word.start,
        end=last_word.end,
        confidence=seg.confidence,
        speaker=seg.speaker,
        words=list(seg.words),
    )


def _fix_overlaps(
    segments: list[TranscriptionSegment],
    min_gap_s: float,
) -> None:
    """인접 세그먼트 간 타임코드 겹침을 제거한다 (in-place)."""
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        required_start = prev.end + min_gap_s

        if curr.start < required_start:
            curr.start = required_start


def verify_no_overlap(
    segments: list[TranscriptionSegment],
    min_gap_s: float = 0.04,
) -> list[tuple[int, int, float]]:
    """모든 세그먼트 쌍에 대해 타임코드 겹침을 검증한다.

    Args:
        segments: 세그먼트 목록.
        min_gap_s: 최소 갭(초).

    Returns:
        겹침이 발견된 (prev_index, curr_index, overlap_amount) 튜플 리스트.
        빈 리스트이면 겹침 없음.
    """
    violations: list[tuple[int, int, float]] = []

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]
        gap = curr.start - prev.end

        if gap < min_gap_s - 0.001:  # 부동소수점 허용 오차
            violations.append((prev.index, curr.index, gap))

    return violations


def find_nearest_word_boundary(
    words: list[WordSegment],
    char_offset: int,
    text: str,
) -> int:
    """문자 오프셋에 해당하는 가장 가까운 단어 인덱스를 찾는다.

    PySBD가 결정한 문자 오프셋을 단어 인덱스로 변환한다.

    Args:
        words: 단어 목록.
        char_offset: 텍스트 내 문자 오프셋.
        text: 전체 텍스트.

    Returns:
        분할 경계가 되는 단어 인덱스 (이 단어까지가 현재 세그먼트에 포함).
    """
    if not words:
        return 0

    # 각 단어의 문자 위치를 누적 계산
    cumulative = 0
    for i, word in enumerate(words):
        word_text = word.word.strip()
        # 공백 포함하여 누적
        cumulative += len(word_text)
        if i < len(words) - 1:
            cumulative += 1  # 단어 간 공백

        if cumulative >= char_offset:
            return i

    return len(words) - 1
