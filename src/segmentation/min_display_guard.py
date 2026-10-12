"""최소 표시 시간 보장 후처리.

min_duration_s 미만 세그먼트를 인접 세그먼트와 병합하여 자막이 너무 짧게 표시되는 것을 방지한다.
병합하면 max_duration_s 를 넘는 경우(멀리 떨어진 자막과 합쳐지는 경우)는 병합하지 않고 앞뒤 빈 시간으로
표시 시간을 늘린다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.core.text import join_words
from src.models.segment import TranscriptionSegment
from src.segmentation.speed_optimizer import lines_needed

if TYPE_CHECKING:
    from src.core.config import SegmentationConfig

log = structlog.get_logger(__name__)

# 표시 시간 비교 여유 (늘린 뒤 부동소수 오차로 다시 "짧음"이 되지 않게)
_DURATION_EPSILON_S = 1e-6
# 병합 대상과 이만큼 이상 떨어져 있으면 병합하지 않고 빈 시간으로 늘린다 (TO-DO 6). 쉼이 길면 다른 사람의
# 말이거나 다른 장면이다: `おはよう`(0.75초)가 2.3초 쉬고 나오는 대사와 합쳐져 6.15초 자막이 됐다
_MAX_MERGE_GAP_S = 1.0


def enforce_min_duration(
    segments: list[TranscriptionSegment],
    config: SegmentationConfig,
    max_chars: int | None = None,
    chars_per_line: int | None = None,
) -> list[TranscriptionSegment]:
    """min_duration_s 미만 세그먼트를 늘리거나 인접 세그먼트와 병합한다.

    병합 우선순위 (REQUIREMENT 6.7):
        1. 동일 화자 인접 세그먼트 우선 (gap 이 짧은 쪽)
        2. 다화자인 경우 다음 세그먼트와 구분자 삽입 후 병합
        3. 병합 후 재검증 (반복)

    병합 대상과 1초 이상 떨어져 있거나, 병합 결과가 ``max_duration_s`` 를 넘거나, 글자 수가
    ``max_chars`` 를 넘거나, 단어 경계로 ``max_lines`` 줄에 들어가지 않으면 병합하지 않고 앞뒤 빈
    시간으로 늘린다 (모자라면 가능한 만큼). 예: 0.96초 자막이 10초 무음 뒤 자막과 합쳐져 11.6초 자막이
    되던 문제 (REQUIREMENT 6.4 Hard Cap), 두 줄에 들어가지 않는 자막이 되던 문제.

    Args:
        segments: 입력 세그먼트 목록.
        config: 세그멘테이션 설정.
        max_chars: 자막 하나의 최대 글자 수 (줄당 글자 수 x 줄 수). None 이면 보지 않는다.
        chars_per_line: 줄당 최대 글자 수. 주면 병합한 단어가 ``max_lines`` 줄에 들어가는지 본다
            (advanced 의 글자 수 상한과 같은 기준이라, 나눈 자막을 다시 합치지 않는다).

    Returns:
        최소 표시 시간이 보장된 세그먼트 목록.
    """
    if not segments:
        return []

    result = list(segments)
    extended = 0
    # 늘리고도 짧은 채로 둔 세그먼트 (다시 고르지 않는다)
    settled: set[int] = set()

    # 늘리거나 병합할 때마다 짧은 세그먼트가 하나 준다. 병합마다 세그먼트가 하나 줄어 반복은 많아야 세그먼트 수만큼이다. 예전에는 50번에서 멈춰
    # 짧은 자막이 수백 개 남았다 (REQUIREMENT 6.7 "반복 검사")
    while len(result) > 1:
        short_indices = [
            i
            for i in _find_short_segments(result, config.min_duration_s)
            if id(result[i]) not in settled
        ]
        if not short_indices:
            break

        idx = short_indices[0]
        target = _find_merge_target(result, idx, config)
        if target is not None and (
            _gap_between(result, idx, target) >= _MAX_MERGE_GAP_S
            or _merged_span(result, idx, target) > config.max_duration_s
            or (max_chars is not None and _merged_chars(result, idx, target) > max_chars)
            or (
                chars_per_line is not None
                and lines_needed(_merged_words(result, idx, target), chars_per_line)
                > config.max_lines
            )
        ):
            _extend_into_gaps(result, idx, config)
            settled.add(id(result[idx]))
            extended += 1
            continue
        # 한 번에 하나씩 병합 (인덱스 변경 때문에)
        result = _merge_short_segment(result, idx, config)

    # 인덱스 재부여
    for i, seg in enumerate(result, start=1):
        seg.index = i

    merged_count = len(segments) - len(result)
    if merged_count > 0 or extended > 0:
        log.info(
            "최소 표시 시간 보장 완료",
            min_duration_s=config.min_duration_s,
            extended=extended,
            merged=merged_count,
            total_segments=len(result),
        )

    return result


def _find_short_segments(
    segments: list[TranscriptionSegment],
    min_duration_s: float,
) -> list[int]:
    """min_duration_s 미만 세그먼트의 인덱스 목록을 반환한다."""
    return [
        i for i, seg in enumerate(segments) if seg.duration < min_duration_s - _DURATION_EPSILON_S
    ]


def _gap_between(segments: list[TranscriptionSegment], idx: int, target: int) -> float:
    """두 세그먼트 사이의 빈 시간."""
    first, last = sorted((idx, target))
    return segments[last].start - segments[first].end


def _merged_span(segments: list[TranscriptionSegment], idx: int, target: int) -> float:
    """두 세그먼트를 병합했을 때의 표시 시간."""
    first, last = sorted((idx, target))
    return segments[last].end - segments[first].start


def _merged_chars(segments: list[TranscriptionSegment], idx: int, target: int) -> int:
    """두 세그먼트를 병합했을 때의 글자 수 (공백 제외)."""
    return sum(len("".join(segments[i].text.split())) for i in (idx, target))


def _merged_words(segments: list[TranscriptionSegment], idx: int, target: int) -> list[str]:
    """두 세그먼트를 병합했을 때의 단어 (단어가 없는 세그먼트는 텍스트를 한 단어로 본다)."""
    first, last = sorted((idx, target))
    return [
        part
        for seg in segments[first : last + 1]
        for part in ([w.word for w in seg.words] or [seg.text])
    ]


def _extend_into_gaps(
    segments: list[TranscriptionSegment], idx: int, config: SegmentationConfig
) -> None:
    """앞뒤 빈 시간(이웃과 min_gap_s 는 둔다)으로 min_duration_s 까지 늘린다. 뒤 먼저, 모자라면 앞.
    그래도 모자라면 이웃에게서 시간을 받는다."""
    seg = segments[idx]
    need = config.min_duration_s - seg.duration
    after_limit = (
        segments[idx + 1].start - config.min_gap_s if idx + 1 < len(segments) else float("inf")
    )
    before_limit = segments[idx - 1].end + config.min_gap_s if idx > 0 else 0.0
    grow_after = min(need, max(0.0, after_limit - seg.end))
    grow_before = min(need - grow_after, max(0.0, seg.start - before_limit))
    seg.end += grow_after
    seg.start -= grow_before
    _borrow_from_neighbors(segments, idx, config)


def _borrow_from_neighbors(
    segments: list[TranscriptionSegment], idx: int, config: SegmentationConfig
) -> None:
    """빈 시간으로도 모자라면 이웃이 min_duration_s 아래로 줄지 않는 만큼 경계를 옮겨 시간을 받는다.

    WhisperX 가 빠른 대사를 짧은 시간에 몰아 정렬하면(26자를 2.2초에) 글자 수 상한으로 나눈 끝 자막이
    0.3초가 되고, 이웃과 붙어 있어 빈 시간도 없다. 표시 시각이 발화와 조금 어긋나는 대신 읽을 시간을 준다.
    앞 이웃 먼저, 모자라면 뒤 이웃. 이웃과의 간격은 그대로 둔다.
    """
    seg = segments[idx]
    need = config.min_duration_s - seg.duration
    if need > _DURATION_EPSILON_S and idx > 0:
        prev = segments[idx - 1]
        take = min(need, max(0.0, prev.duration - config.min_duration_s))
        prev.end -= take
        seg.start -= take
        need -= take
    if need > _DURATION_EPSILON_S and idx + 1 < len(segments):
        nxt = segments[idx + 1]
        take = min(need, max(0.0, nxt.duration - config.min_duration_s))
        nxt.start += take
        seg.end += take


def _merge_short_segment(
    segments: list[TranscriptionSegment],
    short_idx: int,
    config: SegmentationConfig,
) -> list[TranscriptionSegment]:
    """짧은 세그먼트를 최적의 인접 세그먼트와 병합한다."""
    short = segments[short_idx]
    result = list(segments)

    # 병합 대상 결정
    merge_target = _find_merge_target(result, short_idx, config)

    if merge_target is None:
        # 병합 대상이 없으면 (단일 세그먼트) 그대로 유지
        return result

    target_idx = merge_target
    target = result[target_idx]

    # 병합 실행
    if target_idx < short_idx:
        # 이전 세그먼트에 병합
        merged = _merge_two(target, short, config.overlap_text_separator)
        result[target_idx] = merged
        result.pop(short_idx)
    else:
        # 다음 세그먼트에 병합
        merged = _merge_two(short, target, config.overlap_text_separator)
        result[short_idx] = merged
        result.pop(target_idx)

    return result


def _find_merge_target(
    segments: list[TranscriptionSegment],
    short_idx: int,
    config: SegmentationConfig,
) -> int | None:
    """병합 대상 인덱스를 반환한다.

    우선순위 (REQUIREMENT 6.7):
        1. 동일 화자 인접 세그먼트 (gap 이 짧은 쪽 우선, 케이스 A)
        2. 다화자 인접 세그먼트: 다음 세그먼트 (마지막이면 이전, 케이스 B)
    """
    short = segments[short_idx]
    has_prev = short_idx > 0
    has_next = short_idx < len(segments) - 1

    if not has_prev and not has_next:
        return None

    prev = segments[short_idx - 1] if has_prev else None
    next_seg = segments[short_idx + 1] if has_next else None

    # 1. 동일 화자 인접 탐색
    prev_same = has_prev and prev is not None and _same_speaker(short, prev)
    next_same = has_next and next_seg is not None and _same_speaker(short, next_seg)

    if prev_same and next_same:
        # 둘 다 동일 화자 → gap 이 짧은 쪽과 병합
        assert prev is not None and next_seg is not None
        gap_before = short.start - prev.end
        gap_after = next_seg.start - short.end
        return short_idx - 1 if gap_before <= gap_after else short_idx + 1

    if prev_same:
        return short_idx - 1
    if next_same:
        return short_idx + 1

    # 2. 다화자 → 다음 세그먼트와 병합 (마지막 세그먼트면 이전)
    return short_idx + 1 if has_next else short_idx - 1


def _same_speaker(
    a: TranscriptionSegment,
    b: TranscriptionSegment,
) -> bool:
    """두 세그먼트의 화자가 동일한지 판단한다."""
    # 둘 다 None이면 동일 (화자 미지정)
    if a.speaker is None and b.speaker is None:
        return True
    if a.speaker is None or b.speaker is None:
        return False
    # 문자열 비교
    return a.speaker == b.speaker


def _merge_two(
    first: TranscriptionSegment,
    second: TranscriptionSegment,
    separator: str,
) -> TranscriptionSegment:
    """두 세그먼트를 하나로 병합한다.

    동일 화자면 텍스트를 그대로 잇고 (일본어·중국어는 공백 없이, ``join_words``),
    다화자면 구분자로 연결한다.
    """
    same = _same_speaker(first, second)

    if same:
        merged_text = join_words([first.text, second.text])
        merged_speaker = first.speaker
    else:
        merged_text = first.text + separator + second.text
        # 다화자 → 배열로 표현
        merged_speaker = _merge_speakers(first.speaker, second.speaker)

    # 단어 목록 병합
    merged_words = list(first.words) + list(second.words)

    # 신뢰도 가중 평균
    total_words = len(first.words) + len(second.words)
    if total_words > 0:
        avg_conf = (
            first.confidence * len(first.words) + second.confidence * len(second.words)
        ) / total_words
    else:
        avg_conf = (first.confidence + second.confidence) / 2

    return TranscriptionSegment(
        index=first.index,
        text=merged_text,
        start=first.start,
        end=second.end,
        confidence=avg_conf,
        speaker=merged_speaker,
        words=merged_words,
    )


def _merge_speakers(
    a: str | list[str] | None,
    b: str | list[str] | None,
) -> list[str]:
    """화자 ID를 병합한다."""
    speakers: list[str] = []

    if isinstance(a, list):
        speakers.extend(a)
    elif isinstance(a, str):
        speakers.append(a)

    if isinstance(b, list):
        for s in b:
            if s not in speakers:
                speakers.append(s)
    elif isinstance(b, str) and b not in speakers:
        speakers.append(b)

    return speakers
