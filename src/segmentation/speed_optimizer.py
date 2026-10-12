"""CPS 기반 읽기 속도 최적화.

CPS(Characters Per Second) 초과 자막을 자연스러운 분할점에서 자동 분할한다.
언어별 CPS 기본값을 제공한다.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from src.core.text import join_words
from src.models.segment import TranscriptionSegment
from src.segmentation.word_boundary_snapper import snap_segments_to_word_boundaries

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.config import SegmentationConfig
    from src.models.word import WordSegment

log = structlog.get_logger(__name__)

# 언어별 CPS 기본값
CPS_DEFAULTS: dict[str, float] = {
    "en": 17.0,
    "ko": 9.0,
    "ja": 7.0,
    "zh": 5.0,
}
CPS_FALLBACK = 17.0

# 언어별 줄당 최대 글자 수 기본값. 일본어·중국어 자막은 보통 한 줄 13~16자 (전각)
CHARS_PER_LINE_DEFAULTS: dict[str, int] = {
    "ja": 16,
    "zh": 16,
}
CHARS_PER_LINE_FALLBACK = 42

# 분할점 우선순위 패턴 (높은 순)
_SPLIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[.!?。！？]+$"), "sentence"),  # 절 경계
    (re.compile(r"[;:；：]+$"), "clause"),  # 세미콜론/콜론
    (re.compile(r"[,、，]+$"), "comma"),  # 쉼표
]


def get_cps_for_language(language: str, config: SegmentationConfig) -> float:
    """언어에 적합한 CPS 값을 반환한다.

    config.max_cps가 명시적으로 설정되어 있으면 그 값을 사용하고,
    None이면 언어별 기본값을 적용한다.

    Args:
        language: 언어 코드.
        config: 세그멘테이션 설정.

    Returns:
        적용할 max_cps 값.
    """
    if config.max_cps is not None:
        return config.max_cps
    return CPS_DEFAULTS.get(language, CPS_FALLBACK)


def get_chars_per_line(language: str, config: SegmentationConfig) -> int:
    """언어에 적합한 줄당 최대 글자 수를 반환한다.

    config.max_chars_per_line 이 있으면 그 값, None 이면 언어별 기본값.

    Args:
        language: 언어 코드.
        config: 세그멘테이션 설정.

    Returns:
        줄당 최대 글자 수.
    """
    if config.max_chars_per_line is not None:
        return config.max_chars_per_line
    return CHARS_PER_LINE_DEFAULTS.get(language, CHARS_PER_LINE_FALLBACK)


def extend_for_reading(
    segments: list[TranscriptionSegment], language: str, config: SegmentationConfig
) -> int:
    """CPS 가 상한을 넘는 자막의 끝을 뒤쪽 빈 시간으로 늘린다 (in-place, TO-DO 10).

    자막이 실제 말한 시간에 맞춰지면(LLM 부호 넣기로 문장마다 나뉘면) 빠른 대사는 읽을 시간이 모자란다
    (`そんなつもりではなかったと。` 1.1초, CPS 12). 다음 자막 시작 - ``min_gap_s`` 와 ``max_duration_s`` 안에서
    CPS 상한이 되는 길이까지 늘린다. 시작은 그대로 둔다 (말보다 먼저 나오지 않게).

    Args:
        segments: 시간순 자막 목록.
        language: 언어 코드 (CPS 상한).
        config: 세그멘테이션 설정.

    Returns:
        늘린 자막 수.
    """
    max_cps = get_cps_for_language(language, config)
    extended = 0
    for i, seg in enumerate(segments):
        chars = len("".join(seg.text.split()))
        needed = chars / max_cps
        if seg.duration >= needed:
            continue
        limit = segments[i + 1].start - config.min_gap_s if i + 1 < len(segments) else float("inf")
        new_end = min(seg.start + needed, seg.start + config.max_duration_s, limit)
        if new_end > seg.end:
            seg.end = new_end
            extended += 1
    if extended:
        log.info("읽기 시간 확보 (자막 끝 늘림)", extended=extended, max_cps=max_cps)
    return extended


def lines_needed(parts: Sequence[str], chars_per_line: int) -> int:
    """단어를 단어 경계에서만 줄을 바꿔 채울 때 필요한 줄 수.

    전체 글자 수가 두 줄 안이어도 단어 길이 때문에 두 줄로 나뉘지 않는 자막이 있다
    (`早く出てきなさいよ` / `チョコレートパフェとバニラアイスクリーム` 9자 / 20자). 한 줄보다 긴 단어는
    줄을 넘겨 이어 쓴다고 본다 (줄바꿈이 형태소 경계에서 나눈다).

    Args:
        parts: 단어 목록 (일본어는 문절).
        chars_per_line: 줄당 최대 글자 수.

    Returns:
        필요한 줄 수. 단어가 없으면 0.
    """
    lines, line = 0, ""
    for part in parts:
        candidate = join_words([line, part])
        if len(candidate) <= chars_per_line:
            line = candidate
            continue
        if line:
            lines += 1
        line = part.strip()
        while len(line) > chars_per_line:
            lines += 1
            line = line[chars_per_line:]
    return lines + (1 if line else 0)


def optimize_cps(
    segments: list[TranscriptionSegment],
    language: str,
    config: SegmentationConfig,
) -> list[TranscriptionSegment]:
    """CPS 초과 세그먼트를 분할하여 읽기 속도를 최적화한다.

    Args:
        segments: 입력 세그먼트 목록.
        language: 언어 코드.
        config: 세그멘테이션 설정.

    Returns:
        CPS 최적화된 세그먼트 목록.
    """
    max_cps = get_cps_for_language(language, config)
    result: list[TranscriptionSegment] = []
    split_count = 0

    for seg in segments:
        if seg.cps <= max_cps or not seg.words:
            result.append(seg)
            continue

        # CPS 초과 → 분할 시도
        split_segs = _split_for_cps(seg, max_cps, config.min_duration_s)
        if len(split_segs) > 1:
            split_count += 1
        result.extend(split_segs)

    # 인덱스 재부여
    for i, seg in enumerate(result, start=1):
        seg.index = i

    # 경계 스냅
    result = snap_segments_to_word_boundaries(result, config.min_gap_s)

    if split_count > 0:
        log.info(
            "CPS 최적화 완료",
            max_cps=max_cps,
            language=language,
            split_segments=split_count,
            total_segments=len(result),
        )

    return result


def _split_for_cps(
    seg: TranscriptionSegment,
    max_cps: float,
    min_duration_s: float,
) -> list[TranscriptionSegment]:
    """단일 세그먼트를 CPS 제한 내로 분할한다.

    분할점 우선순위: 절 경계 > 쉼표. 문장 부호가 없는 자리는 나누지 않는다 (2026-10-10 변경, 아래).
    양쪽 모두 ``min_duration_s`` 이상이 되는 분할점만 쓴다 (REQUIREMENT 6.5 "분할 후에도 최소 표시 시간 확보"). 나눠도 양쪽 CPS 는 낮아지지 않아
    (분할 전 CPS 가 두 조각 CPS 사이에 있다) 재귀는 더 나눌 분할점이 없을 때 멈춘다. 예전에는 분할점이
    없으면 가운데를 강제로 나눠 한 글자까지 쪼갰다 (일본어 글자 단위 단어, 0.1초짜리 자막).
    """
    words = seg.words
    if len(words) <= 1:
        return [seg]

    # 최적 분할점 탐색
    best_idx = _find_best_split_point(words, max_cps, min_duration_s)

    if best_idx is None or best_idx == 0 or best_idx >= len(words) - 1:
        return [seg]

    # 분할
    left_words = words[: best_idx + 1]
    right_words = words[best_idx + 1 :]

    left_seg = _build_sub_segment(left_words, seg.index, seg.speaker)
    right_seg = _build_sub_segment(right_words, seg.index + 1, seg.speaker)

    # 분할 후에도 CPS 초과하면 재귀 분할
    result: list[TranscriptionSegment] = []
    if left_seg.cps > max_cps and len(left_words) > 1:
        result.extend(_split_for_cps(left_seg, max_cps, min_duration_s))
    else:
        result.append(left_seg)

    if right_seg.cps > max_cps and len(right_words) > 1:
        result.extend(_split_for_cps(right_seg, max_cps, min_duration_s))
    else:
        result.append(right_seg)

    return result


def _find_best_split_point(
    words: list[WordSegment],
    max_cps: float,
    min_duration_s: float,
) -> int | None:
    """최적의 분할 위치(단어 인덱스)를 찾는다.

    우선순위: 절 경계 > 쉼표 (중앙 근처 우선).
    분할 후 양쪽 모두 min_duration_s를 만족해야 한다. 그런 분할점이 없으면 None (나누지 않는다).

    2026-10-10 변경 (사용자 결정, REQUIREMENT 6.5): 문장 부호가 없는 단어 경계에서는 나누지 않는다.
    LLM 부호 넣기로 CPS 가 상한을 살짝 넘은 자막(`駅の近くにあるいつもの店に行った。田中さん、`
    CPS 7.5)을 문장 중간에서 잘랐다. 나누지 않은 자막은 읽기 시간 확보(``extend_for_reading``)가 끝을 늘린다.
    """
    n = len(words)
    if n <= 1:
        return None

    mid = n // 2

    # 우선순위별 후보 수집
    candidates: list[tuple[int, int]] = []  # (priority, word_index)

    for i in range(1, n - 1):  # 첫/마지막 단어 제외
        word = words[i]

        # min_duration 체크
        left_dur = word.end - words[0].start
        right_dur = words[-1].end - words[i + 1].start
        if left_dur < min_duration_s or right_dur < min_duration_s:
            continue

        # 우선순위 결정 (문장 부호가 없는 자리는 후보가 아니다)
        priority = _get_split_priority(word)
        if priority < len(_SPLIT_PATTERNS):
            candidates.append((priority, i))

    if not candidates:
        return None

    # 같은 우선순위 내에서 중앙에 가까운 것 선택
    candidates.sort(key=lambda x: (x[0], abs(x[1] - mid)))
    return candidates[0][1]


def _get_split_priority(word: WordSegment) -> int:
    """단어의 분할 우선순위를 반환한다 (낮을수록 높은 우선순위)."""
    text = word.word.strip()
    for i, (pattern, _name) in enumerate(_SPLIT_PATTERNS):
        if pattern.search(text):
            return i
    return len(_SPLIT_PATTERNS)  # 일반 단어 경계 (최저 우선순위)


def _build_sub_segment(
    words: list[WordSegment],
    index: int,
    speaker: str | list[str] | None,
) -> TranscriptionSegment:
    """단어 목록에서 서브 세그먼트를 생성한다."""
    text = join_words(w.word for w in words)
    confidences = [w.confidence for w in words if w.confidence > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return TranscriptionSegment(
        index=index,
        text=text,
        start=words[0].start,
        end=words[-1].end,
        confidence=avg_conf,
        speaker=speaker,
        words=list(words),
    )
