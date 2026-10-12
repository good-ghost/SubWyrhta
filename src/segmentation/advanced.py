"""advanced 세그멘테이션 모드.

Pause(발화 휴지) + Clause(절 경계) 기반 다층 신호 분할.
Hard Cap 강제 분할, 글자 수 트리거, 동적 임계값 감소, force_split_after_s를 지원한다.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from src.core.text import join_words
from src.models.segment import TranscriptionSegment
from src.segmentation.base import SegmentationStrategy
from src.segmentation.speed_optimizer import get_chars_per_line, lines_needed
from src.segmentation.word_boundary_snapper import (
    snap_segments_to_word_boundaries,
)

if TYPE_CHECKING:
    from src.core.config import SegmentationConfig
    from src.models.word import WordSegment

log = structlog.get_logger(__name__)

# ──────────────────────────────────────────────
# 분할 신호 점수
# ──────────────────────────────────────────────

# Pause 점수: 단어 간 묵음 길이 기반
_STRONG_PAUSE_S = 0.8
_STRONG_PAUSE_SCORE = 3.0
_MEDIUM_PAUSE_S = 0.5
_MEDIUM_PAUSE_SCORE = 1.0
_WEAK_PAUSE_S = 0.3
_WEAK_PAUSE_SCORE = 0.3

# Clause 점수: 구두점 기반 절 경계
_CLAUSE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"[.!?]+$"), 2.5),  # 문장 종결
    (re.compile(r"[;:]+$"), 1.5),  # 세미콜론, 콜론
    (re.compile(r",+$"), 1.0),  # 쉼표
    (re.compile(r"[)\]]+$"), 0.8),  # 닫는 괄호
    (re.compile(r"[、。！？]+$"), 2.5),  # CJK 구두점
    (re.compile(r"[，；：]+$"), 1.0),  # CJK 쉼표
]

# 분할 기본 임계값
_BASE_SPLIT_THRESHOLD = 2.0


class AdvancedSegmentation(SegmentationStrategy):
    """advanced 세그멘테이션.

    Pause + Clause 복합 신호 기반 분할.
    동적 임계값 감소, Hard Cap, force_split_after_s를 지원한다.
    """

    @property
    def mode_name(self) -> str:
        return "advanced"

    def segment(
        self,
        words: list[WordSegment],
        language: str = "en",
    ) -> list[TranscriptionSegment]:
        """단어 목록을 advanced 모드로 분할한다.

        Args:
            words: 단어 단위 타임스탬프 목록.
            language: 언어 코드.

        Returns:
            분할된 TranscriptionSegment 목록.
        """
        if not words:
            return []

        chars_per_line = get_chars_per_line(language, self.config)
        max_chars = chars_per_line * self.config.max_lines

        segments: list[TranscriptionSegment] = []
        current_words: list[WordSegment] = []
        seg_start_time = words[0].start

        for i, word in enumerate(words):
            # ── 0. 글자 수 상한: 이 단어를 넣으면 max_lines 줄에 들어가지 않으면 그 전에서 나눈다 ──
            # 예전에는 넘은 뒤 분할 점수가 있을 때만(아니면 1.2배까지) 나눠, 긴 문절이 붙으면
            # 두 줄에 들어가지 않는 자막(42자)이 나왔다. 글자 수 합만 보면 단어 길이 때문에 두 줄로
            # 나뉘지 않는 자막(9자 / 17자)이 남아 단어 경계로 채운 줄 수를 본다
            if current_words and (
                lines_needed([w.word for w in (*current_words, word)], chars_per_line)
                > self.config.max_lines
            ):
                cut = _best_cut(current_words, word, seg_start_time, self.config)
                segments.append(_build_segment(current_words[: cut + 1], len(segments) + 1))
                current_words = current_words[cut + 1 :]
                seg_start_time = current_words[0].start if current_words else word.start
            current_words.append(word)

            # 마지막 단어면 세그먼트 완성
            if i == len(words) - 1:
                segments.append(_build_segment(current_words, len(segments) + 1))
                break

            # 현재 세그먼트 상태 계산
            elapsed = word.end - seg_start_time
            char_count = sum(len(w.word.strip()) for w in current_words)
            next_word = words[i + 1]

            # ── 1. Hard Cap: max_duration_s 초과 → 즉시 강제 분할 ──
            if elapsed >= self.config.max_duration_s:
                segments.append(_build_segment(current_words, len(segments) + 1))
                current_words = []
                seg_start_time = next_word.start
                continue

            # ── 2. 글자 수 트리거: max_chars 초과 → 분할 ──
            if char_count >= max_chars:
                split_score = _compute_split_score(word, next_word, elapsed, self.config)
                # 글자 수 초과 시 임계값을 낮춰서 분할 확률 높임
                if split_score > 0.0 or char_count > max_chars * 1.2:
                    segments.append(_build_segment(current_words, len(segments) + 1))
                    current_words = []
                    seg_start_time = next_word.start
                    continue

            # ── 3. force_split_after_s 체크 ──
            if elapsed >= self.config.force_split_after_s:
                # 절 경계가 있으면 즉시 분할
                clause_score = _clause_score(word)
                if clause_score > 0:
                    segments.append(_build_segment(current_words, len(segments) + 1))
                    current_words = []
                    seg_start_time = next_word.start
                    continue

            # ── 4. 복합 신호 점수 기반 분할 ──
            split_score = _compute_split_score(word, next_word, elapsed, self.config)
            threshold = _dynamic_threshold(elapsed, self.config)

            if split_score >= threshold:
                segments.append(_build_segment(current_words, len(segments) + 1))
                current_words = []
                seg_start_time = next_word.start

        # 경계 스냅 + 겹침 제거
        segments = snap_segments_to_word_boundaries(segments, self.config.min_gap_s)

        log.info(
            "advanced 세그멘테이션 완료",
            input_words=len(words),
            segments=len(segments),
        )
        return segments


def _best_cut(
    current_words: list[WordSegment],
    next_word: WordSegment,
    seg_start_time: float,
    config: SegmentationConfig,
) -> int:
    """글자 수 상한에 걸렸을 때 나눌 자리 (current_words 의 몇 번째 단어 뒤). 분할 점수가 가장 높은 곳,
    같으면 뒤쪽 (앞 자막을 길게)."""
    best, best_score = len(current_words) - 1, -1.0
    for k, word in enumerate(current_words):
        following = current_words[k + 1] if k + 1 < len(current_words) else next_word
        score = _compute_split_score(word, following, word.end - seg_start_time, config)
        if score >= best_score:
            best, best_score = k, score
    return best


# ──────────────────────────────────────────────
# 분할 신호 계산
# ──────────────────────────────────────────────


def _compute_split_score(
    current_word: WordSegment,
    next_word: WordSegment,
    elapsed: float,
    config: SegmentationConfig,
) -> float:
    """현재 위치의 분할 점수를 계산한다.

    Pause 점수 + Clause 점수의 합산.
    """
    score = 0.0

    # Pause 점수
    score += _pause_score(current_word, next_word)

    # Clause 점수
    score += _clause_score(current_word)

    return score


def _pause_score(current_word: WordSegment, next_word: WordSegment) -> float:
    """단어 간 휴지(묵음) 길이 기반 점수."""
    gap = next_word.start - current_word.end

    if gap >= _STRONG_PAUSE_S:
        return _STRONG_PAUSE_SCORE
    if gap >= _MEDIUM_PAUSE_S:
        return _MEDIUM_PAUSE_SCORE
    if gap >= _WEAK_PAUSE_S:
        return _WEAK_PAUSE_SCORE
    return 0.0


def _clause_score(word: WordSegment) -> float:
    """구두점 기반 절 경계 점수."""
    text = word.word.strip()
    if not text:
        return 0.0

    for pattern, score in _CLAUSE_PATTERNS:
        if pattern.search(text):
            return score

    return 0.0


# ──────────────────────────────────────────────
# 동적 임계값
# ──────────────────────────────────────────────


def _dynamic_threshold(elapsed: float, config: SegmentationConfig) -> float:
    """경과 시간에 따라 분할 임계값을 동적으로 감소시킨다.

    규칙:
        - elapsed < start_s (3초)  → 100%
        - start_s ~ (start_s+10) (3~13초) → 100% → min_ratio 선형 감소
        - elapsed > (start_s+10) (13초+) → min_ratio 고정

    Args:
        elapsed: 현재 세그먼트 경과 시간(초).
        config: 세그멘테이션 설정.

    Returns:
        조정된 임계값.
    """
    if not config.dynamic_threshold:
        return _BASE_SPLIT_THRESHOLD

    start_s = config.dynamic_threshold_start_s  # 기본 3.0
    min_ratio = config.dynamic_threshold_min_ratio  # 기본 0.4
    ramp_duration = 10.0  # 선형 감소 구간 길이 (초)
    end_s = start_s + ramp_duration

    if elapsed < start_s:
        ratio = 1.0
    elif elapsed > end_s:
        ratio = min_ratio
    else:
        # 선형 보간: 1.0 → min_ratio
        progress = (elapsed - start_s) / ramp_duration
        ratio = 1.0 - progress * (1.0 - min_ratio)

    return _BASE_SPLIT_THRESHOLD * ratio


# ──────────────────────────────────────────────
# 세그먼트 빌더
# ──────────────────────────────────────────────


def _build_segment(words: list[WordSegment], index: int) -> TranscriptionSegment:
    """단어 목록에서 세그먼트를 생성한다."""
    text = join_words(w.word for w in words)

    confidences = [w.confidence for w in words if w.confidence > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return TranscriptionSegment(
        index=index,
        text=text,
        start=words[0].start,
        end=words[-1].end,
        confidence=avg_conf,
        words=list(words),
    )
