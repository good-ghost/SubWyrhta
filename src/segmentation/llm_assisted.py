"""llm-assisted 세그멘테이션 모드.

advanced 모드 결과를 윈도우 단위로 LLM에 전달하여
분할점을 재조정한다. LLM 실패 시 fallback_mode(기본: advanced)로 자동 전환.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog

from src.core.config import LLMConfig, SegmentationConfig
from src.core.exceptions import LLMSegmentationError
from src.core.text import join_words
from src.models.segment import TranscriptionSegment
from src.segmentation.advanced import AdvancedSegmentation
from src.segmentation.base import SegmentationStrategy
from src.segmentation.speed_optimizer import get_chars_per_line
from src.segmentation.word_boundary_snapper import snap_segments_to_word_boundaries

if TYPE_CHECKING:
    from src.models.word import WordSegment

log = structlog.get_logger(__name__)

# 윈도우 설정
_WINDOW_SIZE = 10  # 한 번에 LLM에 전달할 세그먼트 수
_WINDOW_OVERLAP = 2  # 윈도우 간 겹침 세그먼트 수

# 기본 폴백 모드
_DEFAULT_FALLBACK_MODE = "advanced"

# ──────────────────────────────────────────────
# 시스템 프롬프트
# ──────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a subtitle segmentation optimizer. Given a list of subtitle segments \
with word-level timestamps, you must determine optimal split points to create \
natural, readable subtitles.

Rules:
1. Each segment should be 1-7 seconds long
2. Split at natural sentence or clause boundaries
3. Keep semantically related words together
4. Respect maximum characters per line constraints
5. Return ONLY a JSON array of word indices where splits should occur

Input format: JSON with segments containing words with timestamps.
Output format: JSON array of split point indices (0-based word indices where \
the NEXT segment should begin).

Example output: [5, 12, 18, 25]
"""


class LLMAssistedSegmentation(SegmentationStrategy):
    """llm-assisted 세그멘테이션.

    advanced 모드로 1차 분할 후, 윈도우 단위로 LLM에 전달하여
    분할점을 재조정한다.
    """

    def __init__(
        self,
        config: SegmentationConfig,
        llm_config: LLMConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._llm_config = llm_config or LLMConfig()
        self._advanced = AdvancedSegmentation(config)
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def mode_name(self) -> str:
        return "llm-assisted"

    @property
    def token_usage(self) -> dict[str, int]:
        """누적 토큰 사용량을 반환한다."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
        }

    def segment(
        self,
        words: list[WordSegment],
        language: str = "en",
    ) -> list[TranscriptionSegment]:
        """llm-assisted 모드로 세그먼트를 분할한다.

        1단계: advanced 모드로 1차 분할
        2단계: 윈도우 단위로 LLM에 전달하여 분할점 재조정
        3단계: 세그멘테이션 제약 조건 재검증

        Args:
            words: 단어 단위 타임스탬프 목록.
            language: 언어 코드.

        Returns:
            분할된 TranscriptionSegment 목록.
        """
        if not words:
            return []

        # 1단계: advanced 분할 (폴백용)
        advanced_segments = self._advanced.segment(words, language)

        if not advanced_segments:
            return advanced_segments

        # 2단계: LLM 재조정 시도
        try:
            segments = self._llm_resegment(words, advanced_segments, language)
        except (LLMSegmentationError, Exception) as e:
            log.warning(
                "LLM 세그멘테이션 실패, 폴백 사용",
                fallback=_DEFAULT_FALLBACK_MODE,
                error=str(e),
            )
            return advanced_segments

        # 3단계: 제약 조건 재검증
        segments = self._enforce_constraints(segments)

        log.info(
            "llm-assisted 세그멘테이션 완료",
            input_words=len(words),
            segments=len(segments),
            input_tokens=self._total_input_tokens,
            output_tokens=self._total_output_tokens,
        )
        return segments

    def _llm_resegment(
        self,
        words: list[WordSegment],
        advanced_segments: list[TranscriptionSegment],
        language: str,
    ) -> list[TranscriptionSegment]:
        """윈도우 단위로 LLM을 호출하여 분할점을 재조정한다."""
        from src.segmentation.llm_client import (
            RateLimiter,
            complete_with_retry,
            create_llm_provider,
        )

        provider = create_llm_provider(self._llm_config)
        rate_limiter = RateLimiter(self._llm_config.rpm)

        # advanced 결과를 윈도우로 분할
        windows = _create_windows(advanced_segments, _WINDOW_SIZE, _WINDOW_OVERLAP)

        all_split_indices: list[int] = []

        for _window_idx, window_segs in enumerate(windows):
            # 윈도우의 모든 단어를 수집
            window_words = _collect_words_from_segments(window_segs)
            if not window_words:
                continue

            # LLM 요청 생성
            user_prompt = _build_user_prompt(window_words, language, self.config)

            # LLM 호출
            response = complete_with_retry(
                provider,
                _SYSTEM_PROMPT,
                user_prompt,
                max_retries=self._llm_config.max_retries,
                rate_limiter=rate_limiter,
            )

            self._total_input_tokens += response.input_tokens
            self._total_output_tokens += response.output_tokens

            # 응답 파싱 → 분할점 인덱스
            split_indices = _parse_llm_response(response.text)

            # 글로벌 인덱스로 변환
            global_offset = _find_global_word_offset(words, window_words[0])
            for idx in split_indices:
                global_idx = global_offset + idx
                if global_idx not in all_split_indices and 0 < global_idx < len(words):
                    all_split_indices.append(global_idx)

        # 분할점 정렬
        all_split_indices.sort()

        # 분할점으로 세그먼트 생성
        if not all_split_indices:
            return advanced_segments

        segments = _build_segments_from_splits(words, all_split_indices)

        # 단어 경계 스냅
        segments = snap_segments_to_word_boundaries(segments, self.config.min_gap_s)

        return segments

    def _enforce_constraints(
        self,
        segments: list[TranscriptionSegment],
    ) -> list[TranscriptionSegment]:
        """세그멘테이션 제약 조건을 재검증한다.

        - max_duration_s 초과 세그먼트 강제 분할
        - min_duration_s 미만 세그먼트 유지 (min_display_guard에서 처리)
        """
        result: list[TranscriptionSegment] = []

        for seg in segments:
            if seg.duration > self.config.max_duration_s and len(seg.words) > 1:
                # 강제 분할: 중앙에서 분할
                split_segs = _force_split_segment(seg, self.config.max_duration_s)
                result.extend(split_segs)
            else:
                result.append(seg)

        # 인덱스 재부여
        for i, seg in enumerate(result, start=1):
            seg.index = i

        return result


# ──────────────────────────────────────────────
# 윈도우 생성
# ──────────────────────────────────────────────


def _create_windows(
    segments: list[TranscriptionSegment],
    window_size: int,
    overlap: int,
) -> list[list[TranscriptionSegment]]:
    """세그먼트를 겹침 있는 윈도우로 분할한다."""
    if len(segments) <= window_size:
        return [segments]

    windows: list[list[TranscriptionSegment]] = []
    step = max(1, window_size - overlap)
    i = 0

    while i < len(segments):
        end = min(i + window_size, len(segments))
        windows.append(segments[i:end])
        if end >= len(segments):
            break
        i += step

    return windows


# ──────────────────────────────────────────────
# 프롬프트 생성
# ──────────────────────────────────────────────


def _collect_words_from_segments(
    segments: list[TranscriptionSegment],
) -> list[WordSegment]:
    """세그먼트 목록에서 모든 단어를 수집한다."""
    words: list[WordSegment] = []
    for seg in segments:
        words.extend(seg.words)
    return words


def _build_user_prompt(
    words: list[WordSegment],
    language: str,
    config: SegmentationConfig,
) -> str:
    """LLM에 전달할 사용자 프롬프트를 생성한다."""
    word_data = [
        {
            "idx": i,
            "word": w.word.strip(),
            "start": round(w.start, 3),
            "end": round(w.end, 3),
        }
        for i, w in enumerate(words)
    ]

    constraints = {
        "language": language,
        "max_chars_per_line": get_chars_per_line(language, config),
        "max_lines": config.max_lines,
        "max_duration_s": config.max_duration_s,
        "min_duration_s": config.min_duration_s,
        "force_split_after_s": config.force_split_after_s,
    }

    prompt_data = {
        "words": word_data,
        "constraints": constraints,
    }

    return json.dumps(prompt_data, ensure_ascii=False)


# ──────────────────────────────────────────────
# 응답 파싱
# ──────────────────────────────────────────────


def _parse_llm_response(response_text: str) -> list[int]:
    """LLM 응답에서 분할점 인덱스 배열을 파싱한다.

    JSON 배열 형태로 응답을 기대하되, 코드 블록이나
    추가 텍스트가 있어도 최선을 다해 파싱한다.
    """
    text = response_text.strip()

    # 코드 블록 제거
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # JSON 배열 추출
    match = re.search(r"\[[\d\s,]*\]", text)
    if match:
        try:
            indices = json.loads(match.group())
            if isinstance(indices, list) and all(isinstance(x, int) for x in indices):
                return sorted(set(indices))
        except (json.JSONDecodeError, TypeError):
            pass

    log.warning("LLM 응답 파싱 실패, 빈 분할점 반환", response=text[:200])
    return []


# ──────────────────────────────────────────────
# 세그먼트 생성
# ──────────────────────────────────────────────


def _find_global_word_offset(
    all_words: list[WordSegment],
    target_word: WordSegment,
) -> int:
    """전체 단어 목록에서 대상 단어의 인덱스를 찾는다."""
    for i, w in enumerate(all_words):
        if w.start == target_word.start and w.word == target_word.word:
            return i
    return 0


def _build_segments_from_splits(
    words: list[WordSegment],
    split_indices: list[int],
) -> list[TranscriptionSegment]:
    """분할점 인덱스로 세그먼트를 생성한다.

    split_indices: 각 새 세그먼트가 시작하는 단어 인덱스 목록.
    """
    segments: list[TranscriptionSegment] = []

    # 분할점 앞에 0 추가, 뒤에 len(words) 추가
    boundaries = [0, *split_indices, len(words)]

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        seg_words = words[start_idx:end_idx]

        if not seg_words:
            continue

        text = join_words(w.word for w in seg_words)
        confidences = [w.confidence for w in seg_words if w.confidence > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        segments.append(
            TranscriptionSegment(
                index=len(segments) + 1,
                text=text,
                start=seg_words[0].start,
                end=seg_words[-1].end,
                confidence=avg_conf,
                words=list(seg_words),
            )
        )

    return segments


def _force_split_segment(
    seg: TranscriptionSegment,
    max_duration_s: float,
) -> list[TranscriptionSegment]:
    """max_duration_s를 초과하는 세그먼트를 강제 분할한다."""
    words = seg.words
    if len(words) <= 1:
        return [seg]

    # 중앙에서 분할
    mid = len(words) // 2
    left_words = words[:mid]
    right_words = words[mid:]

    result: list[TranscriptionSegment] = []

    for word_group in (left_words, right_words):
        if not word_group:
            continue
        text = join_words(w.word for w in word_group)
        confidences = [w.confidence for w in word_group if w.confidence > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        sub = TranscriptionSegment(
            index=seg.index,
            text=text,
            start=word_group[0].start,
            end=word_group[-1].end,
            confidence=avg_conf,
            speaker=seg.speaker,
            words=list(word_group),
        )

        # 재귀 분할
        if sub.duration > max_duration_s and len(word_group) > 1:
            result.extend(_force_split_segment(sub, max_duration_s))
        else:
            result.append(sub)

    return result
