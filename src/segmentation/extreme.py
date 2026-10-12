"""extreme 세그멘테이션 모드.

advanced 모드 결과에 sentence-transformers 기반 semantic 유사도 분할을 추가 적용한다.
의미적으로 단절되는 지점에서 추가 분할하여 가장 자연스러운 자막을 생성한다.

sentence-transformers 미설치 시 advanced 모드로 폴백한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from src.core.text import join_words
from src.models.segment import TranscriptionSegment
from src.segmentation.advanced import AdvancedSegmentation
from src.segmentation.base import SegmentationStrategy
from src.segmentation.word_boundary_snapper import snap_segments_to_word_boundaries

if TYPE_CHECKING:
    from src.core.config import SegmentationConfig
    from src.models.word import WordSegment

log = structlog.get_logger(__name__)

# semantic 분할 기본 임계값 (유사도가 이 값 미만이면 분할)
_DEFAULT_SIMILARITY_THRESHOLD = 0.5

# 분할 검토 최소 세그먼트 길이 (이보다 짧으면 분할하지 않음)
_MIN_SEGMENT_WORDS_FOR_SPLIT = 4


def is_sentence_transformers_available() -> bool:
    """sentence-transformers 패키지 사용 가능 여부."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401

        return True
    except ImportError:
        return False


class ExtremeSegmentation(SegmentationStrategy):
    """extreme 세그멘테이션.

    advanced 모드로 1차 분할 후, sentence-transformers 기반
    semantic 유사도를 이용하여 의미적 단절 지점에서 추가 분할한다.
    """

    def __init__(self, config: SegmentationConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._advanced = AdvancedSegmentation(config)

    @property
    def mode_name(self) -> str:
        return "extreme"

    def segment(
        self,
        words: list[WordSegment],
        language: str = "en",
    ) -> list[TranscriptionSegment]:
        """extreme 모드로 세그먼트를 분할한다.

        1단계: advanced 모드 분할
        2단계: semantic 유사도 기반 추가 분할

        Args:
            words: 단어 단위 타임스탬프 목록.
            language: 언어 코드.

        Returns:
            분할된 TranscriptionSegment 목록.
        """
        if not words:
            return []

        # 1단계: advanced 분할
        segments = self._advanced.segment(words, language)

        if not segments:
            return segments

        # 2단계: semantic 분할
        if not is_sentence_transformers_available():
            log.warning("sentence-transformers 미설치, advanced 모드로 폴백")
            return segments

        try:
            segments = self._apply_semantic_split(segments, language)
        except Exception as e:
            log.warning("semantic 분할 실패, advanced 결과 사용", error=str(e))

        log.info(
            "extreme 세그멘테이션 완료",
            input_words=len(words),
            segments=len(segments),
        )
        return segments

    def _apply_semantic_split(
        self,
        segments: list[TranscriptionSegment],
        language: str,
    ) -> list[TranscriptionSegment]:
        """semantic 유사도 기반 추가 분할."""
        model = self._load_model()

        result: list[TranscriptionSegment] = []

        for seg in segments:
            if len(seg.words) < _MIN_SEGMENT_WORDS_FOR_SPLIT:
                result.append(seg)
                continue

            split_segs = self._try_semantic_split(seg, model)
            result.extend(split_segs)

        # 인덱스 재부여 + 경계 스냅
        for i, seg in enumerate(result, start=1):
            seg.index = i

        result = snap_segments_to_word_boundaries(result, self.config.min_gap_s)
        return result

    def _try_semantic_split(
        self,
        seg: TranscriptionSegment,
        model: Any,
    ) -> list[TranscriptionSegment]:
        """단일 세그먼트의 semantic 분할을 시도한다."""
        words = seg.words
        if len(words) < _MIN_SEGMENT_WORDS_FOR_SPLIT:
            return [seg]

        # 각 단어 위치에서 앞/뒤 텍스트의 semantic 유사도 계산
        best_split = self._find_semantic_split_point(words, model)

        if best_split is None:
            return [seg]

        # 분할
        left_words = words[: best_split + 1]
        right_words = words[best_split + 1 :]

        left = _build_segment(left_words, seg.index, seg.speaker, seg.confidence)
        right = _build_segment(right_words, seg.index + 1, seg.speaker, seg.confidence)

        return [left, right]

    def _find_semantic_split_point(
        self,
        words: list[WordSegment],
        model: Any,
    ) -> int | None:
        """semantic 유사도가 가장 낮은 분할 지점을 찾는다."""
        if len(words) < _MIN_SEGMENT_WORDS_FOR_SPLIT:
            return None

        min_sim = 1.0
        best_idx: int | None = None

        # 각 가능한 분할점에서 유사도 계산
        # 최소 2단어씩 양쪽에 있어야 함
        for i in range(1, len(words) - 2):
            left_text = join_words(w.word for w in words[: i + 1])
            right_text = join_words(w.word for w in words[i + 1 :])

            if not left_text.strip() or not right_text.strip():
                continue

            sim = self._compute_similarity(model, left_text, right_text)

            if sim < min_sim:
                min_sim = sim
                best_idx = i

        # 임계값 미만일 때만 분할
        if best_idx is not None and min_sim < _DEFAULT_SIMILARITY_THRESHOLD:
            return best_idx

        return None

    def _compute_similarity(
        self,
        model: Any,
        text_a: str,
        text_b: str,
    ) -> float:
        """두 텍스트의 cosine 유사도를 계산한다."""
        from sentence_transformers import util

        embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return float(similarity.item())

    def _load_model(self) -> Any:
        """sentence-transformers 모델을 로드한다 (캐싱)."""
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("sentence-transformers 모델 로드 완료", model="all-MiniLM-L6-v2")
        return self._model


def _build_segment(
    words: list[WordSegment],
    index: int,
    speaker: str | list[str] | None,
    confidence: float,
) -> TranscriptionSegment:
    """단어 목록에서 세그먼트를 생성한다."""
    text = join_words(w.word for w in words)
    return TranscriptionSegment(
        index=index,
        text=text,
        start=words[0].start,
        end=words[-1].end,
        confidence=confidence,
        speaker=speaker,
        words=list(words),
    )
