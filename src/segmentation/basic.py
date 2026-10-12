"""basic 세그멘테이션 모드.

PySBD 문장 경계 감지 + 단어 타임스탬프 기반 경계 스냅.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.core.exceptions import SegmentationError
from src.core.text import join_words
from src.models.segment import TranscriptionSegment
from src.segmentation.base import SegmentationStrategy
from src.segmentation.word_boundary_snapper import (
    snap_segments_to_word_boundaries,
    verify_no_overlap,
)

if TYPE_CHECKING:
    from src.models.word import WordSegment

log = structlog.get_logger(__name__)


class BasicSegmentation(SegmentationStrategy):
    """PySBD 기반 basic 세그멘테이션.

    PySBD로 문장 경계를 감지한 뒤, 단어 타임스탬프 기반으로
    경계를 스냅하여 타임코드 겹침을 방지한다.
    """

    @property
    def mode_name(self) -> str:
        return "basic"

    def segment(
        self,
        words: list[WordSegment],
        language: str = "en",
    ) -> list[TranscriptionSegment]:
        """PySBD로 문장 경계를 감지하고 세그먼트를 생성한다.

        Args:
            words: 단어 단위 타임스탬프 목록.
            language: 언어 코드 (PySBD 언어 설정).

        Returns:
            분할된 TranscriptionSegment 목록.

        Raises:
            SegmentationError: PySBD 로드 실패 시.
        """
        if not words:
            return []

        try:
            import pysbd
        except ImportError as e:
            raise SegmentationError(
                "pysbd 패키지가 설치되지 않았습니다. pip install pysbd 로 설치해 주세요."
            ) from e

        # 전체 텍스트 조합
        full_text = join_words(w.word for w in words)

        # PySBD 문장 분할
        pysbd_lang = _map_language(language)
        segmenter = pysbd.Segmenter(language=pysbd_lang, clean=False)
        sentences = segmenter.segment(full_text)

        if not sentences:
            return [_words_to_segment(words, index=1)]

        # 문장 → 세그먼트 변환 (단어 경계 스냅)
        segments = _split_by_sentences(words, sentences, self.config.min_gap_s)

        # 경계 스냅 + 겹침 제거
        segments = snap_segments_to_word_boundaries(segments, self.config.min_gap_s)

        # 겹침 검증
        violations = verify_no_overlap(segments, self.config.min_gap_s)
        if violations:
            log.warning(
                "타임코드 겹침 감지",
                violations=len(violations),
                details=violations[:3],
            )

        log.info(
            "basic 세그멘테이션 완료",
            sentences=len(sentences),
            segments=len(segments),
        )
        return segments


def _map_language(language: str) -> str:
    """언어 코드를 PySBD 지원 형식으로 매핑한다."""
    pysbd_languages = {
        "en": "en",
        "ko": "ko",
        "ja": "ja",
        "zh": "zh",
        "de": "de",
        "fr": "fr",
        "es": "es",
        "it": "it",
        "pt": "pt",
        "ru": "ru",
        "pl": "pl",
        "nl": "nl",
        "ar": "ar",
        "hi": "hi",
        "da": "da",
        "el": "el",
        "sk": "sk",
        "bg": "bg",
        "my": "my",
        "am": "am",
        "ur": "ur",
        "kk": "kk",
    }
    return pysbd_languages.get(language, "en")


def _split_by_sentences(
    words: list[WordSegment],
    sentences: list[str],
    min_gap_s: float,
) -> list[TranscriptionSegment]:
    """PySBD 문장 목록을 기반으로 단어를 세그먼트로 분할한다.

    문자 오프셋으로 분할점을 결정한 뒤, 단어 경계로 스냅한다.
    """
    if not words or not sentences:
        return []

    segments: list[TranscriptionSegment] = []
    word_idx = 0

    for sent_idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        # 이 문장에 해당하는 단어 범위 찾기
        sent_words = _find_words_for_sentence(words, word_idx, sentence)

        if not sent_words:
            continue

        word_start_idx = words.index(sent_words[0], word_idx)
        word_end_idx = words.index(sent_words[-1], word_start_idx)

        seg = TranscriptionSegment(
            index=sent_idx + 1,
            text=sentence,
            start=sent_words[0].start,
            end=sent_words[-1].end,
            confidence=_avg_confidence(sent_words),
            words=sent_words,
        )
        segments.append(seg)

        word_idx = word_end_idx + 1

    # 남은 단어가 있으면 마지막 세그먼트에 추가
    if word_idx < len(words):
        remaining = words[word_idx:]
        if segments:
            last = segments[-1]
            last.text = join_words([last.text, *(w.word for w in remaining)])
            last.words = list(last.words) + remaining
            last.end = remaining[-1].end
        else:
            segments.append(_words_to_segment(remaining, index=1))

    # 인덱스 재부여
    for i, seg in enumerate(segments, start=1):
        seg.index = i

    return segments


def _find_words_for_sentence(
    words: list[WordSegment],
    start_idx: int,
    sentence: str,
) -> list[WordSegment]:
    """문장에 해당하는 단어 범위를 찾는다.

    단어를 순차적으로 누적하면서 문장 텍스트와 매칭한다.
    """
    if start_idx >= len(words):
        return []

    sent_words: list[WordSegment] = []
    accumulated = ""

    # 문장에서 공백/특수문자 제거한 정규화 텍스트
    sent_normalized = _normalize_for_match(sentence)

    for i in range(start_idx, len(words)):
        word = words[i]
        sent_words.append(word)
        accumulated += _normalize_for_match(word.word)

        # 누적 텍스트가 문장을 포함하면 완료
        if len(accumulated) >= len(sent_normalized):
            break

    return sent_words


def _normalize_for_match(text: str) -> str:
    """매칭을 위해 텍스트를 정규화한다 (공백, 대소문자 제거)."""
    return "".join(text.lower().split())


def _words_to_segment(words: list[WordSegment], index: int) -> TranscriptionSegment:
    """단어 목록을 하나의 세그먼트로 만든다."""
    text = join_words(w.word for w in words)
    return TranscriptionSegment(
        index=index,
        text=text,
        start=words[0].start,
        end=words[-1].end,
        confidence=_avg_confidence(words),
        words=list(words),
    )


def _avg_confidence(words: list[WordSegment]) -> float:
    """단어 목록의 평균 신뢰도를 계산한다."""
    if not words:
        return 0.0
    confidences = [w.confidence for w in words if w.confidence > 0]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)
