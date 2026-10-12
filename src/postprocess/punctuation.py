"""구두점 복원.

규칙 기반 + deepmultilingualpunctuation 모델 기반 구두점 복원을 지원한다.
"""

from __future__ import annotations

import re

import structlog

from src.models.segment import TranscriptionSegment

log = structlog.get_logger(__name__)


# ──────────────────────────────────────────────
# 규칙 기반 구두점 복원
# ──────────────────────────────────────────────

# 언어별 문장 종결 패턴 (마지막 단어에 구두점이 없으면 추가)
_SENTENCE_END_PATTERNS: dict[str, str] = {
    "en": ".",
    "ko": ".",
    "ja": "。",
    "zh": "。",
}

# 영어: 쉼표 삽입 힌트 패턴 (접속사/부사 앞)
_EN_COMMA_BEFORE: set[str] = {
    "but",
    "and",
    "or",
    "so",
    "yet",
    "however",
    "therefore",
    "meanwhile",
    "moreover",
    "furthermore",
    "although",
    "because",
    "while",
    "when",
    "if",
    "then",
}

# 한국어: 쉼표 삽입 힌트 — 독립 접속 부사 + 연결 어미
_KO_COMMA_CONJUNCTIONS: set[str] = {
    "그리고",
    "그러나",
    "그래서",
    "하지만",
    "그런데",
    "그러면",
    "그러므로",
}
_KO_COMMA_ENDINGS: list[str] = [
    "고",
    "며",
    "면",
    "지만",
    "는데",
    "니까",
]

# 일본어: 읽점(、) 삽입 힌트
_JA_COMMA_PARTICLES: set[str] = {
    "が",
    "けど",
    "けれど",
    "しかし",
    "そして",
    "また",
    "それから",
}


def restore_punctuation(
    segments: list[TranscriptionSegment],
    language: str = "en",
    use_model: bool = False,
) -> list[TranscriptionSegment]:
    """세그먼트의 구두점을 복원한다.

    Args:
        segments: 입력 세그먼트 목록.
        language: 언어 코드.
        use_model: True이면 deepmultilingualpunctuation 모델 사용 시도.

    Returns:
        구두점 복원된 세그먼트 목록.
    """
    if not segments:
        return []

    # 딥러닝 모델 시도
    if use_model:
        result = _restore_with_model(segments, language)
        if result is not None:
            return result
        log.info("모델 기반 복원 실패, 규칙 기반으로 폴백")

    # 규칙 기반 복원
    return _restore_rule_based(segments, language)


def _restore_rule_based(
    segments: list[TranscriptionSegment],
    language: str,
) -> list[TranscriptionSegment]:
    """규칙 기반 구두점 복원."""
    result: list[TranscriptionSegment] = []
    restored_count = 0

    for seg in segments:
        text = seg.text
        original = text

        # 1. 문장 끝 구두점 추가
        text = _add_sentence_end(text, language)

        # 2. 쉼표 삽입 (언어별)
        text = _add_commas(text, language)

        if text != original:
            restored_count += 1

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

    if restored_count > 0:
        log.info("규칙 기반 구두점 복원 완료", restored=restored_count, total=len(segments))

    return result


def _add_sentence_end(text: str, language: str) -> str:
    """문장 끝에 구두점이 없으면 추가한다."""
    text = text.rstrip()
    if not text:
        return text

    # 이미 구두점이 있으면 건너뜀
    end_chars = ".!?。！？…"
    if text[-1] in end_chars:
        return text

    punctuation = _SENTENCE_END_PATTERNS.get(language, ".")
    return text + punctuation


def _add_commas(text: str, language: str) -> str:
    """규칙 기반으로 쉼표를 삽입한다."""
    if language == "en":
        return _add_commas_en(text)
    if language == "ko":
        return _add_commas_ko(text)
    if language == "ja":
        return _add_commas_ja(text)
    return text


def _add_commas_en(text: str) -> str:
    """영어: 접속사/부사 앞에 쉼표 삽입."""
    words = text.split()
    if len(words) < 3:
        return text

    result: list[str] = [words[0]]
    for i in range(1, len(words)):
        word_lower = words[i].lower().rstrip(".,!?;:")
        # 문장 중간의 접속사 앞에 쉼표
        if word_lower in _EN_COMMA_BEFORE and i > 1:
            prev = result[-1]
            if not prev.endswith(",") and not prev.endswith(";"):
                result[-1] = prev + ","
        result.append(words[i])

    return " ".join(result)


def _add_commas_ko(text: str) -> str:
    """한국어: 접속 부사 앞 + 연결 어미 뒤에 쉼표 삽입."""
    words = text.split()
    if len(words) < 2:
        return text

    result: list[str] = [words[0]]
    for i in range(1, len(words)):
        # 접속 부사 앞에 쉼표
        if words[i] in _KO_COMMA_CONJUNCTIONS:
            prev = result[-1]
            if not prev.endswith(","):
                result[-1] = prev + ","
        result.append(words[i])

    text = " ".join(result)

    # 연결 어미로 끝나는 단어 뒤에 쉼표 (접속 부사 자체는 제외)
    words2 = text.split()
    result2: list[str] = []
    for i, word in enumerate(words2):
        clean = word.rstrip(",")
        is_conjunction = clean in _KO_COMMA_CONJUNCTIONS
        has_ending = not is_conjunction and any(
            clean.endswith(e) and len(clean) > len(e) for e in _KO_COMMA_ENDINGS
        )
        if has_ending and not word.endswith(",") and i < len(words2) - 1:
            result2.append(word + ",")
        else:
            result2.append(word)

    return " ".join(result2)


def _add_commas_ja(text: str) -> str:
    """일본어: 접속 조사 뒤에 읽점(、) 삽입."""
    for particle in _JA_COMMA_PARTICLES:
        text = re.sub(
            rf"({re.escape(particle)})([^\s、。])",
            r"\1、\2",
            text,
        )
    return text


# ──────────────────────────────────────────────
# 모델 기반 구두점 복원
# ──────────────────────────────────────────────


def is_punctuation_model_available() -> bool:
    """deepmultilingualpunctuation 모델 사용 가능 여부."""
    try:
        from deepmultilingualpunctuation import PunctuationModel  # noqa: F401

        return True
    except ImportError:
        return False


def _restore_with_model(
    segments: list[TranscriptionSegment],
    language: str,
) -> list[TranscriptionSegment] | None:
    """deepmultilingualpunctuation 모델로 구두점 복원."""
    if not is_punctuation_model_available():
        return None

    try:
        from deepmultilingualpunctuation import PunctuationModel

        model = PunctuationModel()

        result: list[TranscriptionSegment] = []
        for seg in segments:
            if not seg.text.strip():
                result.append(seg)
                continue

            restored_text = model.restore_punctuation(seg.text)

            result.append(
                TranscriptionSegment(
                    index=seg.index,
                    text=restored_text,
                    start=seg.start,
                    end=seg.end,
                    confidence=seg.confidence,
                    speaker=seg.speaker,
                    words=list(seg.words),
                )
            )

        log.info("모델 기반 구두점 복원 완료", segments=len(result))
        return result

    except Exception as e:
        log.warning("모델 기반 구두점 복원 실패", error=str(e))
        return None
