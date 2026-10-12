"""STT 노이즈 필터링.

필러 단어 제거, 반복 단어 감지, 신뢰도 기반 필터링,
사용자 정의 필터 패턴을 지원한다.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from src.core.text import join_words
from src.models.segment import TranscriptionSegment

if TYPE_CHECKING:
    from src.core.config import NoiseFilterConfig
    from src.models.word import WordSegment

log = structlog.get_logger(__name__)

# ──────────────────────────────────────────────
# 다국어 필러 단어 사전
# ──────────────────────────────────────────────

FILLER_WORDS: dict[str, set[str]] = {
    "en": {
        "uh",
        "um",
        "ah",
        "hmm",
        "hm",
        "mm",
        "mhm",
        "uh-huh",
        "er",
        "eh",
        "oh",
        "like",
        "you know",
        "i mean",
        "so",
    },
    "ja": {
        "あー",
        "えー",
        "うーん",
        "うん",
        "えーと",
        "あのー",
        "あの",
        "まあ",
        "そのー",
        "その",
        "ええ",
        "ねえ",
    },
    "ko": {
        "어",
        "음",
        "아",
        "그",
        "저",
        "에",
        "으음",
        "어어",
        "그러니까",
        "뭐",
        "이제",
        "근데",
        "막",
    },
}

# 반복 감지 최소 횟수
_REPEAT_THRESHOLD = 3
# 반복 단위로 보는 최대 단어 수. 일본어는 문절로 묶여 `次回予告` 가 `次回|予告` 두 단어일 수 있다
# (`次回予告次回予告次回予告`, TO-DO 8)
_MAX_REPEAT_UNIT_WORDS = 4

# Whisper 가 무음·신음·음악 구간에서 지어내는 문구 (학습 데이터의 동영상 끝인사·자막 제작 표시).
# 세그먼트 전체가 이 문구(반복 포함)일 때만 지운다. 대화에도 나오는 인사(おやすみなさい,
# お疲れ様でした 등)는 넣지 않는다. 비교는 공백·문장 부호를 뺀 소문자로 한다
HALLUCINATION_PHRASES: dict[str, set[str]] = {
    "ja": {
        "ご視聴ありがとうございました",
        "ご清聴ありがとうございました",
        "最後までご視聴いただきありがとうございました",
        "チャンネル登録よろしくお願いします",
        "チャンネル登録お願いします",
        "チャンネル登録をお願いします",
        "高評価よろしくお願いします",
    },
    "en": {
        "thanksforwatching",
        "thankyouforwatching",
        "pleasesubscribe",
        "subtitlesbytheamaraorgcommunity",
    },
    "ko": {
        "시청해주셔서감사합니다",
        "구독과좋아요부탁드립니다",
    },
}

# 감탄사·신음만으로 된 일본어 단어 (ああ, んん, はぁ, おー, ふぅ). 문장 부호·장음은 무시한다.
# い 는 넣지 않는다 (はい, いい 는 말이다). うん·ええ 는 필러 목록에서 지운다
_INTERJECTION = re.compile(
    r"^(?:[あぁアァうぅウゥえぇエェおぉオォんンっッ]|[はハふフへヘほホ][ぁぅぇぉァゥェォっッ]?)+$"
)
_PUNCTUATION_AND_SPACE = re.compile(r"[\s、。，\uff0e,.!?！？…・ー〜~\-\"'「」『』()\uff08\uff09]")


def filter_noise(
    segments: list[TranscriptionSegment],
    config: NoiseFilterConfig,
) -> list[TranscriptionSegment]:
    """세그먼트 목록에서 노이즈를 필터링한다.

    처리 순서:
        1. 필러 단어 제거 (워드 레벨)
        2. 반복 단어 감지/제거
        3. 신뢰도 기반 필터링 (remove_low_confidence 일 때만 제거, 아니면 개수만 센다)
        4. 사용자 정의 패턴 제거
        5. 감탄사·신음 단어 제거 (remove_interjections)
        6. Whisper 가 지어낸 문구만 있는 세그먼트 제거
        7. 빈 세그먼트 제거

    Args:
        segments: 입력 세그먼트 목록.
        config: 노이즈 필터 설정.

    Returns:
        필터링된 세그먼트 목록.
    """
    if not config.enabled or not segments:
        return list(segments)

    # 필러 단어 세트 구성
    fillers = _build_filler_set(config.languages)

    # 사용자 패턴 컴파일
    custom_patterns = _compile_patterns(config.custom_patterns)

    result: list[TranscriptionSegment] = []
    removed_words = 0
    removed_segments = 0
    low_confidence_words = 0
    hallucinations = _build_hallucination_set(config.languages)

    for seg in segments:
        words = list(seg.words)

        # 1. 필러 단어 제거
        words, count = _remove_fillers(words, fillers)
        removed_words += count

        # 2. 반복 단어 제거
        words, count = _remove_repeats(words)
        removed_words += count

        # 3. 신뢰도 기반 필터링: 기본은 마킹만 (품질 리포트의 저신뢰도 수로 남는다).
        # 제거하면 작은 모델에서 정상 단어도 크게 빠진다 (실측 2026-10-08, faster-whisper tiny)
        if config.remove_low_confidence:
            words, count = _filter_by_confidence(words, config.confidence_threshold)
            removed_words += count
        else:
            low_confidence_words += sum(
                1 for w in words if 0 < w.confidence < config.confidence_threshold
            )

        # 4. 사용자 정의 패턴 제거
        words, count = _remove_custom_patterns(words, custom_patterns)
        removed_words += count

        # 5. 감탄사·신음 단어 제거
        if config.remove_interjections:
            kept = [w for w in words if not is_interjection(w.word)]
            removed_words += len(words) - len(kept)
            words = kept

        # 6. Whisper 가 지어낸 문구만 있는 세그먼트
        if words and _is_hallucination(join_words(w.word for w in words), hallucinations):
            removed_words += len(words)
            words = []

        # 7. 빈 세그먼트 제거
        if not words:
            removed_segments += 1
            continue

        # 텍스트 재구성
        new_text = join_words(w.word for w in words)
        if not new_text.strip():
            removed_segments += 1
            continue

        result.append(
            TranscriptionSegment(
                index=seg.index,
                text=new_text,
                start=words[0].start,
                end=words[-1].end,
                confidence=seg.confidence,
                speaker=seg.speaker,
                words=words,
            )
        )

    # 인덱스 재부여
    for i, seg in enumerate(result, start=1):
        seg.index = i

    if removed_words > 0 or removed_segments > 0 or low_confidence_words > 0:
        log.info(
            "노이즈 필터링 완료",
            removed_words=removed_words,
            removed_segments=removed_segments,
            low_confidence_words=low_confidence_words,
            remaining_segments=len(result),
        )

    return result


# ──────────────────────────────────────────────
# 필러 단어 제거
# ──────────────────────────────────────────────


def _build_filler_set(languages: list[str]) -> set[str]:
    """활성화된 언어의 필러 단어를 합친 세트를 반환한다."""
    fillers: set[str] = set()
    for lang in languages:
        fillers.update(FILLER_WORDS.get(lang, set()))
    return fillers


def _remove_fillers(
    words: list[WordSegment],
    fillers: set[str],
) -> tuple[list[WordSegment], int]:
    """필러 단어를 제거한다.

    Returns:
        (필터링된 단어 목록, 제거된 수).
    """
    result: list[WordSegment] = []
    removed = 0

    for word in words:
        normalized = word.word.strip().lower().rstrip(".,!?;:")
        if normalized in fillers:
            removed += 1
        else:
            result.append(word)

    return result, removed


# ──────────────────────────────────────────────
# 반복 단어 감지
# ──────────────────────────────────────────────


def _remove_repeats(
    words: list[WordSegment],
) -> tuple[list[WordSegment], int]:
    """같은 단어(또는 단어 묶음, 최대 _MAX_REPEAT_UNIT_WORDS 개)가 3회+ 연속 반복되면 첫 번째만 남긴다.

    Whisper 가 같은 문구를 되풀이해 지어낸 경우다. 두 번 반복(`よかったよかった`)은 그대로 둔다.

    Returns:
        (필터링된 단어 목록, 제거된 수).
    """
    if len(words) < _REPEAT_THRESHOLD:
        return words, 0

    keys = [w.word.strip().lower() for w in words]
    result: list[WordSegment] = []
    removed = 0
    i = 0
    while i < len(words):
        unit, count = _repeat_at(keys, i)
        if count >= _REPEAT_THRESHOLD:
            result.extend(words[i : i + unit])
            removed += unit * (count - 1)
            i += unit * count
        else:
            result.append(words[i])
            i += 1
    return result, removed


def _repeat_at(keys: list[str], start: int) -> tuple[int, int]:
    """start 에서 시작하는 가장 짧은 반복 단위의 (단어 수, 연속 반복 횟수)."""
    for unit in range(1, _MAX_REPEAT_UNIT_WORDS + 1):
        pattern = keys[start : start + unit]
        if len(pattern) < unit:
            break
        count = 1
        while keys[start + unit * count : start + unit * (count + 1)] == pattern:
            count += 1
        if count >= _REPEAT_THRESHOLD:
            return unit, count
    return 1, 1


# ──────────────────────────────────────────────
# 신뢰도 기반 필터링
# ──────────────────────────────────────────────


def _filter_by_confidence(
    words: list[WordSegment],
    threshold: float,
) -> tuple[list[WordSegment], int]:
    """신뢰도가 임계값 미만인 단어를 제거한다.

    confidence가 0.0인 단어는 신뢰도 미제공으로 간주하여 유지한다.

    Returns:
        (필터링된 단어 목록, 제거된 수).
    """
    result: list[WordSegment] = []
    removed = 0

    for word in words:
        # confidence==0.0은 미제공 → 유지
        if word.confidence > 0 and word.confidence < threshold:
            removed += 1
        else:
            result.append(word)

    return result, removed


# ──────────────────────────────────────────────
# 사용자 정의 패턴
# ──────────────────────────────────────────────


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    """사용자 정의 필터 패턴을 컴파일한다."""
    compiled: list[re.Pattern[str]] = []
    for p in patterns:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            log.warning("잘못된 필터 패턴 무시", pattern=p, error=str(e))
    return compiled


def _remove_custom_patterns(
    words: list[WordSegment],
    patterns: list[re.Pattern[str]],
) -> tuple[list[WordSegment], int]:
    """사용자 정의 패턴에 매칭되는 단어를 제거한다.

    Returns:
        (필터링된 단어 목록, 제거된 수).
    """
    if not patterns:
        return words, 0

    result: list[WordSegment] = []
    removed = 0

    for word in words:
        text = word.word.strip()
        if any(p.fullmatch(text) for p in patterns):
            removed += 1
        else:
            result.append(word)

    return result, removed


def is_interjection(text: str) -> bool:
    """감탄사·신음만으로 된 일본어 단어인지 (ああ, んん, はぁ, おー). 문장 부호만 있으면 False."""
    core = _PUNCTUATION_AND_SPACE.sub("", text)
    return bool(core) and bool(_INTERJECTION.match(core))


def _build_hallucination_set(languages: list[str]) -> set[str]:
    phrases: set[str] = set()
    for lang in languages:
        phrases.update(HALLUCINATION_PHRASES.get(lang, set()))
    return phrases


def _is_hallucination(text: str, phrases: set[str]) -> bool:
    """텍스트 전체가 지어낸 문구(한 번 이상 반복)인지."""
    core = _PUNCTUATION_AND_SPACE.sub("", text).lower()
    if not core or not phrases:
        return False
    for phrase in phrases:
        if core == phrase or (
            len(core) % len(phrase) == 0 and core == phrase * (len(core) // len(phrase))
        ):
            return True
    return False
