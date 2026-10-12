"""언어별 읽기(발음) 비교: LLM 오인식 교정이 "소리는 같고 표기만 다른" 교정인지 판단한다.

교정 전후 텍스트를 언어별 읽기로 바꿔 편집 거리 비율을 잰다 (WI-9.018).

- ja: janome 읽기(카타카나)를 발음으로 정규화 (탁점·반탁점을 떼고, 장음 `エイ`·`オウ` 를 `ー` 로). `機会` /
  `機械` 는 둘 다 キカイ (거리 0). janome 가 드문 읽기를 주는 말도 탁점·장음 차이는 같게 본다
- en: 소문자 영문자·숫자만 (글자 편집 거리)
- ko: 한글을 자모로 풀어 비교 (`스미카` / `스네카` 는 모음 하나 차이)

기본은 영어·일본어·한국어다 (2026-10-11 사용자 결정). 다른 언어는 필요할 때 ``register_reading`` 으로
읽기 함수를 등록하면 교정 대상이 된다 (등록하지 않은 언어는 교정하지 않는다).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable

ReadingFn = Callable[[str], str]

# 히라가나 → 카타카나 (같은 소리를 같은 글자로 비교한다)
_HIRAGANA_START, _HIRAGANA_END = 0x3041, 0x3096
_KATAKANA_OFFSET = 0x60

_READERS: dict[str, ReadingFn] = {}


def register_reading(language: str, reader: ReadingFn) -> None:
    """언어의 읽기 함수를 등록한다 (그 언어를 교정 대상으로 만든다).

    Args:
        language: 언어 코드 (`zh` 등).
        reader: 텍스트를 비교용 읽기 문자열로 바꾸는 함수.
    """
    _READERS[language] = reader


def is_supported(language: str) -> bool:
    """읽기 함수가 등록된 언어인지."""
    return language in _READERS


def reading(text: str, language: str) -> str | None:
    """텍스트의 비교용 읽기. 등록하지 않은 언어는 None."""
    reader = _READERS.get(language)
    return reader(text) if reader is not None else None


def reading_distance_ratio(a: str, b: str, language: str) -> float | None:
    """두 텍스트의 읽기 편집 거리 / 긴 쪽 읽기 길이. 등록하지 않은 언어나 읽기가 비면 None.

    Args:
        a: 교정 전 텍스트.
        b: 교정 후 텍스트.
        language: 언어 코드.

    Returns:
        0.0 (같은 읽기) ~ 1.0.
    """
    ra, rb = reading(a, language), reading(b, language)
    if ra is None or rb is None or not (ra or rb):
        return None
    return _levenshtein(ra, rb) / max(len(ra), len(rb))


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _to_katakana(text: str) -> str:
    return "".join(
        chr(ord(ch) + _KATAKANA_OFFSET) if _HIRAGANA_START <= ord(ch) <= _HIRAGANA_END else ch
        for ch in text
    )


def _japanese(text: str) -> str:
    """janome 읽기(카타카나). 읽기가 없는 말(미등록 가타카나·영문)은 표기를 쓴다. 부호·공백은 뺀다."""
    from src.postprocess.word_units import japanese_tokenizer

    tokenizer = japanese_tokenizer()
    if tokenizer is None:
        return _normalize_japanese(_to_katakana(_letters(text)))
    parts: list[str] = []
    for token in tokenizer.tokenize(text):
        value = str(getattr(token, "reading", "*"))
        parts.append(value if value not in ("", "*") else str(token.surface))
    return _normalize_japanese(_to_katakana(_letters("".join(parts))))


# 발음 정규화: 장음으로 읽히는 모음 연속을 장음 부호로 (`ー` 를 빼면 장음이 많은 말의 비교 길이가 짧아져
# 비율이 커졌다: `スーパーマーケット` / `スーパーマーケット` 처럼 장음이 많은 말)
_LONG_VOWELS = (("エイ", "エー"), ("オウ", "オー"))
_DAKUTEN_MARKS = frozenset("\u3099\u309a")


def _normalize_japanese(kana: str) -> str:
    """탁점·반탁점을 떼고 장음 표기를 맞춘다 (같은 소리로 들리는 표기 차이를 무시한다)."""
    text = "".join(ch for ch in unicodedata.normalize("NFD", kana) if ch not in _DAKUTEN_MARKS)
    text = unicodedata.normalize("NFC", text)
    for long_vowel, short in _LONG_VOWELS:
        text = text.replace(long_vowel, short)
    return text


def _latin(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isascii() and ch.isalnum())


def _hangul(text: str) -> str:
    """한글 음절을 자모로 푼다 (NFD). 부호·공백은 뺀다."""
    return _letters(unicodedata.normalize("NFD", text))


def _letters(text: str) -> str:
    """글자(문자·숫자·결합 자모)만 남긴다."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LNM")


register_reading("ja", _japanese)
register_reading("en", _latin)
register_reading("ko", _hangul)
