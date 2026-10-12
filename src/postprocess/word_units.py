"""STT 단어 단위 보정: 일본어 문절 묶기, 늘어진 단어 시간 자르기.

WhisperX 는 일본어를 글자 단위로 정렬한다. 그대로 세그멘테이션하면 단어 경계를 몰라
`ご視聴ありがとうございまし` / `た` 처럼 단어 중간에서 자막이 나뉜다. 형태소 분석(janome)으로
문절(조사·조동사·기호·접미어·비자립어는 앞 단어에, 접두어는 뒤 단어에 붙인다) 단위로 글자를 묶는다.

정렬이 무음·신음 구간의 글자 하나에 긴 시간을 주기도 한다 (`あーき` 8.9초). 한 단어는 나눌 수 없어
자막이 최대 표시 시간을 넘으므로, 글자 수에 비해 지나치게 긴 단어는 끝을 잘라낸다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import structlog

from src.models.word import WordSegment

if TYPE_CHECKING:
    from collections.abc import Sequence

log = structlog.get_logger(__name__)

# 앞 문절에 붙이는 품사 (IPADIC 대분류): 조사, 조동사, 기호(구두점 등)
_ATTACH_POS = frozenset({"助詞", "助動詞", "記号"})
# 앞 문절에 붙이는 품사 세분류: 비자립(てみる 의 み 등), 접미(さん 등)
_ATTACH_SUB_POS = frozenset({"非自立", "接尾"})
# 뒤 문절에 붙이는 품사: 접두사(ご予約 의 ご)
_PREFIX_POS = "接頭詞"
# 명사끼리 이어지면 복합어로 묶는다 (自動運転車両, ラストチャンス, 21世紀). 대명사·부사적 명사는 뺀다
# (私何も, 今日も 를 한 단어로 보지 않게)
_NOUN_POS = "名詞"
_NOT_COMPOUND_NOUN = frozenset({"代名詞", "副詞可能"})
# 동사 연용형 뒤의 동사는 복합 동사로 묶는다 (計り知れない)
_VERB_POS = "動詞"
_CONTINUATIVE = "連用形"
# 이 문자 앞에서는 문절이 시작하지 않는다 (작은 가나·장음). 형태소 분석이 긴 무부호 문장에서
# `まし|ょう` 처럼 잘못 끊는 경우를 막는다
_NO_PHRASE_START = frozenset("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶー")
# janome 은 사전에 없는 말을 만나면 뒤따르는 같은 글자 종류를 통째로 미등록어 하나로 묶는다
# (`行かなきゃいけないのに…この` → `行か|なき|ゃいけないのにどうしてこんなに|遅い…`). 작은 가나·장음으로
# 시작하는 미등록어는 그 글자를 떼고 나머지를 다시 분석한다. 다시 분석하는 깊이 상한
_UNKNOWN_NODE = "UNKNOWN"
_MAX_REANALYSIS_DEPTH = 3
# 이만큼 쉬면 형태소 분석을 따로 한다 (긴 무부호 문장을 한 번에 분석하면 결과가 무너진다)
_ANALYSIS_PAUSE_S = 0.2

# 정렬 붕괴: 언어별 CPS 상한의 이 배수보다 빠르게 이어진 구간이 _COLLAPSED_MIN_CHARS 글자 이상이면 몰린
# 것으로 본다 (TO-DO 9). 여러 사람의 대사가 이어진 긴 STT 세그먼트에서 50글자가 1.6초(초당 30자)에 몰렸다.
# 몰린 구간 끝 단어는 조금 덜 빠르다 (`お兄ちゃんの` 초당 14자). 일본어(CPS 7)에서 초당 12.6자. 글자 밀도가
# 다른 언어(영어 CPS 17 → 초당 30.6자)도 같은 비율로 본다
_COLLAPSED_CPS_RATIO = 1.8
_COLLAPSED_MIN_CHARS = 10
# 몰린 구간 바로 앞뒤 단어가 CPS 상한보다 느리면 남는 시간을 몰린 구간에 준다. 부호만 있는 단어는 0 글자
# 단어 가운데 있으면 그 뒤에서 단어를 나누는 부호 (문장·절 경계)
_INNER_BREAK_MARKS = frozenset("、。，,!?！？")
# 부호만 있는 단어에 남기는 시간
_PUNCTUATION_S = 0.02
# 전각 괄호는 ruff RUF001 때문에 chr 로 쓴다
_PUNCTUATION_ONLY = frozenset("、。，,.!?！？…・「」『』()" + chr(0xFF08) + chr(0xFF09))

# 단어 시간 상한: 글자당 0.5초, 짧은 단어도 1초까지는 둔다 (길게 끈 발음 "あー" 등)
_SECONDS_PER_CHAR_CAP = 0.5
_MIN_WORD_CAP_S = 1.0


def group_words_into_phrases(
    words: Sequence[WordSegment], language: str, text: str | None = None
) -> list[WordSegment]:
    """일본어 글자 단위 단어를 문절 단위 단어로 묶는다. 다른 언어는 그대로 돌려준다.

    단어 사이가 0.2초 이상 쉬면 그 앞뒤를 따로 분석한다. STT 원문(text)의 공백 자리는 단어 경계로
    지킨다 (일본어로 정렬한 영어 단어도 글자로 나뉘어, 공백 없이 이으면 `quantummobile` 처럼 붙는다).
    janome 이 없거나 형태소 분석 결과가 원문과 맞지 않으면 그대로 돌려준다 (글자 단위로 처리된다).

    Args:
        words: 한 STT 세그먼트의 단어 목록.
        language: 언어 코드.
        text: STT 세그먼트 원문 (공백 위치를 단어 경계로 쓴다). None 이면 쓰지 않는다.

    Returns:
        문절 단위 단어 목록 (시작은 첫 글자, 끝은 마지막 글자, 신뢰도는 평균).
    """
    if language != "ja" or len(words) <= 1:
        return list(words)
    tokenizer = _japanese_tokenizer()
    if tokenizer is None:
        return list(words)

    joined = "".join(w.word.strip() for w in words)
    ends: set[int] = set()
    offset = 0
    for chunk in _split_at_pauses(words):
        chunk_text = "".join(w.word.strip() for w in chunk)
        chunk_ends = _phrase_ends(tokenizer, chunk_text)
        if chunk_ends is None:
            log.warning(
                "일본어 형태소 분석 결과가 원문과 달라 글자 단위로 둔다", text=chunk_text[:40]
            )
            return list(words)
        ends |= {offset + e for e in chunk_ends}
        offset += len(chunk_text)
    if text is not None:
        ends |= _space_boundaries(text, joined)

    grouped: list[WordSegment] = []
    current: list[WordSegment] = []
    offset = 0
    for word in words:
        current.append(word)
        offset += len(word.word.strip())
        if offset in ends:
            grouped.append(_merge(current))
            current = []
    if current:
        grouped.append(_merge(current))
    return grouped


def clamp_word_durations(words: Sequence[WordSegment]) -> int:
    """글자 수에 비해 지나치게 긴 단어의 끝을 잘라낸다 (in-place).

    상한은 ``max(1초, 글자 수 * 0.5초)``. 시작은 두고 끝을 당긴다.

    Args:
        words: 단어 목록.

    Returns:
        잘라낸 단어 수.
    """
    clamped = 0
    for word in words:
        cap = max(_MIN_WORD_CAP_S, len(word.word.strip()) * _SECONDS_PER_CHAR_CAP)
        if word.duration > cap:
            word.end = word.start + cap
            clamped += 1
    return clamped


def spread_collapsed_words(words: Sequence[WordSegment], max_cps: float) -> int:
    """정렬이 몰린 구간의 시각을 앞뒤 단어의 남는 시간까지 넓혀 글자 수대로 다시 나눈다 (in-place).

    WhisperX 가 여러 사람의 대사가 이어진 긴 STT 세그먼트를 정렬하면 글자가 0.02~0.03초 간격으로 몰리고
    (50글자가 1.6초), 바로 뒤 단어나 부호가 남는 시간을 가져간다
    (뒤 4글자 단어가 1.44초, `!` 가 0.92초). 그대로 두면 자막이 0.6초처럼 짧아진다 (TO-DO 9).
    바로 앞뒤 한 단어만 줄여 맞는 시각을 넓게 바꾸지 않는다.

    Args:
        words: 한 STT 세그먼트의 단어 목록 (시간순).
        max_cps: 언어별 CPS 상한 (몰림 기준과 정상 속도).

    Returns:
        다시 나눈 구간 수.
    """
    fast_s_per_char = 1.0 / (max_cps * _COLLAPSED_CPS_RATIO)
    normal_s_per_char = 1.0 / max_cps
    _give_punctuation_time_to_previous(words)
    spread = 0
    i = 0
    while i < len(words):
        end = i
        while end < len(words) and _is_fast(words[end], fast_s_per_char):
            end += 1
        run = list(words[i:end])
        if sum(_chars(w) for w in run) >= _COLLAPSED_MIN_CHARS:
            before = words[i - 1] if i > 0 else None
            after = words[end] if end < len(words) else None
            _spread_run(run, before, after, normal_s_per_char)
            spread += 1
        i = max(end, i + 1)
    return spread


def split_at_inner_punctuation(words: Sequence[WordSegment]) -> list[WordSegment]:
    """단어 가운데 있는 문장 부호 뒤에서 단어를 나눈다 (시각은 글자 수 비율).

    Whisper 원문이나 LLM 부호 넣기가 부호를 문절 가운데 두면(`そうか?そうだよ、`, `見ただけだ。だから`)
    그 자리가 단어 경계가 아니라 절 경계 점수(단어 끝 부호)·CPS 분할·LLM 마지막 검토의 경계 옮기기와
    나누기가 쓰지 못한다. 부호만 있는 조각은 만들지 않는다 (이어진 부호는 앞 조각에 둔다).

    Args:
        words: 단어 목록.

    Returns:
        나눈 단어 목록 (나눌 곳이 없으면 같은 단어 객체).
    """
    result: list[WordSegment] = []
    for word in words:
        text = word.word.strip()
        cuts = [
            i + 1
            for i, ch in enumerate(text[:-1])
            if ch in _INNER_BREAK_MARKS
            and text[i + 1] not in _PUNCTUATION_ONLY
            and not (ch in ",，" and text[i - 1 : i].isdigit() and text[i + 1].isdigit())
        ]
        if not cuts:
            result.append(word)
            continue
        position = word.start
        for a, b in pairwise([0, *cuts, len(text)]):
            end = position + word.duration * (b - a) / len(text)
            result.append(WordSegment(text[a:b], position, end, word.confidence))
            position = end
        result[-1].end = word.end
    return result


def _give_punctuation_time_to_previous(words: Sequence[WordSegment]) -> None:
    """부호만 있는 단어가 쥔 시간을 앞 단어로 넘긴다 (4글자 단어 다음 `!` 0.92초). 부호는 끝에 짧게 남긴다."""
    for prev, word in pairwise(words):
        if _chars(word) == 0 and word.duration > _PUNCTUATION_S:
            prev.end = word.end - _PUNCTUATION_S
            word.start = prev.end


def _chars(word: WordSegment) -> int:
    """시간을 나눌 때 세는 글자 수 (부호는 세지 않는다)."""
    return sum(1 for ch in word.word.strip() if ch not in _PUNCTUATION_ONLY)


def _is_fast(word: WordSegment, fast_s_per_char: float) -> bool:
    chars = _chars(word)
    return chars == 0 or word.duration / chars < fast_s_per_char


def _spare(word: WordSegment, normal_s_per_char: float) -> float:
    """정상 속도로 말할 때보다 남는 시간."""
    return max(0.0, word.duration - _chars(word) * normal_s_per_char)


def _spread_run(
    run: list[WordSegment],
    before: WordSegment | None,
    after: WordSegment | None,
    normal_s_per_char: float,
) -> None:
    start, end = run[0].start, run[-1].end
    if before is not None:
        before.end -= _spare(before, normal_s_per_char)
        start = min(start, before.end)
    if after is not None:
        after.start += _spare(after, normal_s_per_char)
        end = max(end, after.start)
    total = sum(_chars(w) for w in run)
    position = start
    for word in run:
        word.start = position
        position += (end - start) * _chars(word) / total
        word.end = position


def _split_at_pauses(words: Sequence[WordSegment]) -> list[list[WordSegment]]:
    """단어 사이가 _ANALYSIS_PAUSE_S 이상 쉬는 곳에서 나눈다."""
    chunks: list[list[WordSegment]] = [[words[0]]]
    for prev, word in pairwise(words):
        if word.start - prev.end >= _ANALYSIS_PAUSE_S:
            chunks.append([])
        chunks[-1].append(word)
    return chunks


def _space_boundaries(text: str, joined: str) -> set[int]:
    """원문 공백이 있던 자리 (공백을 뺀 글자 위치). 원문과 단어가 다르면 빈 집합."""
    if "".join(text.split()) != joined:
        return set()
    boundaries: set[int] = set()
    count = 0
    for ch in text.strip():
        if ch.isspace():
            boundaries.add(count)
        else:
            count += 1
    return boundaries


def japanese_morpheme_boundaries(text: str) -> set[int]:
    """형태소 경계(작은 가나·장음 앞은 뺀다). 문절이 길어 줄에 들어가지 않을 때 줄바꿈 후보로 쓴다."""
    tokenizer = _japanese_tokenizer()
    if tokenizer is None or not text:
        return set()
    positions: set[int] = set()
    position = 0
    for morpheme in _morphemes(tokenizer, text):
        position += len(morpheme.surface)
        if position < len(text) and text[position] not in _NO_PHRASE_START:
            positions.add(position)
    return positions


def japanese_phrase_boundaries(text: str) -> set[int] | None:
    """일본어 텍스트의 문절 경계(글자 위치) 집합. janome 이 없거나 분석이 원문과 다르면 None.

    줄바꿈 위치를 고를 때 쓴다 (``src/postprocess/subtitle_text.py``).
    """
    tokenizer = _japanese_tokenizer()
    if tokenizer is None or not text:
        return None
    return _phrase_ends(tokenizer, text)


def japanese_tokenizer() -> Any:
    """janome 형태소 분석기 (한 번만 만든다). janome 이 없으면 None. 읽기 비교(``reading``)도 쓴다."""
    return _japanese_tokenizer()


@lru_cache(maxsize=1)
def _japanese_tokenizer() -> Any:
    """janome 형태소 분석기 (사전을 읽는 데 시간이 걸려 한 번만 만든다). 없으면 None."""
    try:
        from janome.tokenizer import Tokenizer
    except ImportError:
        log.warning("janome 패키지가 없어 일본어 단어 경계 없이 글자 단위로 처리한다")
        return None
    return Tokenizer()


@dataclass(frozen=True)
class _Morpheme:
    """형태소 (janome Token 에서 쓰는 값만)."""

    surface: str
    part_of_speech: str
    infl_form: str = ""


def _morphemes(tokenizer: Any, text: str, depth: int = 0) -> list[_Morpheme]:
    """형태소 목록. 작은 가나·장음으로 시작하는 미등록어는 그 글자를 떼고 나머지를 다시 분석한다."""
    result: list[_Morpheme] = []
    for token in tokenizer.tokenize(text):
        surface = str(token.surface)
        head = len(surface) - len(surface.lstrip("".join(_NO_PHRASE_START)))
        if (
            str(getattr(token, "node_type", "")) == _UNKNOWN_NODE
            and 0 < head < len(surface)
            and depth < _MAX_REANALYSIS_DEPTH
        ):
            # 뗀 글자는 앞 문절에 붙는다 (_phrase_ends 의 작은 가나 규칙). 품사는 붙는 쪽인 기호로 둔다
            result.append(_Morpheme(surface[:head], "記号"))
            result.extend(_morphemes(tokenizer, surface[head:], depth + 1))
            continue
        result.append(
            _Morpheme(surface, str(token.part_of_speech), str(getattr(token, "infl_form", "")))
        )
    return result


def _phrase_ends(tokenizer: Any, text: str) -> set[int] | None:
    """문절이 끝나는 글자 위치 집합. 형태소를 이은 길이가 원문과 다르면 None."""
    ends: list[int] = []
    position = 0
    after_prefix = False
    prev_major, prev_sub, prev_infl = "", "", ""
    for token in _morphemes(tokenizer, text):
        surface = token.surface
        pos_major, pos_sub = [*token.part_of_speech.split(","), "", ""][:2]
        compound_noun = (
            prev_major == _NOUN_POS
            and pos_major in (_NOUN_POS, _PREFIX_POS)
            and prev_sub not in _NOT_COMPOUND_NOUN
            and pos_sub not in _NOT_COMPOUND_NOUN
        )
        compound_verb = (
            prev_major == _VERB_POS and pos_major == _VERB_POS and prev_infl == _CONTINUATIVE
        )
        attach = bool(ends) and (
            after_prefix
            or pos_major in _ATTACH_POS
            or pos_sub in _ATTACH_SUB_POS
            or compound_noun
            or compound_verb
            or surface[:1] in _NO_PHRASE_START
        )
        position += len(surface)
        if attach:
            ends[-1] = position
        else:
            ends.append(position)
        after_prefix = pos_major == _PREFIX_POS
        prev_major, prev_sub, prev_infl = pos_major, pos_sub, token.infl_form
    if position != len(text):
        return None
    return set(ends)


def _merge(words: list[WordSegment]) -> WordSegment:
    if len(words) == 1:
        return words[0]
    return WordSegment(
        word="".join(w.word.strip() for w in words),
        start=min(w.start for w in words),
        end=max(w.end for w in words),
        confidence=sum(w.confidence for w in words) / len(words),
    )
