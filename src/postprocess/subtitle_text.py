"""자막 텍스트 마무리: 발화 구분, 앞에 붙은 문장 부호, 줄바꿈.

세그멘테이션이 끝난 자막에 적용한다 (``TranscriptionPipeline._segment``).

- 발화 구분: 서로 다른 STT 세그먼트(발화)가 한 자막에 합쳐지면 일본어·중국어는 사이에 전각 공백을 넣는다.
  공백 없이 이으면 `初めてですあ、はい` 처럼 한 문장으로 읽힌다.
- 문장 부호: 자막이 `?` `。` `ー` 같은 부호로 시작하면 앞 자막 끝으로 옮긴다. 앞 발화의 끝이
  다음 STT 세그먼트 첫 글자로 넘어온 경우다 (`?いい子です`). 부호만 남은 자막은 지운다.
- 줄바꿈: 줄당 최대 글자 수(언어별 기본값)를 넘으면 두 줄로 나눈다. 자막의 단어(일본어는 문절) 경계에서
  나누고, 발화 구분·문장 부호 뒤를 먼저 고른다. 단어와 텍스트가 맞지 않으면 일본어는 형태소 분석,
  그 밖은 공백에서 나눈다. 작은 가나·장음 앞에서는 나누지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

from src.core.text import CJK_UTTERANCE_SEPARATOR, join_utterances, join_words

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.models.segment import TranscriptionSegment

# 자막 앞에서 앞 자막으로 옮기는 부호. 말줄임표(…)는 이어지는 말을 뜻해 옮기지 않는다
_LEADING_PUNCTUATION = re.compile(r"^[、。，\uff0e,.!?！？ー〜~]+")
# 앞 자막 끝 부호와 옮긴 부호를 합칠 때 보는 부호. 물음·느낌표가 있으면 쉼표·마침표는 버린다
_SENTENCE_MARKS = frozenset("?!？！")
_SOFT_MARKS = frozenset("、。，,.\uff0e")
# 합친 부호는 이 글자 수까지 (`!?`)
_MAX_MARK_RUN = 2
# 자막 안에서 이어진 문장 부호 (말줄임표는 넣지 않는다)
_MARK_RUN = re.compile("[" + re.escape("".join(sorted(_SENTENCE_MARKS | _SOFT_MARKS))) + "]{2,}")
# 줄을 나누기 좋은 자리: 이 문자 뒤
_PREFERRED_BREAK_AFTER = frozenset("、。，,.!?！？… " + chr(0xFF0E) + CJK_UTTERANCE_SEPARATOR)
_LINE_EDGE_SPACES = " " + CJK_UTTERANCE_SEPARATOR
# 이 문자 앞에서는 줄을 나누지 않는다 (작은 가나·장음: 단어 중간이다)
_NO_LINE_START = frozenset("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶー")


def mark_utterance_breaks(
    segments: Sequence[TranscriptionSegment], utterance_ends: set[int]
) -> None:
    """자막 안에서 발화가 바뀌는 곳에 구분을 넣어 텍스트를 다시 만든다 (in-place).

    단어로 만든 텍스트가 지금 텍스트와 같은 자막만 다시 만든다 (겹침 병합처럼 텍스트를 따로 이은
    다화자 자막은 그대로 둔다).

    Args:
        segments: 자막 목록.
        utterance_ends: 발화(STT 세그먼트)의 마지막 단어 ``id`` 집합.
    """
    for seg in segments:
        if isinstance(seg.speaker, list) or len(seg.words) < 2:
            continue
        if join_words(w.word for w in seg.words) != seg.text:
            continue
        utterances: list[list[str]] = [[]]
        for i, word in enumerate(seg.words):
            utterances[-1].append(word.word)
            if id(word) in utterance_ends and i < len(seg.words) - 1:
                utterances.append([])
        if len(utterances) > 1:
            seg.text = join_utterances(join_words(u) for u in utterances)


def move_leading_punctuation(
    segments: Sequence[TranscriptionSegment],
) -> list[TranscriptionSegment]:
    """자막 앞의 문장 부호를 앞 자막 끝으로 옮기고, 부호만 남은 자막은 지운다.

    Args:
        segments: 자막 목록 (텍스트가 바뀐다).

    Returns:
        부호만 남은 자막을 뺀 목록 (번호를 다시 매긴다).
    """
    result: list[TranscriptionSegment] = []
    for seg in segments:
        text = seg.text.lstrip(_LINE_EDGE_SPACES)
        match = _LEADING_PUNCTUATION.match(text)
        if match:
            if result:
                result[-1].text = _append_punctuation(result[-1].text, match.group(0))
            text = text[match.end() :].lstrip(_LINE_EDGE_SPACES)
        if not text:
            continue
        # Whisper 원문에 이미 이어진 부호도 합친다 (`もう8時!??`, 단어 `!?` + `?`)
        seg.text = _MARK_RUN.sub(lambda m: _merge_marks(m.group(0)), text)
        result.append(seg)
    for i, seg in enumerate(result, start=1):
        seg.index = i
    return result


def _append_punctuation(text: str, moved: str) -> str:
    """앞 자막 끝에 옮긴 부호를 붙인다. 앞 자막이 부호로 끝나면 둘을 합친다.

    예전에는 그대로 이어 `いかがですか??`, `急いで!??` 가 됐고, LLM 부호 넣기가 STT 문장 끝에 `。` 를
    넣으면 `すみません。?` 가 됐다 (TO-DO 3). 물음·느낌표가 있으면 쉼표·마침표는 버리고, 같은 부호는
    하나로, 최대 두 글자(`!?`)까지 둔다. 장음·물결(`ー〜`)이 섞이면 그대로 잇는다.
    """
    body = text.rstrip(_LINE_EDGE_SPACES)
    marks = _SENTENCE_MARKS | _SOFT_MARKS
    if not set(moved) <= marks:
        return body + moved
    stem = body.rstrip("".join(marks))
    return stem + _merge_marks(body[len(stem) :] + moved)


def _merge_marks(run: str) -> str:
    """이어진 문장 부호를 합친다: 물음·느낌표가 있으면 쉼표·마침표는 버리고, 같은 부호는 하나로, 최대 두 글자."""
    if any(ch in _SENTENCE_MARKS for ch in run):
        run = "".join(ch for ch in run if ch in _SENTENCE_MARKS)
    else:
        run = run[-1]
    merged = ""
    for ch in run:
        if ch not in merged:
            merged += ch
    return merged[:_MAX_MARK_RUN]


def wrap_lines(
    segments: Sequence[TranscriptionSegment],
    language: str,
    chars_per_line: int,
    max_lines: int,
    speaker_separator: str = "",
) -> int:
    """줄당 글자 수를 넘는 자막을 두 줄로 나눈다 (in-place).

    Args:
        segments: 자막 목록.
        language: 언어 코드 (일본어는 문절 경계에서 나눈다).
        chars_per_line: 줄당 최대 글자 수.
        max_lines: 최대 줄 수. 1 이면 나누지 않는다.
        speaker_separator: 다화자 자막의 화자 구분자 (``segmentation.overlap_text_separator``).
            그 자리에서 줄을 나누면 구분자를 지운다 (화자마다 한 줄, TO-DO 12).

    Returns:
        나눈 자막 수.
    """
    if max_lines < 2:
        return 0
    wrapped = 0
    for seg in segments:
        if "\n" in seg.text or len(seg.text) <= chars_per_line:
            continue
        text = _wrapped_text(seg, language, chars_per_line, speaker_separator)
        if text is not None:
            seg.text = text
            wrapped += 1
    return wrapped


def _wrapped_text(
    seg: TranscriptionSegment, language: str, limit: int, separator: str
) -> str | None:
    """두 줄로 나눈 텍스트. 나눌 자리가 없으면 None.

    어디서도 두 줄에 들어가지 않고 줄 가운데에 발화 구분(전각 공백)이 있으면, 구분을 빼고 다시 나눠 본다.
    advanced 는 나중에 들어갈 발화 구분을 모르고 줄 수를 세 한 글자 차이로 넘친다
    (긴 복합 명사 두 개 사이에 구분이 있으면 15 / 17 → 구분을 빼면 15 / 16, TO-DO 11).
    """
    text = _split_at(seg, language, limit, separator)
    if text is None or _fits(text, limit) or CJK_UTTERANCE_SEPARATOR not in seg.text:
        return text
    compact = replace(seg, text=seg.text.replace(CJK_UTTERANCE_SEPARATOR, ""))
    retry = _split_at(compact, language, limit, separator)
    return retry if retry is not None and _fits(retry, limit) else text


def _split_at(seg: TranscriptionSegment, language: str, limit: int, separator: str) -> str | None:
    position = _best_break(seg, language, limit, separator)
    if position is None:
        return None
    left, right = _line_pair(seg.text, position, separator)
    return f"{left}\n{right}" if left and right else None


def _fits(text: str, limit: int) -> bool:
    return all(len(line) <= limit for line in text.split("\n"))


def _line_pair(text: str, position: int, separator: str) -> tuple[str, str]:
    """position 에서 나눈 두 줄. 줄 끝·앞의 공백·발화 구분과 화자 구분자는 지운다."""
    left = text[:position].rstrip(_LINE_EDGE_SPACES)
    right = text[position:].lstrip(_LINE_EDGE_SPACES)
    mark = separator.strip()
    if mark:
        if left.endswith(mark):
            left = left[: -len(mark)].rstrip(_LINE_EDGE_SPACES)
        if right.startswith(mark):
            right = right[len(mark) :].lstrip(_LINE_EDGE_SPACES)
    return left, right


def _best_break(
    seg: TranscriptionSegment, language: str, limit: int, separator: str = ""
) -> int | None:
    """줄을 나눌 글자 위치. 두 줄 모두 limit 안에 드는 자리 중 부호·발화 구분 뒤를 먼저, 그다음 균형."""
    text = seg.text
    candidates = sorted(
        p
        for p in (_word_boundaries(seg) or _break_candidates(text, language))
        | _punctuation_breaks(text)
        if 0 < p < len(text) and text[p] not in _NO_LINE_START
    )

    def lengths(pos: int) -> tuple[int, int]:
        left, right = _line_pair(text, pos, separator)
        return len(left), len(right)

    fitting = [p for p in candidates if max(lengths(p)) <= limit]
    if not fitting and language == "ja":
        # 문절 하나가 길어 어느 문절 경계로도 두 줄에 들어가지 않으면 형태소 경계까지 본다
        from src.postprocess.word_units import japanese_morpheme_boundaries

        finer = sorted(japanese_morpheme_boundaries(text) - set(candidates))
        fitting = [p for p in finer if max(lengths(p)) <= limit]
        if not fitting:
            # 그래도 들어가지 않으면(발화 구분 전각 공백도 한 글자다) 형태소 경계까지 넣어 가장 덜
            # 넘치는 자리. 문절 경계만 보면 긴 복합 명사 앞뒤로 21자 / 9자가 됐다 (15 / 17 이 낫다)
            candidates = sorted({*candidates, *(p for p in finer if text[p] not in _NO_LINE_START)})
    preferred = [p for p in fitting if text[p - 1] in _PREFERRED_BREAK_AFTER]
    pool = preferred or fitting or candidates
    if not pool:
        return None
    return min(pool, key=lambda p: (max(lengths(p)), abs(lengths(p)[0] - lengths(p)[1])))


def _word_boundaries(seg: TranscriptionSegment) -> set[int] | None:
    """자막 단어(일본어는 문절) 사이의 글자 위치. 텍스트를 단어로 다시 맞출 수 없으면 None."""
    if len(seg.words) < 2:
        return None
    text = seg.text.lower()
    positions: set[int] = set()
    i = 0
    for word in seg.words[:-1]:
        token = word.word.strip().lower()
        while i < len(text) and text[i] in _LINE_EDGE_SPACES:
            i += 1
        if not token or text[i : i + len(token)] != token:
            return None
        i += len(token)
        positions.add(i)
    return positions


def _punctuation_breaks(text: str) -> set[int]:
    """문장 부호·발화 구분 바로 뒤."""
    return {i + 1 for i, ch in enumerate(text) if ch in _PREFERRED_BREAK_AFTER}


def _break_candidates(text: str, language: str) -> set[int]:
    """단어를 쓸 수 없을 때 줄을 나눌 수 있는 글자 위치 (그 앞에서 끊는다). 양 끝은 뺀다."""
    from src.postprocess.word_units import japanese_phrase_boundaries

    positions = _punctuation_breaks(text)
    if language == "ja":
        positions |= japanese_phrase_boundaries(text) or set()
    elif not positions and language == "zh":
        # 중국어는 단어 묶기가 없어 어느 글자 사이든 나눌 수 있다
        positions = set(range(1, len(text)))
    return {p for p in positions if 0 < p < len(text)}
