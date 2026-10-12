"""자막 텍스트 연결.

STT 단어를 이어 자막 텍스트를 만들 때 쓴다. 일본어·중국어는 WhisperX 가 글자 단위로 정렬해
"단어" 하나가 한 글자다. 공백으로 이으면 `も し も し` 처럼 글자마다 공백이 들어가고 글자 수(CPS)도
커지므로, 한자·가나·전각 문자와 맞닿는 곳에는 공백을 넣지 않는다. 한글은 띄어쓰기를 하므로 공백을 둔다.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# 공백 없이 잇는 문자: CJK 기호·구두점, 히라가나, 가타카나, 가타카나 확장, 한자(확장 A·통합·호환), 전각·반각 형태
_NO_SPACE_CHAR = re.compile(
    r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff"
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)


# 앞 단어에 붙는 라틴 문장 부호 (앞에 공백을 두지 않는다): `crying .` → `crying.`
_ATTACH_TO_PREVIOUS = frozenset(".,!?;:%)]}'\u2019")
# 뒤 단어에 붙는 문자 (뒤에 공백을 두지 않는다): 여는 괄호, 축약형의 아포스트로피 (`can ' t` → `can't`)
_ATTACH_TO_NEXT = frozenset("([{'\u2018\u2019")


def _needs_space(left: str, right: str) -> bool:
    """두 조각 사이에 공백이 필요한지 판단한다.

    맞닿는 글자가 공백 없이 쓰는 문자(한자·가나·전각)이거나, 오른쪽이 문장 부호·아포스트로피로
    시작하거나, 왼쪽이 여는 괄호·아포스트로피로 끝나면 불필요하다.
    """
    if _NO_SPACE_CHAR.match(left[-1]) or _NO_SPACE_CHAR.match(right[0]):
        return False
    return right[0] not in _ATTACH_TO_PREVIOUS and left[-1] not in _ATTACH_TO_NEXT


def join_words(parts: Iterable[str]) -> str:
    """단어(또는 자막 텍스트) 조각을 언어에 맞게 잇는다.

    Args:
        parts: 이을 조각. 앞뒤 공백은 지우고, 빈 조각은 건너뛴다.

    Returns:
        이은 텍스트. 한자·가나·전각 문자와 맞닿는 곳은 공백 없이, 그 밖은 공백 하나로 잇는다.
    """
    text = ""
    for raw in parts:
        part = raw.strip()
        if not part:
            continue
        if text and _needs_space(text, part):
            text += " "
        text += part
    return text


# 일본어·중국어에서 서로 다른 발화 사이에 넣는 구분 (전각 공백)
CJK_UTTERANCE_SEPARATOR = chr(0x3000)
# 이 문자로 끝나면 발화 경계가 이미 드러나 구분을 넣지 않는다
_SENTENCE_END = re.compile(r"[。\uff0e！？!?…]$")


def join_utterances(utterances: Iterable[str]) -> str:
    """서로 다른 발화(각각 이은 텍스트)를 하나로 잇는다.

    한자·가나끼리 맞닿으면 전각 공백을 넣는다: 공백 없이 이으면 `初めてですあ、はい` 처럼 두 발화가
    한 문장으로 읽힌다. 앞 발화가 문장 부호로 끝나면 넣지 않는다. 그 밖은 ``join_words`` 와 같다.

    Args:
        utterances: 발화 텍스트 목록.

    Returns:
        이은 텍스트.
    """
    text = ""
    for raw in utterances:
        part = raw.strip()
        if not part:
            continue
        if text:
            if _needs_space(text, part):
                text += " "
            elif not _SENTENCE_END.search(text):
                text += CJK_UTTERANCE_SEPARATOR
        text += part
    return text
