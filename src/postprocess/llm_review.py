"""LLM 줄바꿈 수정 검증: LLM 이 옮긴 줄바꿈을 받아도 되는지 확인한다.

LLM 마지막 검토(``llm_finalize``)의 줄바꿈 항목이 쓴다. 처음(9.14)에는 이 모듈이 줄바꿈·부호 검토 단계였고
(9.16 에서 줄바꿈 전용), 9.17 에서 마지막 검토가 흡수했다.

받는 조건:
- 줄바꿈 위치 말고는 바뀐 것이 없다 (글자·부호 그대로. 공백·발화 구분은 비교에서 뺀다)
- 줄 수와 줄당 글자 수 제한 안이다
지시와 달리 자막 끝에 붙인 `。` `、` 와 공백 없는 언어에 넣은 반각 공백은 떼고 비교한다.
"""

from __future__ import annotations

from src.postprocess.llm_text import NO_SPACE_LANGUAGES, comparable

# LLM 이 지시와 달리 자막 끝에 붙인 부호. 떼고 나서 비교한다 (줄바꿈만 바꾼 응답을 살린다)
_TRAILING_DROP = frozenset("、。，,.")


def tidy_revision(original: str, proposed: str, language: str) -> str:
    """LLM 수정에서 받지 않을 꾸밈을 뗀다: 자막 끝에 새로 붙인 `。` `、`, 공백 없는 언어에 넣은 반각 공백.

    Args:
        original: 규칙으로 만든 자막 텍스트.
        proposed: LLM 이 돌려준 텍스트.
        language: 언어 코드.

    Returns:
        정리한 텍스트.
    """
    text = proposed.strip()
    if language in NO_SPACE_LANGUAGES and " " not in original:
        text = text.replace(" ", "")
    original_end = original.rstrip()[-1:]
    while text and text[-1] in _TRAILING_DROP and text[-1] != original_end:
        text = text[:-1]
    return text


def is_acceptable_revision(
    original: str, proposed: str, language: str, chars_per_line: int, max_lines: int
) -> bool:
    """LLM 이 고친 자막을 받아도 되는지 (줄바꿈 위치만 바뀌었고 줄 제한 안).

    Args:
        original: 규칙으로 만든 자막 텍스트.
        proposed: LLM 이 돌려준 텍스트.
        language: 언어 코드.
        chars_per_line: 줄당 최대 글자 수.
        max_lines: 최대 줄 수.

    Returns:
        받아도 되면 True.
    """
    lines = [line.strip() for line in proposed.strip().split("\n")]
    if not all(lines) or len(lines) > max_lines:
        return False
    if any(len(line) > chars_per_line for line in lines):
        return False
    return comparable(original, language) == comparable(" ".join(lines), language)
