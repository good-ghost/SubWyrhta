"""LLM 텍스트 단계 공용: 묶음 호출, 응답 해석, 부호만 더했는지 검증.

LLM 부호 넣기(``llm_punctuation``, 세그멘테이션 전)와 LLM 자막 검토(``llm_review``, 줄바꿈 뒤)가 쓴다.
둘 다 글자를 바꾸지 않고 문장 부호만 더하게 하고, 코드가 그것을 확인한 뒤에 받는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import LLMSegmentationError
from src.core.text import CJK_UTTERANCE_SEPARATOR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.config import LLMConfig
    from src.segmentation.llm_client import LLMProvider

log = structlog.get_logger(__name__)

# 한 번에 보내는 줄 수 (자막 300개면 15회 호출)
WINDOW_SIZE = 20

# 검수 메모·교정 이유를 쓰는 언어 (프롬프트에 넣는 이름). 자막 언어가 아니라 사람이 리포트를 읽는 화면 언어다
# (2026-10-12 사용자 요청: 화면은 한국어, 영상은 일본어인데 메모가 대부분 일본어였다). 다른 언어는 필요할 때 더한다
NOTE_LANGUAGE_NAMES: dict[str, str] = {
    "ko": "Korean (한국어)",
    "en": "English",
    "ja": "Japanese (日本語)",
    "zh": "Chinese (中文)",
}
DEFAULT_NOTE_LANGUAGE = "ko"
# 메모 언어를 모델이 자막 언어로 따라가지 않게 프롬프트 끝에 따로 적는다 ({field}: 메모가 들어가는 JSON 키)
_NOTE_LANGUAGE_RULE = (
    'Language of "{field}": always write "{field}" in {name}, whatever the language of the '
    "{source}. The person who reads it reads {name}, not necessarily the language of the {source}. "
    "Quote words from the {source} exactly as they are, but write the explanation around them "
    "in {name}.\n"
)


# 프롬프트의 규칙만으로는 모델이 메모를 자막 언어로 쓰는 일이 남는다 (glm-5.3-flash: 묶음에 따라 11개 중 8개가
# 일본어, 2026-10-12 측정). 언어가 다른 메모만 모아 한 번 더 보내 옮긴다
TRANSLATE_NOTES_PROMPT = """\
You translate short review notes about subtitles into {name}. A note explains a possible problem \
in a subtitle and may quote subtitle words inside quotation marks (「」, 『』, '', ""). Translate only \
the explanation into {name}. Copy every quoted part, including its quotation marks, exactly as it \
is: never translate, transliterate or correct quoted text. Do not add or remove information.
Reply with JSON only: {{"notes": [{{"id": <id>, "text": "<the note in {name}>"}}]}}.
"""
# 인용: 낫표, 곧은·굽은 따옴표 (굽은 따옴표는 헷갈리는 글자라 코드로 적는다)
_QUOTED = re.compile(
    r"「[^」]*」|『[^』]*』|'[^']*'|\"[^\"]*\"|\u2018[^\u2019]*\u2019|\u201c[^\u201d]*\u201d"
)
_HANGUL = re.compile(r"[가-힣]")
_KANA = re.compile(r"[ぁ-ゖァ-ヺ]")
_HAN = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def is_written_in(text: str, language: str) -> bool:
    """글이 그 언어로 쓰였는지 글자 종류로 본다 (따옴표 안의 인용은 빼고 센다).

    한국어는 한글, 일본어는 가나가 하나라도 있으면 그 언어로 본다 (인용하지 않은 자막 낱말이 섞여도 된다).
    판단할 수 없는 언어 코드는 맞는 것으로 본다 (옮기지 않는다).

    Args:
        text: 메모.
        language: 언어 코드.
    """
    body = _QUOTED.sub("", text)
    hangul, kana = len(_HANGUL.findall(body)), len(_KANA.findall(body))
    han, latin = len(_HAN.findall(body)), len(_LATIN.findall(body))
    if language == "ko":
        return hangul > 0
    if language == "ja":
        return kana > 0
    if language == "zh":
        return han > 0 and kana == 0 and hangul == 0
    if language == "en":
        return latin > 0 and latin >= hangul + kana + han
    return True


def localize_notes(
    notes: Sequence[str],
    note_language: str,
    llm_config: LLMConfig,
    stats: LLMTextStats,
    provider: LLMProvider | None = None,
    known: dict[str, dict[str, Any]] | None = None,
    breaker: LLMBreaker | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """메모 언어가 아닌 메모를 LLM 으로 메모 언어로 옮긴다.

    인용한 자막 낱말이 그대로 남고 옮긴 글이 메모 언어일 때만 받는다. 아니면 원래 메모를 둔다
    (메모를 잃지 않는다). 옮길 메모가 없으면 호출하지 않는다.

    Args:
        notes: 메모 목록.
        note_language: 메모 언어 코드.
        llm_config: LLM 프로바이더 설정.
        stats: 결과 집계 (호출한 묶음·토큰을 더한다).
        provider: LLM 프로바이더 (테스트용).
        known: 이전에 받은 응답 (묶음 키 → 응답).
        breaker: 작업 단위 LLM 차단기.

    Returns:
        (메모 목록 (순서 그대로), 단계 캐시에 남길 응답).
    """
    localized = list(notes)
    targets = [
        at
        for at, note in enumerate(notes)
        if note.strip() and not is_written_in(note, note_language)
    ]
    if not targets:
        return localized, {}
    prompt = TRANSLATE_NOTES_PROMPT.format(name=note_language_name(note_language))
    answers, records = request_windows(
        [{"text": notes[at]} for at in targets],
        prompt,
        "notes",
        note_language,
        llm_config,
        stats,
        provider,
        known,
        breaker,
    )
    translated = 0
    for start, count, data in answers:
        for id_, text in id_text_pairs(data.get("notes")) or []:
            position = id_ - 1
            if not start <= position < start + count:
                continue
            original = notes[targets[position]]
            if _keeps_quotes(original, text) and is_written_in(text, note_language):
                localized[targets[position]] = text.strip()
                translated += 1
    log.info(
        "검수 메모 언어 맞춤",
        note_language=note_language,
        mismatched=len(targets),
        translated=translated,
    )
    return localized, records


def _keeps_quotes(original: str, translated: str) -> bool:
    """원래 메모가 인용한 자막 낱말이 옮긴 메모에 모두 그대로 있는지 (따옴표 종류는 달라도 된다)."""
    return all(quoted[1:-1] in translated for quoted in _QUOTED.findall(original))


def note_language_name(note_language: str) -> str:
    """메모 언어의 프롬프트용 이름 (모르는 코드는 기본 언어)."""
    return NOTE_LANGUAGE_NAMES.get(note_language, NOTE_LANGUAGE_NAMES[DEFAULT_NOTE_LANGUAGE])


def note_language_rule(note_language: str, field: str, source: str) -> str:
    """메모를 쓸 언어를 정하는 프롬프트 문장.

    Args:
        note_language: 메모 언어 코드 (``NOTE_LANGUAGE_NAMES`` 에 없으면 기본 언어).
        field: 메모가 들어가는 JSON 키 (``note``, ``reason``).
        source: 메모가 가리키는 글 (``subtitles``, ``lines``).
    """
    name = note_language_name(note_language)
    return _NOTE_LANGUAGE_RULE.format(field=field, name=name, source=source)


# 공백 없이 쓰는 언어 (글자 비교에서 공백·줄바꿈을 모두 뺀다)
NO_SPACE_LANGUAGES = frozenset({"ja", "zh"})
# LLM 이 더해도 되는 문장 부호. 일본어는 Whisper 출력처럼 `?` `!` 를 반각으로, 중국어는 전각으로 쓴다.
# 그 밖의 언어(en, ko 등)는 라틴 부호
_ADDABLE_PUNCTUATION: dict[str, str] = {"ja": "、。?!", "zh": "，。" + chr(0xFF1F) + chr(0xFF01)}
_ADDABLE_PUNCTUATION_DEFAULT = ",.?!"
# 이 부호 옆에 부호를 더하면 겹침으로 본다
PUNCTUATION = frozenset("、。，,.!?！？…")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class LLMBreaker:
    """작업 하나 안의 LLM 차단기 (TO-DO 20).

    LLM 호출이 재시도까지 실패하면(인증·한도·장애) 열려, 같은 작업의 이후 LLM 호출을 모두 건너뛴다.
    프로바이더가 멈추면 한 단계가 포기하기까지 ``timeout_s`` x (``max_retries`` + 1) 를 기다려(300초면
    약 20분) LLM 단계 셋에 1시간이 늘었다. 단계 캐시에 있는 응답은 호출이 없으니 계속 쓴다.
    작업(``TranscriptionPipeline.run``)마다 새로 만든다.

    Attributes:
        is_open: 열렸는가 (이후 호출을 건너뛴다).
        reason: 연 원인 (오류 메시지).
    """

    is_open: bool = False
    reason: str = ""

    def trip(self, reason: str) -> None:
        """차단기를 연다 (처음 원인만 남긴다)."""
        if not self.is_open:
            self.is_open = True
            self.reason = reason
            log.warning("LLM 호출 실패로 이 작업의 이후 LLM 호출을 건너뛴다", error=reason)


@dataclass
class LLMTextStats:
    """LLM 텍스트 단계 결과 집계.

    Attributes:
        windows: LLM 을 호출한 묶음 수.
        changed: 받아들인 수정 수.
        rejected: 검증에서 버린 수정 수.
        failed: 호출·응답 오류로 처리하지 못한 묶음 수.
        cached: 단계 캐시에서 가져온 줄 수 (호출하지 않았다).
        input_tokens: 입력 토큰 합.
        output_tokens: 출력 토큰 합.
    """

    windows: int = 0
    changed: int = 0
    rejected: int = 0
    failed: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def request_windows(
    items: Sequence[dict[str, Any]],
    system_prompt: str,
    list_key: str,
    language: str,
    llm_config: LLMConfig,
    stats: LLMTextStats,
    provider: LLMProvider | None = None,
    known: dict[str, dict[str, Any]] | None = None,
    breaker: LLMBreaker | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[list[tuple[int, int, dict[str, Any]]], dict[str, dict[str, Any]]]:
    """항목을 WINDOW_SIZE 개씩 LLM 에 보내 JSON 응답을 받는다.

    LLM 에는 ``{"language": ..., list_key: [{"id": 1, ...item}, ...]}`` 을 보낸다. 번호는 전체 기준
    (1부터)이라 묶음이 바뀌어도 LLM 이 앞뒤를 헷갈리지 않는다. 재시도까지 실패하면(인증·한도·서버 장애)
    차단기를 열고 남은 묶음은 보내지 않는다 (단계 캐시에 있는 묶음은 쓴다). 차단기가 이미 열려 있으면
    처음부터 보내지 않는다. 응답에서 JSON 객체를 읽을 수 없는 묶음은 건너뛴다.

    Args:
        items: 보낼 항목 (순서대로). 각 항목에 ``id`` 를 붙여 보낸다.
        system_prompt: 시스템 프롬프트.
        list_key: 요청 JSON 의 목록 이름 (``subtitles``, ``lines``).
        language: 언어 코드.
        llm_config: LLM 프로바이더 설정.
        stats: 결과 집계 (windows·failed·토큰을 더한다).
        provider: LLM 프로바이더 (테스트용). None 이면 설정으로 만든다.
        known: 이전에 받은 응답 (묶음 키 → 응답). 키가 같은 묶음은 보내지 않는다 (단계 캐시).
        breaker: 작업 단위 차단기. None 이면 이 호출 안에서만 차단한다.
        extra: 요청 JSON 에 함께 넣을 값 (``language`` 다음, 목록 앞). 메모 언어처럼 묶음마다 다시 알릴 것.

    Returns:
        (응답을 받은 묶음마다 (첫 항목의 0-based 위치, 항목 수, 응답 JSON 객체),
        응답을 받은 묶음의 키 → 응답). 묶음 키는 시스템 프롬프트와 요청 내용의 해시다.
    """
    from src.segmentation.llm_client import RateLimiter, complete_with_retry, create_llm_provider

    answers: list[tuple[int, int, dict[str, Any]]] = []
    records: dict[str, dict[str, Any]] = {}
    known = known or {}
    breaker = breaker if breaker is not None else LLMBreaker()
    rate_limiter = RateLimiter(llm_config.rpm)

    for start in range(0, len(items), WINDOW_SIZE):
        window = items[start : start + WINDOW_SIZE]
        user_prompt = json.dumps(
            {
                "language": language,
                **(extra or {}),
                list_key: [{"id": start + 1 + i, **item} for i, item in enumerate(window)],
            },
            ensure_ascii=False,
        )
        key = hashlib.sha256((system_prompt + user_prompt).encode("utf-8")).hexdigest()
        if key in known:
            stats.cached += len(window)
            answers.append((start, len(window), known[key]))
            records[key] = known[key]
            continue
        if breaker.is_open:
            stats.failed += 1
            continue
        try:
            if provider is None:
                provider = create_llm_provider(llm_config)
            response = complete_with_retry(
                provider,
                system_prompt,
                user_prompt,
                max_retries=llm_config.max_retries,
                rate_limiter=rate_limiter,
            )
        except LLMSegmentationError as e:
            stats.failed += 1
            breaker.trip(str(e))
            continue
        stats.windows += 1
        stats.input_tokens += response.input_tokens
        stats.output_tokens += response.output_tokens
        data = parse_json_object(response.text)
        if data is None:
            stats.failed += 1
            continue
        answers.append((start, len(window), data))
        records[key] = data
    return answers, records


def collect_changes(
    texts: Sequence[str],
    system_prompt: str,
    list_key: str,
    language: str,
    llm_config: LLMConfig,
    stats: LLMTextStats,
    provider: LLMProvider | None = None,
    breaker: LLMBreaker | None = None,
) -> tuple[dict[int, str], set[int]]:
    """텍스트를 묶음으로 보내 ``{"changes": [{"id": ..., "text": ...}]}`` 응답의 고친 텍스트를 모은다.

    Args:
        texts: 보낼 텍스트 (순서대로).
        system_prompt: 시스템 프롬프트.
        list_key: 요청 JSON 의 목록 이름.
        language: 언어 코드.
        llm_config: LLM 프로바이더 설정.
        stats: 결과 집계.
        provider: LLM 프로바이더 (테스트용).
        breaker: 작업 단위 차단기.

    Returns:
        (texts 의 0-based 위치 → LLM 이 돌려준 텍스트, 응답을 받은 위치 집합).
    """
    changes: dict[int, str] = {}
    answered: set[int] = set()
    answers, _ = request_windows(
        [{"text": t} for t in texts],
        system_prompt,
        list_key,
        language,
        llm_config,
        stats,
        provider,
        breaker=breaker,
    )
    for start, count, data in answers:
        parsed = id_text_pairs(data.get("changes"))
        if parsed is None:
            log.warning("LLM 응답에 changes 목록이 없음")
            stats.failed += 1
            continue
        answered.update(range(start, start + count))
        for id_, text in parsed:
            if start < id_ <= start + count:
                changes[id_ - 1] = text
    return changes, answered


def parse_json_object(text: str) -> dict[str, Any] | None:
    """LLM 응답에서 JSON 객체 (코드 블록으로 감싸도 된다). 읽을 수 없으면 None."""
    match = _JSON_OBJECT.search(text)
    if match is None:
        log.warning("LLM 응답에 JSON 이 없음", response=text[:200])
        return None
    try:
        data: Any = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        log.warning("LLM 응답을 읽을 수 없음", error=str(e), response=text[:200])
        return None
    return data if isinstance(data, dict) else None


def id_text_pairs(items: Any) -> list[tuple[int, str]] | None:
    """``[{"id": int, "text": str}, ...]`` 에서 (번호, 텍스트) 목록. 목록이 아니면 None (모양이 다른 항목은 뺀다)."""
    if not isinstance(items, list):
        return None
    return [
        (item["id"], item["text"])
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("id"), int)
        and isinstance(item.get("text"), str)
    ]


def parse_changes(text: str) -> list[tuple[int, str]] | None:
    """LLM 응답에서 (번호, 텍스트) 목록. JSON 이 아니거나 모양이 다르면 None.

    Args:
        text: LLM 응답 (코드 블록으로 감싸도 된다).

    Returns:
        (번호, 텍스트) 목록. 읽을 수 없으면 None.
    """
    data = parse_json_object(text)
    if data is None:
        return None
    return id_text_pairs(data.get("changes"))


def addable_punctuation(language: str) -> str:
    """LLM 이 더해도 되는 문장 부호."""
    return _ADDABLE_PUNCTUATION.get(language, _ADDABLE_PUNCTUATION_DEFAULT)


def comparable(text: str, language: str) -> str:
    """글자 비교용 텍스트: 공백 없는 언어는 공백·줄바꿈·발화 구분을 모두 빼고, 그 밖은 공백 하나로."""
    text = text.replace("\n", " ").replace(CJK_UTTERANCE_SEPARATOR, " ")
    if language in NO_SPACE_LANGUAGES:
        return "".join(text.split())
    return " ".join(text.split())


def added_punctuation(original: str, proposed: str, language: str) -> list[tuple[int, str]] | None:
    """proposed 가 original 에 더한 부호 목록 [(original 의 몇 글자 뒤, 부호)]. 부호만 더한 게 아니면 None.

    Args:
        original: 원문 (``comparable`` 을 거친 값).
        proposed: LLM 이 돌려준 텍스트 (``comparable`` 을 거친 값).
        language: 언어 코드.

    Returns:
        더한 부호 목록. 글자를 바꿨거나, 더한 부호가 다른 부호와 붙었으면 None.
    """
    addable = addable_punctuation(language)
    added: list[tuple[int, str]] = []
    i = 0
    for j, ch in enumerate(proposed):
        if i < len(original) and ch == original[i]:
            i += 1
            continue
        if ch not in addable:
            return None
        neighbours = proposed[j - 1 : j] + proposed[j + 1 : j + 2]
        if any(n in PUNCTUATION for n in neighbours):
            return None
        added.append((i, ch))
    return added if i == len(original) else None
