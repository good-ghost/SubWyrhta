"""완료한 작업의 자막 검수: SRT 자막과 LLM 검수 리포트(``*.review.json``)를 합쳐 보이고, 고친 자막을 SRT 에 쓴다.

검수 리포트는 자막을 바꾸지 않고 사람이 확인할 항목만 남긴다 (오인식 의심, 부호, 반영하지 않은 교정 제안).
이 모듈은 그 항목을 시각으로 자막에 붙이고, 사용자가 고친 자막 글자만 SRT 에 다시 쓴다 (번호·시각은 그대로).
저장한 자막은 검수를 마친 것으로 적어 둔다 (글자를 바꾸지 않고 저장해도 검수 완료다).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger(__name__)

SRT_SUFFIX = ".srt"
REPORT_SUFFIX = ".review.json"
# 처음 고칠 때 남기는 원본 (다시 만들려면 STT 를 다시 돌려야 한다)
BACKUP_SUFFIX = ".orig.srt"
MAX_TEXT_LENGTH = 500
# 검수를 마친 자막 번호를 적는 파일 (작업 폴더에 둔다: 출력 파일이 아니다, 작업을 지우면 함께 지워진다)
STATE_FILE = "review-state.json"
_STATE_KEY = "reviewed"

# 검수 종류: 리포트의 issues 종류 + 교정 제안(반영하지 않음)·반영한 교정
ISSUE_TYPES = ("misrecognition", "punctuation", "other")
TYPE_SUGGESTION = "suggestion"
TYPE_CORRECTED = "corrected"

_TIMING = re.compile(
    r"^(\d{1,3}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,3}):(\d{2}):(\d{2})[,.](\d{3})"
)
_TIMING_ARROW = "-->"
_BLOCK_GAP = re.compile(r"\n[ \t]*\n")
_BOM = "\ufeff"
# 요청 스레드 둘이 같은 SRT 를 함께 고치면 한쪽 수정이 사라진다 (읽기 → 쓰기 사이)
_WRITE_LOCK = threading.Lock()


class ReviewError(ValueError):
    """검수 요청을 처리할 수 없다 (``code`` 가 API 오류 코드)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Cue:
    """SRT 자막 하나. ``timing`` 은 파일의 시각 줄 그대로다 (다시 쓸 때 바꾸지 않는다)."""

    index: int
    timing: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Note:
    """검수 항목 하나 (리포트의 시각, 종류, 메모)."""

    start: float
    end: float
    type: str
    note: str


def is_subtitle(name: str) -> bool:
    """검수할 SRT 출력인지 (남겨 둔 원본은 뺀다)."""
    return name.endswith(SRT_SUFFIX) and not name.endswith(BACKUP_SUFFIX)


def backup_name(name: str) -> str:
    """SRT 이름 → 원본을 남길 이름 (``a.ja.srt`` → ``a.ja.orig.srt``)."""
    return name[: -len(SRT_SUFFIX)] + BACKUP_SUFFIX


def parse_srt(text: str) -> list[Cue]:
    """SRT 본문을 자막 목록으로 읽는다.

    Args:
        text: SRT 파일 내용.

    Returns:
        자막 목록 (파일 순서).

    Raises:
        ReviewError: ``srt_invalid`` — 시각 줄이 없는 블록이 있거나 번호가 겹친다.
    """
    body = text.replace("\r\n", "\n").replace("\r", "\n").lstrip(_BOM).strip()
    if not body:
        return []
    cues: list[Cue] = []
    seen: set[int] = set()
    for block in _BLOCK_GAP.split(body):
        lines = block.strip("\n").split("\n")
        # 번호 줄이 없는 SRT 도 있다: 시각 줄이 첫 줄이면 순서를 번호로 쓴다
        at = 0 if _TIMING.match(lines[0].strip()) else 1
        match = _TIMING.match(lines[at].strip()) if len(lines) > at else None
        number = lines[0].strip()
        if match is None or (at == 1 and not number.isdigit()):
            raise ReviewError("srt_invalid")
        index = int(number) if at == 1 else len(cues) + 1
        if index in seen:
            raise ReviewError("srt_invalid")
        seen.add(index)
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(part) for part in match.groups())
        cues.append(
            Cue(
                index=index,
                timing=lines[at].strip(),
                start=h1 * 3600 + m1 * 60 + s1 + ms1 / 1000,
                end=h2 * 3600 + m2 * 60 + s2 + ms2 / 1000,
                text="\n".join(lines[at + 1 :]),
            )
        )
    return cues


def format_srt(cues: list[Cue]) -> str:
    """자막 목록을 SRT 본문으로 (번호·시각 줄은 읽은 그대로)."""
    return "\n".join(f"{cue.index}\n{cue.timing}\n{cue.text}\n" for cue in cues)


def clean_text(text: str) -> str:
    """사용자가 고친 자막 글자를 SRT 에 넣을 수 있게 다듬고 검증한다.

    빈 줄은 SRT 에서 자막을 끝내므로 뺀다. 시각 줄처럼 보이는 글자(``-->``)는 받지 않는다.

    Raises:
        ReviewError: ``text_empty``, ``text_too_long``, ``text_invalid``.
    """
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned = "\n".join(line for line in lines if line)
    if not cleaned:
        raise ReviewError("text_empty")
    if len(cleaned) > MAX_TEXT_LENGTH:
        raise ReviewError("text_too_long")
    if _TIMING_ARROW in cleaned:
        raise ReviewError("text_invalid")
    return cleaned


def replace_text(cues: list[Cue], index: int, text: str) -> list[Cue]:
    """번호가 ``index`` 인 자막의 글자를 바꾼 새 목록.

    Raises:
        ReviewError: ``cue_not_found`` 또는 :func:`clean_text` 의 오류.
    """
    cleaned = clean_text(text)
    if all(cue.index != index for cue in cues):
        raise ReviewError("cue_not_found")
    return [replace(cue, text=cleaned) if cue.index == index else cue for cue in cues]


def read_report(path: Path | None) -> dict[str, Any]:
    """검수 리포트를 읽는다. 없거나 읽을 수 없으면 빈 리포트 (자막만 검수한다)."""
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("검수 리포트를 읽지 못함", path=str(path), error=str(e))
        return {}
    return data if isinstance(data, dict) else {}


def report_notes(report: dict[str, Any]) -> list[Note]:
    """리포트의 issues·suggested·corrected 를 검수 항목으로 편다.

    예전 리포트에는 suggested·corrected 가 없다 (없는 키는 건너뛴다).
    """
    notes: list[Note] = []
    for item in _items(report.get("issues")):
        kind = item.get("type")
        notes.append(
            _note(item, kind if kind in ISSUE_TYPES else "other", str(item.get("note", "")))
        )
    for key, kind in (("suggested", TYPE_SUGGESTION), ("corrected", TYPE_CORRECTED)):
        notes.extend(_note(item, kind, _correction_text(item)) for item in _items(report.get(key)))
    return [note for note in notes if note.note]


def attach_notes(cues: list[Cue], notes: list[Note]) -> dict[int, list[Note]]:
    """검수 항목을 자막에 붙인다: 시각이 가장 많이 겹치는 자막, 겹치지 않으면 시작이 가장 가까운 자막.

    리포트의 시각은 마지막 검토 전 자막의 것이라 최종 SRT 와 조금 다를 수 있다 (경계 옮기기·나누기).

    Returns:
        자막 번호 → 그 자막의 검수 항목 (리포트 순서).
    """
    attached: dict[int, list[Note]] = {}
    if not cues:
        return attached
    for note in notes:
        best = max(cues, key=lambda cue: (_overlap(cue, note), -abs(cue.start - note.start)))
        attached.setdefault(best.index, []).append(note)
    return attached


def save_text(path: Path, index: int, text: str) -> tuple[Cue, bool]:
    """SRT 파일에서 자막 하나의 글자를 바꿔 쓴다. 처음 고칠 때 원본을 옆에 남긴다.

    글자가 그대로면 파일을 다시 쓰지 않고 원본도 남기지 않는다 (확인만 하고 저장한 경우).

    Args:
        path: 고칠 SRT 파일.
        index: 자막 번호.
        text: 새 글자.

    Returns:
        (저장한 자막, 이번에 원본을 남겼는지).

    Raises:
        ReviewError: ``srt_invalid``, ``cue_not_found``, ``text_*``.
    """
    with _WRITE_LOCK:
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ReviewError("srt_invalid") from e
        before = parse_srt(original)
        cues = replace_text(before, index, text)
        if cues == before:
            return next(cue for cue in cues if cue.index == index), False
        backup = path.with_name(backup_name(path.name))
        backed_up = not backup.exists()
        if backed_up:
            shutil.copy2(path, backup)
        # 쓰다가 멈춰도 자막 파일이 잘리지 않게 옆에 쓰고 바꿔 놓는다
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(format_srt(cues), encoding="utf-8")
        os.replace(temporary, path)
    log.info("자막 수정", path=str(path), index=index, backed_up=backed_up)
    return next(cue for cue in cues if cue.index == index), backed_up


def read_reviewed(path: Path, name: str) -> set[int]:
    """검수를 마친 자막 번호. 상태 파일이 없거나 읽을 수 없으면 없는 것으로 본다.

    Args:
        path: 작업의 검수 상태 파일.
        name: SRT 출력 이름.
    """
    indices = _read_state(path).get(name, [])
    return {index for index in indices if isinstance(index, int)}


def mark_reviewed(path: Path, name: str, index: int) -> None:
    """자막 하나를 검수 완료로 적는다.

    Args:
        path: 작업의 검수 상태 파일.
        name: SRT 출력 이름.
        index: 자막 번호.
    """
    with _WRITE_LOCK:
        state = _read_state(path)
        reviewed = {value for value in state.get(name, []) if isinstance(value, int)}
        if index in reviewed:
            return
        state[name] = sorted(reviewed | {index})
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps({_STATE_KEY: state}), encoding="utf-8")
        os.replace(temporary, path)


def _read_state(path: Path) -> dict[str, list[Any]]:
    """상태 파일의 SRT 이름 → 번호 목록. 깨진 파일은 빈 상태로 본다 (검수 표시만 사라진다)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.warning("검수 상태 파일을 읽지 못함", path=str(path), error=str(e))
        return {}
    reviewed = data.get(_STATE_KEY) if isinstance(data, dict) else None
    if not isinstance(reviewed, dict):
        return {}
    return {name: value for name, value in reviewed.items() if isinstance(value, list)}


def _items(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _seconds(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else 0.0


def _note(item: dict[str, Any], kind: str, text: str) -> Note:
    return Note(_seconds(item.get("start")), _seconds(item.get("end")), kind, text.strip())


def _correction_text(item: dict[str, Any]) -> str:
    """교정 항목의 메모: ``바꾸기 전 → 바꾼 뒤 (이유)``."""
    change = f"{item.get('from', '')} → {item.get('to', '')}"
    reason = str(item.get("reason", "")).strip()
    return f"{change} ({reason})" if reason else change


def _overlap(cue: Cue, note: Note) -> float:
    return max(0.0, min(cue.end, note.end) - max(cue.start, note.start))
