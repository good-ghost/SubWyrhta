"""화자 이름 매핑 + 프로파일.

SPEAKER_00 등의 자동 레이블을 실제 이름으로 치환한다.
CLI 옵션 문자열, JSON 프로파일 파일, config 딕셔너리를 지원한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.models.segment import TranscriptionSegment

log = structlog.get_logger(__name__)


def parse_speaker_map_string(map_string: str) -> dict[str, str]:
    """CLI --speaker-map 옵션 문자열을 파싱한다.

    형식: ``"SPEAKER_00:John,SPEAKER_01:Jane"``

    Args:
        map_string: 쉼표로 구분된 ``ID:이름`` 쌍.

    Returns:
        ``{speaker_id: name}`` 딕셔너리.
    """
    result: dict[str, str] = {}
    if not map_string or not map_string.strip():
        return result

    for pair in map_string.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        speaker_id, name = pair.split(":", 1)
        speaker_id = speaker_id.strip()
        name = name.strip()
        if speaker_id and name:
            result[speaker_id] = name

    return result


def load_speaker_profile(profile_path: str | Path) -> dict[str, str]:
    """JSON 프로파일 파일에서 화자 매핑을 로드한다.

    JSON 형식 예시::

        {
            "speakers": {
                "SPEAKER_00": {"name": "John", "color": "#FF0000"},
                "SPEAKER_01": {"name": "Jane", "color": "#00FF00"}
            }
        }

    또는 단순 형식::

        {"SPEAKER_00": "John", "SPEAKER_01": "Jane"}

    Args:
        profile_path: JSON 프로파일 파일 경로.

    Returns:
        ``{speaker_id: name}`` 딕셔너리.
    """
    path = Path(profile_path)
    if not path.exists():
        log.warning("화자 프로파일 파일 없음", path=str(path))
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("화자 프로파일 로드 실패", path=str(path), error=str(e))
        return {}

    return _extract_names(data)


def _extract_names(data: Any) -> dict[str, str]:
    """JSON 데이터에서 화자 이름을 추출한다."""
    if not isinstance(data, dict):
        return {}

    # "speakers" 키가 있는 상세 형식
    if "speakers" in data and isinstance(data["speakers"], dict):
        result: dict[str, str] = {}
        for speaker_id, info in data["speakers"].items():
            if isinstance(info, dict) and "name" in info:
                result[str(speaker_id)] = str(info["name"])
            elif isinstance(info, str):
                result[str(speaker_id)] = info
        return result

    # 단순 형식: {"SPEAKER_00": "John"}
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[str(key)] = value

    return result


def build_speaker_map(
    config_map: dict[str, str] | None = None,
    cli_map_string: str | None = None,
    profile_path: str | Path | None = None,
) -> dict[str, str]:
    """여러 소스에서 화자 매핑을 통합한다.

    우선순위: CLI 옵션 > 프로파일 파일 > config 딕셔너리.

    Args:
        config_map: config.yaml의 speaker_map.
        cli_map_string: --speaker-map CLI 옵션 문자열.
        profile_path: JSON 프로파일 파일 경로.

    Returns:
        통합된 ``{speaker_id: name}`` 딕셔너리.
    """
    result: dict[str, str] = {}

    # 1. config (최저 우선순위)
    if config_map:
        result.update(config_map)

    # 2. 프로파일 파일
    if profile_path is not None:
        result.update(load_speaker_profile(profile_path))

    # 3. CLI 옵션 (최고 우선순위)
    if cli_map_string:
        result.update(parse_speaker_map_string(cli_map_string))

    if result:
        log.info("화자 이름 매핑 구성", mappings=result)

    return result


def apply_speaker_names(
    segments: list[TranscriptionSegment],
    speaker_map: dict[str, str],
) -> None:
    """세그먼트의 화자 ID를 실제 이름으로 치환한다 (in-place).

    Args:
        segments: 세그먼트 목록.
        speaker_map: ``{speaker_id: name}`` 딕셔너리.
    """
    if not speaker_map:
        return

    for seg in segments:
        if isinstance(seg.speaker, str) and seg.speaker in speaker_map:
            seg.speaker = speaker_map[seg.speaker]
        elif isinstance(seg.speaker, list):
            seg.speaker = [speaker_map.get(s, s) for s in seg.speaker]
