"""체크포인트 저장/재개.

배치 처리 중간 결과를 .checkpoint.json 파일로 저장하고,
중단 시 이어서 처리할 수 있게 한다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_CHECKPOINT_VERSION = "1.0"
_CHECKPOINT_SUFFIX = ".checkpoint.json"


def get_checkpoint_path(source_file: str | Path) -> Path:
    """소스 파일에 대응하는 체크포인트 파일 경로를 반환한다."""
    return Path(str(source_file) + _CHECKPOINT_SUFFIX)


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """설정 딕셔너리의 SHA256 해시를 계산한다.

    설정이 변경되면 체크포인트를 무효화하기 위해 사용한다.

    Args:
        config_dict: 직렬화 가능한 설정 딕셔너리.

    Returns:
        'sha256:...' 형식의 해시 문자열.
    """
    serialized = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def save_checkpoint(
    source_file: str,
    config_hash: str,
    total_chunks: int,
    completed_chunks: list[int],
    partial_segments: list[dict[str, Any]],
    checkpoint_path: str | Path | None = None,
) -> Path:
    """체크포인트를 저장한다.

    Args:
        source_file: 원본 파일 경로.
        config_hash: 설정 해시.
        total_chunks: 전체 청크 수.
        completed_chunks: 완료된 청크 인덱스 목록.
        partial_segments: 완료된 청크의 세그먼트 데이터.
        checkpoint_path: 체크포인트 경로. None이면 자동 생성.

    Returns:
        저장된 체크포인트 파일 Path.
    """
    path = Path(checkpoint_path) if checkpoint_path else get_checkpoint_path(source_file)
    now = datetime.now(tz=UTC).isoformat()

    data: dict[str, Any] = {
        "version": _CHECKPOINT_VERSION,
        "source_file": source_file,
        "created_at": now,
        "updated_at": now,
        "config_hash": config_hash,
        "total_chunks": total_chunks,
        "completed_chunks": sorted(completed_chunks),
        "last_completed_chunk": max(completed_chunks) if completed_chunks else -1,
        "partial_segments": partial_segments,
    }

    # 기존 체크포인트가 있으면 created_at 유지
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if "created_at" in existing:
                data["created_at"] = existing["created_at"]
        except (json.JSONDecodeError, OSError):
            pass

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    log.debug(
        "체크포인트 저장",
        path=str(path),
        completed=len(completed_chunks),
        total=total_chunks,
    )
    return path


def load_checkpoint(
    source_file: str,
    config_hash: str,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """체크포인트를 로드한다.

    config_hash가 일치하지 않으면 체크포인트를 무효화한다.

    Args:
        source_file: 원본 파일 경로.
        config_hash: 현재 설정 해시.
        checkpoint_path: 체크포인트 경로. None이면 자동 생성.

    Returns:
        체크포인트 데이터 딕셔너리. 없거나 무효하면 None.
    """
    path = Path(checkpoint_path) if checkpoint_path else get_checkpoint_path(source_file)

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("체크포인트 파싱 실패", path=str(path), error=str(e))
        return None
    if not isinstance(data, dict):
        log.warning("체크포인트 형식 오류: 객체가 아님", path=str(path))
        return None

    # 버전 검증
    if data.get("version") != _CHECKPOINT_VERSION:
        log.warning(
            "체크포인트 버전 불일치, 무시",
            expected=_CHECKPOINT_VERSION,
            actual=data.get("version"),
        )
        return None

    # 설정 해시 검증
    if data.get("config_hash") != config_hash:
        log.warning("설정 변경 감지, 체크포인트 무효화", path=str(path))
        return None

    log.info(
        "체크포인트 로드",
        path=str(path),
        completed=len(data.get("completed_chunks", [])),
        total=data.get("total_chunks", 0),
    )
    return data


def delete_checkpoint(
    source_file: str,
    checkpoint_path: str | Path | None = None,
) -> bool:
    """체크포인트 파일을 삭제한다.

    Args:
        source_file: 원본 파일 경로.
        checkpoint_path: 체크포인트 경로. None이면 자동 생성.

    Returns:
        삭제 성공 여부.
    """
    path = Path(checkpoint_path) if checkpoint_path else get_checkpoint_path(source_file)

    if not path.exists():
        return False

    try:
        path.unlink()
        log.info("체크포인트 삭제", path=str(path))
        return True
    except OSError as e:
        log.warning("체크포인트 삭제 실패", path=str(path), error=str(e))
        return False


def get_remaining_chunks(
    checkpoint: dict[str, Any],
) -> list[int]:
    """미완료 청크 인덱스 목록을 반환한다.

    Args:
        checkpoint: load_checkpoint() 결과.

    Returns:
        처리가 필요한 청크 인덱스 목록.
    """
    total = checkpoint.get("total_chunks", 0)
    completed = set(checkpoint.get("completed_chunks", []))
    return [i for i in range(total) if i not in completed]
