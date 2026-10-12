"""비밀 값 저장소: 데이터 폴더의 ``secrets.json`` (권한 600).

관리 계정의 비밀번호 해시와 세션 서명 키를 둔다. 값은 로그·응답에 내보내지 않는다.
"""

from __future__ import annotations

import json
import os
import threading
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger(__name__)

SECRETS_FILE = "secrets.json"
_FILE_MODE = 0o600


class SecretStore:
    """키-값 비밀 저장소. 쓰기는 임시 파일에 쓴 뒤 rename 한다 (중간에 끊겨도 파일이 깨지지 않게).

    Args:
        path: 저장 파일 경로.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    @classmethod
    def for_data_dir(cls, data_dir: Path) -> SecretStore:
        """데이터 폴더의 기본 저장소."""
        return cls(data_dir / SECRETS_FILE)

    def get(self, key: str) -> str | None:
        """값. 없으면 None."""
        with self._lock:
            value = self._read().get(key)
        return value if isinstance(value, str) else None

    def set(self, key: str, value: str) -> None:
        """값을 저장한다."""
        with self._lock:
            data = self._read()
            data[key] = value
            self._write(data)

    def _read(self) -> dict[str, object]:
        if not self._path.exists():
            return {}
        try:
            loaded = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"비밀 저장소를 읽을 수 없습니다: {self._path} ({e})") from e
        return loaded if isinstance(loaded, dict) else {}

    def _write(self, data: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.chmod(tmp, _FILE_MODE)
        os.replace(tmp, self._path)
