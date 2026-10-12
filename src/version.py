"""SubWyrhta 버전.

버전의 기준은 ``pyproject.toml`` 의 ``[project] version`` 하나다. 소스 체크아웃에서는 그 파일을 읽고,
패키지로 설치해 파일이 없으면 설치된 패키지 메타데이터를 쓴다. 설치 메타데이터(egg-info)는
다시 설치하기 전까지 갱신되지 않아 pyproject 를 먼저 본다.
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGE_NAME = "subwyrhta"
_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
# 버전을 알 수 없을 때 (pyproject 도 패키지 메타데이터도 없음)
UNKNOWN_VERSION = "0+unknown"


def _pyproject_version() -> str | None:
    """소스 체크아웃의 pyproject.toml 버전. 설치본처럼 파일이 없거나 읽을 수 없으면 None."""
    try:
        with _PYPROJECT.open("rb") as f:
            project = tomllib.load(f).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return None
    value = project.get("version")
    return value if isinstance(value, str) and value else None


def _read_version() -> str:
    from_pyproject = _pyproject_version()
    if from_pyproject is not None:
        return from_pyproject
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return UNKNOWN_VERSION


__version__ = _read_version()
