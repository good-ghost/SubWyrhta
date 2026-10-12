"""LLM 오인식 교정의 용어 목록 (인물 이름·작품 용어).

용어 목록의 위치는 환경 변수(.env) ``SUBWYRHTA_GLOSSARY`` 로 정한다 (2026-10-11 사용자 결정, WI-9.018).
파일이면 그 파일을, 폴더면 ``common.txt`` 와 ``<입력 파일 이름(확장자 제외)>.txt`` 를 읽는다.
한 줄에 용어 하나, 빈 줄과 ``#`` 으로 시작하는 줄은 건너뛴다.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

GLOSSARY_ENV = "SUBWYRHTA_GLOSSARY"
_COMMON_FILE = "common.txt"
# 프롬프트가 지나치게 길어지지 않게 (용어 하나 수십 토큰)
MAX_TERMS = 200


def load_glossary(source: str | Path | None = None) -> list[str]:
    """환경 변수가 가리키는 용어 목록을 읽는다. 설정하지 않았거나 없으면 빈 목록.

    Args:
        source: 입력 파일 (폴더 방식에서 ``<이름>.txt`` 를 찾는다).

    Returns:
        용어 목록 (중복 제거, 순서 유지, 최대 MAX_TERMS 개).
    """
    location = os.environ.get(GLOSSARY_ENV, "").strip()
    if not location:
        return []
    path = Path(location).expanduser()
    files: list[Path]
    if path.is_dir():
        files = [path / _COMMON_FILE]
        if source is not None:
            files.append(path / f"{Path(source).stem}.txt")
    else:
        files = [path]
    terms: list[str] = []
    for file in files:
        if not file.is_file():
            continue
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            log.warning("용어 목록을 읽을 수 없음", path=str(file), error=str(e))
            continue
        terms.extend(
            line.strip() for line in lines if line.strip() and not line.strip().startswith("#")
        )
    unique = list(dict.fromkeys(terms))
    if len(unique) > MAX_TERMS:
        log.warning("용어가 많아 앞부분만 쓴다", terms=len(unique), max_terms=MAX_TERMS)
    if unique:
        log.info("용어 목록 읽음", terms=min(len(unique), MAX_TERMS))
    return unique[:MAX_TERMS]
