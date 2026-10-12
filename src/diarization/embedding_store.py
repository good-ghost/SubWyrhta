"""화자 임베딩 저장/재사용.

화자 분리 결과의 임베딩을 파일로 내보내고,
이전 프로파일의 임베딩을 재사용할 수 있게 한다.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# 임베딩 파일 포맷 버전
_FORMAT_VERSION = "1.0"


def export_embeddings(
    embeddings: Any,
    speakers: list[str],
    output_path: str | Path,
    source_file: str = "",
) -> Path:
    """화자 임베딩을 파일로 내보낸다.

    Args:
        embeddings: 화자 임베딩 데이터 (numpy 배열 또는 tensor).
        speakers: 화자 ID 목록 (임베딩 순서와 일치).
        output_path: 출력 파일 경로.
        source_file: 원본 오디오 파일 경로 (메타데이터).

    Returns:
        저장된 파일의 Path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_file": source_file,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "embeddings": _serialize_embeddings(embeddings, speakers),
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("화자 임베딩 내보내기 완료", path=str(path), speakers=len(speakers))
    return path


def load_embeddings(embedding_path: str | Path) -> dict[str, Any]:
    """화자 임베딩 파일을 로드한다.

    Args:
        embedding_path: 임베딩 파일 경로.

    Returns:
        임베딩 데이터 딕셔너리::

            {
                "speakers": ["SPEAKER_00", ...],
                "embeddings": {speaker_id: list[float], ...},
                "source_file": "...",
            }

    Raises:
        FileNotFoundError: 파일이 없을 때.
        ValueError: 파일 형식이 잘못되었을 때.
    """
    path = Path(embedding_path)
    if not path.exists():
        raise FileNotFoundError(f"임베딩 파일이 존재하지 않습니다: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"임베딩 파일 파싱 실패: {path} ({e})") from e

    if not isinstance(data, dict) or "speakers" not in data:
        raise ValueError(f"잘못된 임베딩 파일 형식: {path}")

    log.info(
        "화자 임베딩 로드 완료",
        path=str(path),
        speakers=len(data.get("speakers", [])),
    )
    return data


def _serialize_embeddings(
    embeddings: Any,
    speakers: list[str],
) -> dict[str, list[float]]:
    """임베딩 데이터를 JSON 직렬화 가능한 형태로 변환한다."""
    result: dict[str, list[float]] = {}

    if embeddings is None:
        return result

    try:
        import numpy as np

        if isinstance(embeddings, np.ndarray):
            for i, speaker in enumerate(speakers):
                if i < len(embeddings):
                    result[speaker] = embeddings[i].tolist()
            return result
    except ImportError:
        pass

    # dict 형태 {speaker: embedding_list}
    if isinstance(embeddings, dict):
        for speaker in speakers:
            if speaker in embeddings:
                emb = embeddings[speaker]
                if hasattr(emb, "tolist"):
                    result[speaker] = emb.tolist()
                elif isinstance(emb, list):
                    result[speaker] = emb
        return result

    return result


# ──────────────────────────────────────────────
# 임베딩 재사용: 이전 화자와 같은 화자 찾기
# ──────────────────────────────────────────────

# 같은 화자로 볼 최소 코사인 유사도. pyannote 화자 임베딩은 같은 사람끼리 대개 0.6 이상,
# 다른 사람끼리는 0.3 아래로 나온다 (재사용은 선택 기능이라 보수적으로 둔다)
SAME_SPEAKER_SIMILARITY = 0.5
_SPEAKER_LABEL_PREFIX = "SPEAKER_"


def match_known_speakers(
    current: dict[str, list[float]],
    known: dict[str, list[float]],
    min_similarity: float = SAME_SPEAKER_SIMILARITY,
) -> dict[str, str]:
    """지금 화자를 이전 파일의 화자와 짝지어 새 화자 ID 를 정한다.

    코사인 유사도가 높은 쌍부터 하나씩 짝짓는다 (한 화자는 한 번만). 짝이 없는 화자는 이전 ID 와
    겹치지 않는 새 ``SPEAKER_nn`` 을 받는다.

    Args:
        current: 지금 파일의 ``{화자 ID: 임베딩}``.
        known: 이전 파일(``--speaker-embeddings``)의 ``{화자 ID: 임베딩}``.
        min_similarity: 같은 화자로 볼 최소 유사도.

    Returns:
        ``{지금 화자 ID: 쓸 화자 ID}`` (모든 지금 화자를 포함한다).
    """
    pairs = sorted(
        (
            (_cosine(vec, known_vec), speaker, known_id)
            for speaker, vec in current.items()
            for known_id, known_vec in known.items()
            if len(vec) == len(known_vec)
        ),
        reverse=True,
    )
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for similarity, speaker, known_id in pairs:
        if similarity < min_similarity:
            break
        if speaker in mapping or known_id in used:
            continue
        mapping[speaker] = known_id
        used.add(known_id)

    taken = set(known) | used
    next_index = _next_label_index(taken)
    for speaker in sorted(current):
        if speaker in mapping:
            continue
        while f"{_SPEAKER_LABEL_PREFIX}{next_index:02d}" in taken:
            next_index += 1
        label = f"{_SPEAKER_LABEL_PREFIX}{next_index:02d}"
        mapping[speaker] = label
        taken.add(label)

    log.info(
        "이전 화자와 대조",
        matched={k: v for k, v in mapping.items() if v in known},
        new=[v for v in mapping.values() if v not in known],
    )
    return mapping


def known_embeddings(data: dict[str, Any]) -> dict[str, list[float]]:
    """``load_embeddings`` 결과에서 ``{화자 ID: 임베딩}`` 만 꺼낸다 (숫자 목록이 아닌 항목은 버린다)."""
    raw = data.get("embeddings")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, list[float]] = {}
    for speaker, vec in raw.items():
        if isinstance(vec, list) and vec and all(isinstance(x, int | float) for x in vec):
            result[str(speaker)] = [float(x) for x in vec]
    return result


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


def _next_label_index(labels: set[str]) -> int:
    """``SPEAKER_nn`` 형식 ID 중 가장 큰 번호 + 1."""
    numbers = [
        int(label[len(_SPEAKER_LABEL_PREFIX) :])
        for label in labels
        if label.startswith(_SPEAKER_LABEL_PREFIX) and label[len(_SPEAKER_LABEL_PREFIX) :].isdigit()
    ]
    return max(numbers, default=-1) + 1
