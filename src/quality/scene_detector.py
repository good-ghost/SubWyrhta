"""장면 감지 모듈.

시간 갭 기반 장면 전환 감지와 PySceneDetect 연동(선택)을 제공한다.
장면 경계에서 자막 갭을 보장하여 장면 전환 시 자막이 겹치지 않도록 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.models.segment import TranscriptionSegment

log = structlog.get_logger(__name__)

# 기본 설정
DEFAULT_SCENE_GAP_S = 6.0  # 이 이상의 묵음 갭이면 장면 전환으로 간주
DEFAULT_MIN_SCENE_SEGMENTS = 2  # 최소 장면 크기 (세그먼트 수)
DEFAULT_SCENE_BOUNDARY_GAP_S = 0.5  # 장면 경계에서 보장할 자막 갭


def is_pyscenedetect_available() -> bool:
    """PySceneDetect 패키지 사용 가능 여부."""
    try:
        from scenedetect import ContentDetector, detect  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class SceneBoundary:
    """장면 경계 정보.

    Attributes:
        time_s: 장면 전환 시간(초).
        source: 감지 방식 ('gap' 또는 'pyscenedetect').
    """

    time_s: float
    source: str = "gap"


@dataclass
class Scene:
    """장면 정보.

    Attributes:
        index: 장면 인덱스 (0-based).
        start_s: 장면 시작 시간(초).
        end_s: 장면 종료 시간(초).
        segment_count: 포함된 세그먼트 수.
    """

    index: int
    start_s: float
    end_s: float
    segment_count: int = 0

    @property
    def duration_s(self) -> float:
        """장면 길이(초)."""
        return self.end_s - self.start_s


# ──────────────────────────────────────────────
# 시간 갭 기반 장면 감지
# ──────────────────────────────────────────────


def detect_scenes_by_gap(
    segments: list[TranscriptionSegment],
    gap_threshold_s: float = DEFAULT_SCENE_GAP_S,
    min_scene_segments: int = DEFAULT_MIN_SCENE_SEGMENTS,
) -> list[SceneBoundary]:
    """인접 세그먼트 간 시간 갭으로 장면 전환을 감지한다.

    Args:
        segments: 자막 세그먼트 목록.
        gap_threshold_s: 장면 전환으로 간주할 최소 갭(초).
        min_scene_segments: 최소 장면 크기. 이보다 작은 장면은 병합된다.

    Returns:
        감지된 장면 경계 목록.
    """
    if len(segments) < 2:
        return []

    # 1. 갭 기반 경계 감지
    raw_boundaries: list[SceneBoundary] = []
    for i in range(1, len(segments)):
        gap = segments[i].start - segments[i - 1].end
        if gap >= gap_threshold_s:
            boundary_time = (segments[i - 1].end + segments[i].start) / 2
            raw_boundaries.append(SceneBoundary(time_s=boundary_time, source="gap"))

    if not raw_boundaries:
        return []

    # 2. 최소 장면 크기 제약 → 자동 병합
    boundaries = _merge_small_scenes(
        segments,
        raw_boundaries,
        min_scene_segments,
    )

    if boundaries:
        log.info(
            "장면 감지 완료 (갭 기반)",
            boundaries=len(boundaries),
            gap_threshold_s=gap_threshold_s,
        )

    return boundaries


# ──────────────────────────────────────────────
# PySceneDetect 연동
# ──────────────────────────────────────────────


def detect_scenes_by_video(
    video_path: str,
    gap_threshold_s: float = DEFAULT_SCENE_GAP_S,
) -> list[SceneBoundary]:
    """PySceneDetect로 영상 기반 장면 전환을 감지한다.

    Args:
        video_path: 영상 파일 경로.
        gap_threshold_s: 장면 전환 최소 갭(사용되지 않음, 호환용).

    Returns:
        감지된 장면 경계 목록. PySceneDetect 미설치 시 빈 리스트.
    """
    if not is_pyscenedetect_available():
        log.debug("PySceneDetect 미설치, 영상 기반 장면 감지 스킵")
        return []

    try:
        from scenedetect import ContentDetector, detect

        scene_list = detect(video_path, ContentDetector())

        boundaries: list[SceneBoundary] = []
        for _start, end in scene_list:
            boundaries.append(
                SceneBoundary(
                    time_s=end.get_seconds(),
                    source="pyscenedetect",
                )
            )

        if boundaries:
            log.info(
                "장면 감지 완료 (PySceneDetect)",
                boundaries=len(boundaries),
            )

        return boundaries

    except Exception as e:
        log.warning("PySceneDetect 장면 감지 실패", error=str(e))
        return []


# ──────────────────────────────────────────────
# 장면 경계 자막 갭 보장
# ──────────────────────────────────────────────


def ensure_scene_boundary_gaps(
    segments: list[TranscriptionSegment],
    boundaries: list[SceneBoundary],
    boundary_gap_s: float = DEFAULT_SCENE_BOUNDARY_GAP_S,
) -> list[TranscriptionSegment]:
    """장면 경계에서 자막 갭을 보장한다.

    장면 경계 부근의 세그먼트 end/start를 조정하여
    장면 전환 시 자막이 겹치지 않도록 한다.

    Args:
        segments: 자막 세그먼트 목록 (in-place 수정).
        boundaries: 장면 경계 목록.
        boundary_gap_s: 장면 경계에서 보장할 갭(초).

    Returns:
        갭 보장된 세그먼트 목록.
    """
    if not boundaries or not segments:
        return segments

    boundary_times = sorted(b.time_s for b in boundaries)
    adjusted_count = 0

    for bt in boundary_times:
        half_gap = boundary_gap_s / 2

        for i in range(len(segments)):
            seg = segments[i]

            # 장면 경계 직전 세그먼트: end가 경계를 넘으면 조정
            if seg.start < bt and seg.end > bt - half_gap:
                new_end = bt - half_gap
                if new_end > seg.start + 0.1:  # 최소 0.1초는 유지
                    seg.end = new_end
                    adjusted_count += 1

            # 장면 경계 직후 세그먼트: start가 경계에 너무 가까우면 조정
            if seg.start >= bt and seg.start < bt + half_gap:
                new_start = bt + half_gap
                if new_start < seg.end - 0.1:
                    seg.start = new_start
                    adjusted_count += 1

    if adjusted_count > 0:
        log.info(
            "장면 경계 갭 보장",
            boundaries=len(boundary_times),
            adjusted_segments=adjusted_count,
            gap_s=boundary_gap_s,
        )

    return segments


# ──────────────────────────────────────────────
# 장면 목록 생성
# ──────────────────────────────────────────────


def build_scenes(
    segments: list[TranscriptionSegment],
    boundaries: list[SceneBoundary],
) -> list[Scene]:
    """장면 경계를 기반으로 장면 목록을 생성한다.

    Args:
        segments: 자막 세그먼트 목록.
        boundaries: 장면 경계 목록.

    Returns:
        Scene 목록.
    """
    if not segments:
        return []

    boundary_times = sorted(b.time_s for b in boundaries)

    # 경계로 세그먼트를 장면별로 분할
    scenes: list[Scene] = []
    scene_start = segments[0].start
    scene_seg_count = 0
    scene_idx = 0
    boundary_idx = 0

    for seg in segments:
        # 현재 세그먼트가 다음 경계를 넘는지 확인
        if boundary_idx < len(boundary_times) and seg.start >= boundary_times[boundary_idx]:
            # 현재 장면 완료
            if scene_seg_count > 0:
                scenes.append(
                    Scene(
                        index=scene_idx,
                        start_s=scene_start,
                        end_s=boundary_times[boundary_idx],
                        segment_count=scene_seg_count,
                    )
                )
                scene_idx += 1

            scene_start = seg.start
            scene_seg_count = 0
            boundary_idx += 1

        scene_seg_count += 1

    # 마지막 장면
    if scene_seg_count > 0:
        scenes.append(
            Scene(
                index=scene_idx,
                start_s=scene_start,
                end_s=segments[-1].end,
                segment_count=scene_seg_count,
            )
        )

    return scenes


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────


def _merge_small_scenes(
    segments: list[TranscriptionSegment],
    boundaries: list[SceneBoundary],
    min_scene_segments: int,
) -> list[SceneBoundary]:
    """최소 장면 크기 미만의 장면을 인접 장면으로 병합한다."""
    if min_scene_segments <= 1 or not boundaries:
        return boundaries

    boundary_times = [b.time_s for b in boundaries]

    # 각 장면의 세그먼트 수 계산
    scene_counts: list[int] = []
    seg_idx = 0
    for bt in boundary_times:
        count = 0
        while seg_idx < len(segments) and segments[seg_idx].start < bt:
            count += 1
            seg_idx += 1
        scene_counts.append(count)
    # 마지막 장면
    scene_counts.append(len(segments) - seg_idx)

    # 너무 작은 장면의 경계 제거
    keep: list[bool] = [True] * len(boundaries)
    for i, count in enumerate(scene_counts):
        if count < min_scene_segments and i < len(keep):
            keep[i] = False

    result = [b for b, k in zip(boundaries, keep, strict=True) if k]
    removed = len(boundaries) - len(result)
    if removed > 0:
        log.debug("장면 병합", removed=removed, remaining=len(result))

    return result
