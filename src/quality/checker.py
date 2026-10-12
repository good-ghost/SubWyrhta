"""자막 품질 검사기 + 자동 교정.

줄 길이, CPS, 타임코드 역전/겹침, 초단 자막을 검사하고
auto_fix가 활성화되어 있으면 자동 교정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import structlog

from src.models.result import QualitySummary, TranscriptionResult

if TYPE_CHECKING:
    from src.core.config import QualityConfig, SegmentationConfig
    from src.models.segment import TranscriptionSegment

log = structlog.get_logger(__name__)


class Severity(Enum):
    """검사 항목 심각도."""

    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class Issue:
    """품질 검사 이슈.

    Attributes:
        segment_index: 문제가 발생한 세그먼트 인덱스.
        check: 검사 항목 이름.
        severity: 심각도.
        message: 상세 메시지.
        auto_fixed: 자동 교정 여부.
    """

    segment_index: int
    check: str
    severity: Severity
    message: str
    auto_fixed: bool = False


@dataclass
class QualityReport:
    """품질 검사 리포트.

    Attributes:
        issues: 발견된 이슈 목록.
        total_segments: 전체 세그먼트 수.
        quality_summary: 품질 요약 (신뢰도 기반).
    """

    issues: list[Issue] = field(default_factory=list)
    total_segments: int = 0
    quality_summary: QualitySummary = field(default_factory=QualitySummary)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    @property
    def auto_fixed_count(self) -> int:
        return sum(1 for i in self.issues if i.auto_fixed)


def check_quality(
    result: TranscriptionResult,
    quality_config: QualityConfig,
    seg_config: SegmentationConfig,
) -> QualityReport:
    """자막 품질을 검사하고, auto_fix 시 자동 교정한다.

    Args:
        result: TranscriptionResult 인스턴스 (in-place 교정됨).
        quality_config: 품질 관리 설정.
        seg_config: 세그멘테이션 설정 (CPS/줄길이 기준 참조).

    Returns:
        QualityReport.
    """
    if not quality_config.enabled:
        return QualityReport(total_segments=result.segment_count)

    from src.segmentation.speed_optimizer import get_chars_per_line, get_cps_for_language

    report = QualityReport(total_segments=result.segment_count)
    auto_fix = quality_config.auto_fix

    # 1. 타임코드 역전 검사
    _check_timecode_inversion(result.segments, report, auto_fix)

    # 2. 타임코드 겹침 검사
    _check_timecode_overlap(result.segments, report, auto_fix, seg_config.min_gap_s)

    # 3. 줄 길이 초과 검사
    _check_line_length(
        result.segments,
        report,
        auto_fix,
        get_chars_per_line(result.metadata.language, seg_config),
    )

    # 4. CPS 초과 검사
    max_cps = get_cps_for_language(result.metadata.language, seg_config)
    _check_cps(result.segments, report, max_cps)

    # 5. 초단 자막 검사
    _check_short_duration(result.segments, report, seg_config.min_duration_s)

    # 6. 빈 자막 검사
    _check_empty_segments(result, report, auto_fix)

    # 7. 신뢰도 기반 품질 요약
    report.quality_summary = _compute_quality_summary(result.segments)
    result.quality = report.quality_summary

    log.info(
        "품질 검사 완료",
        total=report.total_segments,
        errors=report.error_count,
        warnings=report.warning_count,
        auto_fixed=report.auto_fixed_count,
        grade=report.quality_summary.grade,
    )

    return report


# ──────────────────────────────────────────────
# 개별 검사 항목
# ──────────────────────────────────────────────


def _check_timecode_inversion(
    segments: list[TranscriptionSegment],
    report: QualityReport,
    auto_fix: bool,
) -> None:
    """타임코드 역전 검사 (end < start)."""
    for seg in segments:
        if seg.end < seg.start:
            fixed = False
            if auto_fix:
                seg.start, seg.end = seg.end, seg.start
                fixed = True
            report.issues.append(
                Issue(
                    segment_index=seg.index,
                    check="timecode_inversion",
                    severity=Severity.ERROR,
                    message=f"end({seg.end:.3f}) < start({seg.start:.3f})",
                    auto_fixed=fixed,
                )
            )


def _check_timecode_overlap(
    segments: list[TranscriptionSegment],
    report: QualityReport,
    auto_fix: bool,
    min_gap_s: float,
) -> None:
    """타임코드 겹침 검사 (연속 자막 start/end 겹침)."""
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        if curr.start < prev.end:
            overlap = prev.end - curr.start
            fixed = False
            if auto_fix:
                # prev.end를 curr.start - min_gap_s로 조정
                prev.end = curr.start - min_gap_s
                if prev.end < prev.start:
                    prev.end = prev.start + 0.01
                fixed = True
            report.issues.append(
                Issue(
                    segment_index=curr.index,
                    check="timecode_overlap",
                    severity=Severity.WARNING,
                    message=f"겹침 {overlap:.3f}s (seg {prev.index} end={prev.end:.3f})",
                    auto_fixed=fixed,
                )
            )


def _check_line_length(
    segments: list[TranscriptionSegment],
    report: QualityReport,
    auto_fix: bool,
    max_chars_per_line: int,
) -> None:
    """줄 길이 초과 검사 + 자동 줄바꿈."""
    for seg in segments:
        lines = seg.text.split("\n")
        has_violation = any(len(line) > max_chars_per_line for line in lines)

        if has_violation:
            fixed = False
            if auto_fix:
                seg.text = _auto_line_break(seg.text, max_chars_per_line)
                fixed = True
            report.issues.append(
                Issue(
                    segment_index=seg.index,
                    check="line_length",
                    severity=Severity.ERROR,
                    message=f"줄 길이 초과 (max={max_chars_per_line})",
                    auto_fixed=fixed,
                )
            )


def _check_cps(
    segments: list[TranscriptionSegment],
    report: QualityReport,
    max_cps: float,
) -> None:
    """CPS 초과 검사 (재분할은 별도 파이프라인에서 수행)."""
    for seg in segments:
        if seg.cps > max_cps:
            report.issues.append(
                Issue(
                    segment_index=seg.index,
                    check="cps_exceeded",
                    severity=Severity.WARNING,
                    message=f"CPS {seg.cps:.1f} > {max_cps:.1f}",
                )
            )


def _check_short_duration(
    segments: list[TranscriptionSegment],
    report: QualityReport,
    min_duration_s: float,
) -> None:
    """초단 자막 검사 (병합은 별도 파이프라인에서 수행)."""
    for seg in segments:
        if seg.duration < min_duration_s:
            report.issues.append(
                Issue(
                    segment_index=seg.index,
                    check="short_duration",
                    severity=Severity.WARNING,
                    message=f"duration {seg.duration:.3f}s < {min_duration_s:.1f}s",
                )
            )


def _check_empty_segments(
    result: TranscriptionResult,
    report: QualityReport,
    auto_fix: bool,
) -> None:
    """빈 자막 검사 + 자동 제거."""
    empty_indices: list[int] = []

    for i, seg in enumerate(result.segments):
        if not seg.text.strip():
            empty_indices.append(i)
            report.issues.append(
                Issue(
                    segment_index=seg.index,
                    check="empty_segment",
                    severity=Severity.ERROR,
                    message="텍스트 없음",
                    auto_fixed=auto_fix,
                )
            )

    if auto_fix and empty_indices:
        result.segments = [seg for i, seg in enumerate(result.segments) if i not in empty_indices]
        result.reindex()


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────


def _auto_line_break(text: str, max_chars: int) -> str:
    """텍스트를 max_chars 이하로 자동 줄바꿈한다.

    가장 가까운 공백 위치에서 줄바꿈을 삽입한다.
    """
    if len(text) <= max_chars:
        return text

    lines: list[str] = []
    current_line = ""

    for word in text.split():
        if current_line and len(current_line) + 1 + len(word) > max_chars:
            lines.append(current_line)
            current_line = word
        else:
            current_line = f"{current_line} {word}".strip()

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def _compute_quality_summary(segments: list[TranscriptionSegment]) -> QualitySummary:
    """신뢰도 기반 품질 요약을 계산한다."""
    if not segments:
        return QualitySummary()

    confidences = [s.confidence for s in segments if s.confidence > 0]
    if not confidences:
        return QualitySummary()

    avg_conf = sum(confidences) / len(confidences)
    low_count = sum(1 for c in confidences if c < 0.7)

    # 추정 WER: 단순 추정 (1 - avg_confidence)
    estimated_wer = max(0.0, 1.0 - avg_conf)

    # 등급 부여
    if avg_conf >= 0.9:
        grade = "A"
    elif avg_conf >= 0.8:
        grade = "B"
    elif avg_conf >= 0.7:
        grade = "C"
    else:
        grade = "D"

    return QualitySummary(
        avg_confidence=avg_conf,
        low_confidence_count=low_count,
        estimated_wer=estimated_wer,
        grade=grade,
    )
