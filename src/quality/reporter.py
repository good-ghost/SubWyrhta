"""품질 리포트 생성기.

터미널 출력 및 파일 저장을 지원하는 품질 리포트를 생성한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.quality.checker import QualityReport

log = structlog.get_logger(__name__)


def format_report(
    report: QualityReport,
    source_file: str = "",
) -> str:
    """품질 리포트를 포맷된 문자열로 생성한다.

    Args:
        report: QualityReport 인스턴스.
        source_file: 원본 파일 이름 (헤더 표시용).

    Returns:
        포맷된 리포트 문자열.
    """
    lines: list[str] = []

    # 헤더
    title = source_file or "Quality Report"
    lines.append(f"[Quality Report] {title}")

    # 기본 통계
    lines.append(f"  총 자막 수:       {report.total_segments:,}")

    # 신뢰도 기반 지표
    qs = report.quality_summary
    if qs.grade:
        lines.append(f"  평균 신뢰도:      {qs.avg_confidence:.3f} ({qs.grade}등급)")
        pct = (
            (qs.low_confidence_count / report.total_segments * 100)
            if report.total_segments > 0
            else 0
        )
        lines.append(f"  저신뢰도 구간:    {qs.low_confidence_count}개 ({pct:.2f}%)")
        lines.append(f"  추정 WER:         ~{qs.estimated_wer * 100:.1f}%")

    # 이슈별 집계
    issue_counts = _count_issues_by_check(report)

    for check, (total, fixed) in issue_counts.items():
        label = _check_label(check)
        if fixed > 0:
            lines.append(f"  {label}:  {total}개 → {fixed}개 자동 교정")
        else:
            lines.append(f"  {label}:  {total}개")

    # 이슈 없음
    if not report.issues:
        lines.append("  이슈 없음")

    return "\n".join(lines)


def print_report(
    report: QualityReport,
    source_file: str = "",
) -> None:
    """품질 리포트를 터미널에 출력한다."""
    print(format_report(report, source_file))


def save_report(
    report: QualityReport,
    output_path: str | Path,
    source_file: str = "",
) -> Path:
    """품질 리포트를 파일로 저장한다.

    Args:
        report: QualityReport 인스턴스.
        output_path: 출력 파일 경로.
        source_file: 원본 파일 이름.

    Returns:
        저장된 파일의 Path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = format_report(report, source_file)
    path.write_text(content, encoding="utf-8")

    log.info("품질 리포트 저장", path=str(path))
    return path


def report_to_dict(report: QualityReport, source_file: str = "") -> dict[str, Any]:
    """품질 리포트를 딕셔너리로 변환한다 (JSON 직렬화용).

    Args:
        report: QualityReport 인스턴스.
        source_file: 원본 파일 이름.

    Returns:
        리포트 딕셔너리.
    """
    issue_counts = _count_issues_by_check(report)

    return {
        "source_file": source_file,
        "total_segments": report.total_segments,
        "errors": report.error_count,
        "warnings": report.warning_count,
        "auto_fixed": report.auto_fixed_count,
        "quality_summary": report.quality_summary.to_dict()
        if report.quality_summary.grade
        else None,
        "issues_by_check": {
            check: {"count": total, "auto_fixed": fixed}
            for check, (total, fixed) in issue_counts.items()
        },
    }


def _count_issues_by_check(report: QualityReport) -> dict[str, tuple[int, int]]:
    """검사 항목별 이슈 수 및 교정 수를 집계한다.

    Returns:
        ``{check_name: (total_count, auto_fixed_count)}``
    """
    counts: dict[str, list[int]] = {}

    for issue in report.issues:
        if issue.check not in counts:
            counts[issue.check] = [0, 0]
        counts[issue.check][0] += 1
        if issue.auto_fixed:
            counts[issue.check][1] += 1

    return {k: (v[0], v[1]) for k, v in counts.items()}


def _check_label(check: str) -> str:
    """검사 항목 이름을 사람이 읽기 쉬운 레이블로 변환한다."""
    labels: dict[str, str] = {
        "timecode_inversion": "타임코드 역전",
        "timecode_overlap": "타임코드 겹침",
        "line_length": "줄 길이 초과",
        "cps_exceeded": "CPS 초과    ",
        "short_duration": "초단 자막    ",
        "empty_segment": "빈 자막      ",
    }
    return labels.get(check, check.ljust(14))
