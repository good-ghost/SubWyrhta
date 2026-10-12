"""품질 검사기 단위 테스트."""

from __future__ import annotations

from src.core.config import QualityConfig, SegmentationConfig
from src.models.result import Metadata, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.quality.checker import (
    Issue,
    QualityReport,
    Severity,
    _auto_line_break,
    _compute_quality_summary,
    check_quality,
)


def _seg(idx: int, text: str, start: float, end: float, conf: float = 0.9) -> TranscriptionSegment:
    return TranscriptionSegment(idx, text, start, end, conf)


def _result(segments: list[TranscriptionSegment], lang: str = "en") -> TranscriptionResult:
    return TranscriptionResult(metadata=Metadata(language=lang), segments=segments)


_QCFG = QualityConfig(enabled=True, auto_fix=True)
_SCFG = SegmentationConfig(max_chars_per_line=42, min_gap_s=0.04, min_duration_s=1.0)


class TestTimecodeInversion:
    def test_detect_and_fix(self) -> None:
        r = _result([_seg(1, "Test.", 3.0, 1.0)])  # inverted
        report = check_quality(r, _QCFG, _SCFG)
        issues = [i for i in report.issues if i.check == "timecode_inversion"]
        assert len(issues) == 1
        assert issues[0].auto_fixed is True
        assert r.segments[0].start == 1.0
        assert r.segments[0].end == 3.0

    def test_no_issue_when_correct(self) -> None:
        r = _result([_seg(1, "Test.", 1.0, 3.0)])
        report = check_quality(r, _QCFG, _SCFG)
        assert all(i.check != "timecode_inversion" for i in report.issues)


class TestTimecodeOverlap:
    def test_detect_and_fix(self) -> None:
        r = _result([_seg(1, "A.", 0.0, 2.0), _seg(2, "B.", 1.5, 3.0)])
        report = check_quality(r, _QCFG, _SCFG)
        issues = [i for i in report.issues if i.check == "timecode_overlap"]
        assert len(issues) == 1
        assert issues[0].auto_fixed is True
        assert r.segments[0].end < r.segments[1].start

    def test_no_overlap(self) -> None:
        r = _result([_seg(1, "A.", 0.0, 1.0), _seg(2, "B.", 1.5, 2.5)])
        report = check_quality(r, _QCFG, _SCFG)
        assert all(i.check != "timecode_overlap" for i in report.issues)


class TestLineLength:
    def test_detect_and_fix(self) -> None:
        long = (
            "This is a very long subtitle line that exceeds the maximum allowed characters per line"
        )
        r = _result([_seg(1, long, 0.0, 3.0)])
        report = check_quality(r, _QCFG, _SCFG)
        issues = [i for i in report.issues if i.check == "line_length"]
        assert len(issues) == 1
        assert issues[0].auto_fixed is True
        for line in r.segments[0].text.split("\n"):
            assert len(line) <= 42

    def test_within_limit(self) -> None:
        r = _result([_seg(1, "Short text.", 0.0, 1.0)])
        report = check_quality(r, _QCFG, _SCFG)
        assert all(i.check != "line_length" for i in report.issues)


class TestCPSCheck:
    def test_exceeded(self) -> None:
        r = _result([_seg(1, "x" * 100, 0.0, 1.0)])  # 100 CPS
        report = check_quality(r, _QCFG, _SCFG)
        issues = [i for i in report.issues if i.check == "cps_exceeded"]
        assert len(issues) == 1
        assert issues[0].severity == Severity.WARNING

    def test_within_limit(self) -> None:
        r = _result([_seg(1, "Hello.", 0.0, 2.0)])  # 3 CPS
        report = check_quality(r, _QCFG, _SCFG)
        assert all(i.check != "cps_exceeded" for i in report.issues)


class TestShortDuration:
    def test_detected(self) -> None:
        r = _result([_seg(1, "Hi.", 0.0, 0.3)])  # 0.3s
        report = check_quality(r, _QCFG, _SCFG)
        issues = [i for i in report.issues if i.check == "short_duration"]
        assert len(issues) == 1

    def test_within_limit(self) -> None:
        r = _result([_seg(1, "Ok.", 0.0, 1.5)])
        report = check_quality(r, _QCFG, _SCFG)
        assert all(i.check != "short_duration" for i in report.issues)


class TestEmptySegment:
    def test_removed(self) -> None:
        r = _result([_seg(1, "Keep.", 0.0, 1.0), _seg(2, "", 2.0, 3.0)])
        check_quality(r, _QCFG, _SCFG)
        assert len(r.segments) == 1
        assert r.segments[0].text == "Keep."

    def test_reindexed(self) -> None:
        r = _result([_seg(1, "A.", 0.0, 1.0), _seg(2, "", 2.0, 3.0), _seg(3, "B.", 4.0, 5.0)])
        check_quality(r, _QCFG, _SCFG)
        assert r.segments[1].index == 2


class TestDisabledQC:
    def test_no_checks(self) -> None:
        r = _result([_seg(1, "", 0.0, 1.0)])
        report = check_quality(r, QualityConfig(enabled=False), _SCFG)
        assert len(report.issues) == 0
        assert len(r.segments) == 1  # not removed


class TestQualitySummary:
    def test_grade_a(self) -> None:
        s = _compute_quality_summary([_seg(1, "a", 0, 1, 0.95)])
        assert s.grade == "A"

    def test_grade_b(self) -> None:
        s = _compute_quality_summary([_seg(1, "a", 0, 1, 0.85)])
        assert s.grade == "B"

    def test_grade_c(self) -> None:
        s = _compute_quality_summary([_seg(1, "a", 0, 1, 0.75)])
        assert s.grade == "C"

    def test_grade_d(self) -> None:
        s = _compute_quality_summary([_seg(1, "a", 0, 1, 0.5)])
        assert s.grade == "D"

    def test_empty(self) -> None:
        s = _compute_quality_summary([])
        assert s.grade == ""

    def test_low_confidence_count(self) -> None:
        segs = [_seg(1, "a", 0, 1, 0.5), _seg(2, "b", 1, 2, 0.9)]
        s = _compute_quality_summary(segs)
        assert s.low_confidence_count == 1


class TestAutoLineBreak:
    def test_short_text(self) -> None:
        assert _auto_line_break("short", 42) == "short"

    def test_long_text(self) -> None:
        text = "This is a long subtitle text that definitely exceeds the limit"
        result = _auto_line_break(text, 30)
        for line in result.split("\n"):
            assert len(line) <= 30


class TestQualityReportProperties:
    def test_counts(self) -> None:
        report = QualityReport(
            issues=[
                Issue(1, "a", Severity.ERROR, "msg", auto_fixed=True),
                Issue(2, "b", Severity.WARNING, "msg"),
                Issue(3, "c", Severity.ERROR, "msg"),
            ]
        )
        assert report.error_count == 2
        assert report.warning_count == 1
        assert report.auto_fixed_count == 1
