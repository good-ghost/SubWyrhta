"""VAD + Pre-VAD 단위 테스트."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.audio.vad import (
    SpeechSegment,
    _enforce_max_chunk,
    _enforce_min_chunk,
    _merge_by_gap,
    apply_offset,
    export_vad_timeline,
    merge_speech_segments,
)
from src.core.config import VADConfig

if TYPE_CHECKING:
    from pathlib import Path


class TestSpeechSegment:
    """SpeechSegment 데이터클래스 테스트."""

    def test_duration(self) -> None:
        seg = SpeechSegment(start=1.0, end=3.5)
        assert seg.duration == pytest.approx(2.5)

    def test_to_dict(self) -> None:
        seg = SpeechSegment(start=0.5, end=2.0, skip_alignment=True)
        d = seg.to_dict()
        assert d["start"] == 0.5
        assert d["end"] == 2.0
        assert d["duration"] == 1.5
        assert d["skip_alignment"] is True

    def test_default_skip_alignment(self) -> None:
        seg = SpeechSegment(start=0.0, end=1.0)
        assert seg.skip_alignment is False


class TestMergeByGap:
    """gap 기반 병합 테스트."""

    def test_merge_close_segments(self) -> None:
        segs = [SpeechSegment(0.0, 1.0), SpeechSegment(1.3, 2.0)]
        result = _merge_by_gap(segs, 0.5)
        assert len(result) == 1
        assert result[0].end == 2.0

    def test_keep_distant_segments(self) -> None:
        segs = [SpeechSegment(0.0, 1.0), SpeechSegment(3.0, 4.0)]
        result = _merge_by_gap(segs, 0.5)
        assert len(result) == 2

    def test_chain_merge(self) -> None:
        segs = [
            SpeechSegment(0.0, 1.0),
            SpeechSegment(1.2, 2.0),  # gap=0.2
            SpeechSegment(2.3, 3.0),  # gap=0.3
        ]
        result = _merge_by_gap(segs, 0.5)
        assert len(result) == 1
        assert result[0].end == 3.0

    def test_empty(self) -> None:
        assert _merge_by_gap([], 0.5) == []

    def test_single(self) -> None:
        segs = [SpeechSegment(0.0, 1.0)]
        result = _merge_by_gap(segs, 0.5)
        assert len(result) == 1


class TestEnforceMinChunk:
    """최소 청크 길이 보장 테스트."""

    def test_short_merged_with_next(self) -> None:
        segs = [SpeechSegment(0.0, 0.3), SpeechSegment(1.0, 3.0)]
        result = _enforce_min_chunk(segs, 1.0)
        assert len(result) == 1
        assert result[0].end == 3.0

    def test_last_short_merged_with_prev(self) -> None:
        segs = [SpeechSegment(0.0, 5.0), SpeechSegment(6.0, 6.3)]
        result = _enforce_min_chunk(segs, 1.0)
        assert len(result) == 1
        assert result[0].end == 6.3

    def test_all_long_unchanged(self) -> None:
        segs = [SpeechSegment(0.0, 2.0), SpeechSegment(3.0, 5.0)]
        result = _enforce_min_chunk(segs, 1.0)
        assert len(result) == 2

    def test_single(self) -> None:
        segs = [SpeechSegment(0.0, 0.3)]
        result = _enforce_min_chunk(segs, 1.0)
        assert len(result) == 1


class TestEnforceMaxChunk:
    """최대 청크 길이 제한 테스트."""

    def test_split_long_segment(self) -> None:
        segs = [SpeechSegment(0.0, 65.0)]
        result = _enforce_max_chunk(segs, 30.0)
        assert len(result) >= 3
        assert result[-1].end == pytest.approx(65.0)
        for seg in result:
            assert seg.duration <= 30.1

    def test_short_unchanged(self) -> None:
        segs = [SpeechSegment(0.0, 10.0)]
        result = _enforce_max_chunk(segs, 30.0)
        assert len(result) == 1

    def test_exact_boundary(self) -> None:
        segs = [SpeechSegment(0.0, 30.0)]
        result = _enforce_max_chunk(segs, 30.0)
        assert len(result) == 1


class TestMergeSpeechSegments:
    """전체 병합 파이프라인 테스트."""

    def test_full_pipeline(self) -> None:
        raw = [
            SpeechSegment(0.0, 0.5),
            SpeechSegment(0.6, 1.5),  # gap=0.1
            SpeechSegment(5.0, 6.0),
            SpeechSegment(6.2, 6.5),  # gap=0.2
        ]
        config = VADConfig(merge_gap_s=0.5, min_chunk_s=1.0, max_chunk_s=30.0)
        result = merge_speech_segments(raw, config)
        assert len(result) >= 1
        # All chunks >= min_chunk_s (or merged)

    def test_alignment_skip_marking(self) -> None:
        raw = [SpeechSegment(0.0, 1.5)]  # 1.5s < 2.0s threshold
        config = VADConfig(merge_gap_s=0.5, min_chunk_s=0.5, max_chunk_s=30.0)
        result = merge_speech_segments(raw, config)
        assert result[0].skip_alignment is True

    def test_no_skip_for_long(self) -> None:
        raw = [SpeechSegment(0.0, 5.0)]
        config = VADConfig(merge_gap_s=0.5, min_chunk_s=1.0, max_chunk_s=30.0)
        result = merge_speech_segments(raw, config)
        assert result[0].skip_alignment is False

    def test_empty(self) -> None:
        config = VADConfig()
        assert merge_speech_segments([], config) == []


class TestApplyOffset:
    """타임스탬프 오프셋 보정 테스트."""

    def test_basic_offset(self) -> None:
        segs = [{"start": 0.0, "end": 1.5, "text": "Hello"}]
        result = apply_offset(segs, 10.0)
        assert result[0]["start"] == 10.0
        assert result[0]["end"] == 11.5

    def test_word_offset(self) -> None:
        segs = [
            {
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "Hi", "start": 0.0, "end": 0.5},
                ],
            }
        ]
        result = apply_offset(segs, 5.0)
        assert result[0]["words"][0]["start"] == 5.0
        assert result[0]["words"][0]["end"] == 5.5

    def test_word_without_timestamps(self) -> None:
        segs = [{"start": 0.0, "end": 1.0, "words": [{"word": "orphan"}]}]
        result = apply_offset(segs, 5.0)
        assert result[0]["words"][0] == {"word": "orphan"}


class TestExportVadTimeline:
    """VAD 타임라인 내보내기 테스트."""

    def test_export(self, tmp_path: Path) -> None:
        segs = [SpeechSegment(0.0, 2.5), SpeechSegment(3.0, 5.0)]
        out = export_vad_timeline(segs, tmp_path / "vad.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total_segments"] == 2
        assert data["total_speech_s"] == pytest.approx(4.5)

    def test_creates_dirs(self, tmp_path: Path) -> None:
        segs = [SpeechSegment(0.0, 1.0)]
        out = export_vad_timeline(segs, tmp_path / "sub" / "vad.json")
        assert out.exists()
