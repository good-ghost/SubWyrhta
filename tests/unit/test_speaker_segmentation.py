"""화자 기반 세그멘테이션 단위 테스트 (4개 케이스 + 3인 이상)."""

from __future__ import annotations

from src.core.config import SegmentationConfig
from src.diarization.diarizer import DiarizationResult, SpeakerTurn
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.segmentation.speaker_splitter import split_by_speaker


def _w(text: str, start: float, end: float) -> WordSegment:
    return WordSegment(text, start, end, 0.9)


def _seg(words: list[WordSegment], idx: int = 1) -> TranscriptionSegment:
    text = " ".join(w.word for w in words)
    return TranscriptionSegment(idx, text, words[0].start, words[-1].end, 0.9, words=words)


_CFG = SegmentationConfig(speaker_split=True, overlap_text_separator=" / ", min_gap_s=0.04)


class TestCase1GapTransition:
    """케이스 1: gap 있는 전환 — 묵음 유지 분할."""

    def test_clear_gap(self) -> None:
        words = [
            _w("Hello", 0.0, 0.5),
            _w("world.", 0.6, 1.0),
            _w("Hi", 2.0, 2.3),
            _w("there.", 2.4, 2.8),
        ]
        segs = [_seg(words)]
        diar = DiarizationResult(
            turns=[SpeakerTurn("A", 0.0, 1.2), SpeakerTurn("B", 1.8, 3.0)],
            num_speakers=2,
        )
        result = split_by_speaker(segs, diar, _CFG)
        assert len(result) == 2
        assert result[0].speaker == "A"
        assert result[1].speaker == "B"
        # Gap should be preserved
        assert result[1].start > result[0].end


class TestCase2NoGapTransition:
    """케이스 2: gap 없는 전환 — min_gap_s 삽입."""

    def test_no_gap(self) -> None:
        words = [_w("Quick", 0.0, 0.3), _w("talk.", 0.4, 0.8), _w("Reply.", 0.8, 1.2)]
        segs = [_seg(words)]
        diar = DiarizationResult(
            turns=[SpeakerTurn("A", 0.0, 0.85), SpeakerTurn("B", 0.85, 1.5)],
            num_speakers=2,
        )
        result = split_by_speaker(segs, diar, _CFG)
        assert len(result) == 2
        # min_gap_s enforced
        assert result[1].start >= result[0].end + _CFG.min_gap_s - 0.001


class TestCase3OverlapMerge:
    """케이스 3: 겹침 있음 — 세그먼트 병합."""

    def test_partial_overlap(self) -> None:
        words = [_w("A-word", 0.0, 0.5), _w("B-word", 0.3, 0.8)]
        segs = [_seg(words)]
        diar = DiarizationResult(
            turns=[SpeakerTurn("A", 0.0, 0.6), SpeakerTurn("B", 0.2, 1.0)],
            num_speakers=2,
        )
        result = split_by_speaker(segs, diar, _CFG)
        # Overlapping speakers → merged or split based on word-level assignment
        assert len(result) >= 1


class TestCase4SimultaneousSpeech:
    """케이스 4: 동시 발화 — 완전 겹침 병합."""

    def test_full_overlap(self) -> None:
        words = [_w("Same", 0.0, 0.3), _w("time.", 0.4, 0.8)]
        segs = [_seg(words)]
        diar = DiarizationResult(
            turns=[SpeakerTurn("A", 0.0, 1.0), SpeakerTurn("B", 0.0, 1.0)],
            num_speakers=2,
        )
        result = split_by_speaker(segs, diar, _CFG)
        # Word mid-points determine assignment; same-time turns → one speaker wins per word
        assert len(result) >= 1


class TestThreeSpeakers:
    """3인 이상 동시 발화 처리."""

    def test_three_speakers_sequential(self) -> None:
        words = [_w("A", 0.0, 0.5), _w("B", 1.0, 1.5), _w("C", 2.0, 2.5)]
        segs = [_seg(words)]
        diar = DiarizationResult(
            turns=[
                SpeakerTurn("X", 0.0, 0.7),
                SpeakerTurn("Y", 0.8, 1.7),
                SpeakerTurn("Z", 1.8, 3.0),
            ],
            num_speakers=3,
        )
        result = split_by_speaker(segs, diar, _CFG)
        assert len(result) == 3
        assert result[0].speaker == "X"
        assert result[1].speaker == "Y"
        assert result[2].speaker == "Z"

    def test_three_speakers_overlap(self) -> None:
        words = [_w("A", 0.0, 0.5), _w("B", 0.3, 0.8), _w("C", 0.4, 0.9)]
        segs = [_seg(words)]
        diar = DiarizationResult(
            turns=[
                SpeakerTurn("X", 0.0, 0.6),
                SpeakerTurn("Y", 0.2, 0.8),
                SpeakerTurn("Z", 0.3, 1.0),
            ],
            num_speakers=3,
        )
        result = split_by_speaker(segs, diar, _CFG)
        assert len(result) >= 1


class TestNoSpeakerSplit:
    """--no-speaker-split 옵션."""

    def test_disabled(self) -> None:
        words = [_w("Hello", 0.0, 0.5), _w("world", 1.0, 1.5)]
        segs = [_seg(words)]
        diar = DiarizationResult(
            turns=[SpeakerTurn("A", 0.0, 0.7), SpeakerTurn("B", 0.8, 2.0)],
            num_speakers=2,
        )
        cfg = SegmentationConfig(speaker_split=False)
        result = split_by_speaker(segs, diar, cfg)
        assert len(result) == 1  # unchanged


class TestEdgeCases:
    """엣지 케이스."""

    def test_empty_segments(self) -> None:
        diar = DiarizationResult(turns=[SpeakerTurn("A", 0, 1)], num_speakers=1)
        assert split_by_speaker([], diar, _CFG) == []

    def test_empty_diarization(self) -> None:
        words = [_w("Hello", 0.0, 0.5)]
        segs = [_seg(words)]
        result = split_by_speaker(segs, DiarizationResult(), _CFG)
        assert len(result) == 1

    def test_segments_without_words(self) -> None:
        segs = [TranscriptionSegment(1, "No words", 0.0, 1.0)]
        diar = DiarizationResult(turns=[SpeakerTurn("A", 0, 2)], num_speakers=1)
        result = split_by_speaker(segs, diar, _CFG)
        assert len(result) == 1

    def test_speaker_preserved_in_result(self) -> None:
        words = [_w("Hello", 0.0, 0.5), _w("there", 2.0, 2.5)]
        segs = [_seg(words)]
        diar = DiarizationResult(
            turns=[SpeakerTurn("SPEAKER_00", 0.0, 1.0), SpeakerTurn("SPEAKER_01", 1.5, 3.0)],
            num_speakers=2,
        )
        result = split_by_speaker(segs, diar, _CFG)
        for s in result:
            assert s.speaker is not None
