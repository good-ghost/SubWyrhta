"""화자 기반 세그멘테이션.

화자 전환 지점에서 분할하고, 겹침/동시 발화는 병합한다.

처리 흐름:
    1. diarization 결과의 화자 전환 지점 탐색
    2. 케이스 1: gap 있는 전환 → 묵음 유지 분할
    3. 케이스 2: gap 없는 전환 → min_gap_s 삽입 후 분할
    4. 케이스 3/4: 겹침/동시 발화 → 세그먼트 병합 (구분자 연결)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.core.text import join_words
from src.models.segment import TranscriptionSegment

if TYPE_CHECKING:
    from src.core.config import SegmentationConfig
    from src.diarization.diarizer import DiarizationResult, SpeakerTurn
    from src.models.word import WordSegment

log = structlog.get_logger(__name__)


def split_by_speaker(
    segments: list[TranscriptionSegment],
    diarization: DiarizationResult,
    config: SegmentationConfig,
) -> list[TranscriptionSegment]:
    """화자 전환 지점에서 세그먼트를 분할하고 겹침을 병합한다.

    Args:
        segments: STT 세그먼트 목록.
        diarization: 화자 분리 결과.
        config: 세그멘테이션 설정.

    Returns:
        화자 기반 분할/병합된 세그먼트 목록.
    """
    if not config.speaker_split:
        log.debug("화자 기반 분할 비활성화 (--no-speaker-split)")
        return list(segments)

    if not diarization.turns or not segments:
        return list(segments)

    # 1단계: 단어 레벨 화자 할당
    all_words = _collect_all_words(segments)
    if not all_words:
        return list(segments)

    _assign_speaker_to_words(all_words, diarization.turns)

    # 2단계: 화자 전환 지점에서 분할
    raw_groups = _group_by_speaker(all_words)

    # 3단계: 겹침 병합
    merged = _merge_overlapping_groups(raw_groups, config.overlap_text_separator)

    # 4단계: 세그먼트 구성
    result = _build_segments(merged, config.min_gap_s)

    log.info(
        "화자 기반 세그멘테이션 완료",
        input_segments=len(segments),
        output_segments=len(result),
        speakers=diarization.num_speakers,
    )
    return result


def _collect_all_words(segments: list[TranscriptionSegment]) -> list[WordSegment]:
    """모든 세그먼트에서 단어를 수집한다."""
    words: list[WordSegment] = []
    for seg in segments:
        words.extend(seg.words)
    return words


def _assign_speaker_to_words(
    words: list[WordSegment],
    turns: list[SpeakerTurn],
) -> None:
    """각 단어에 화자 ID를 할당한다 (단어 객체에 _speaker 속성 추가).

    단어의 중간 시점이 속하는 화자 턴을 찾아 할당한다. 어느 턴에도 속하지 않으면 가장 가까운 턴의
    화자로 둔다: 화자 없음으로 두면 같은 화자의 말이 턴 사이 틈에서 갈라지고, 그 조각이 다른
    화자처럼 구분자(` / `)로 이어졌다.
    """
    for word in words:
        mid = (word.start + word.end) / 2
        speaker = _find_speaker_at(mid, turns)
        word._speaker = speaker  # type: ignore[attr-defined]


def _find_speaker_at(time_s: float, turns: list[SpeakerTurn]) -> str | None:
    """주어진 시점의 화자를 찾는다. 속한 턴이 없으면 가장 가까운 턴의 화자 (턴이 없으면 None)."""
    for turn in turns:
        if turn.start <= time_s <= turn.end:
            return turn.speaker
    if not turns:
        return None
    nearest = min(turns, key=lambda t: t.start - time_s if time_s < t.start else time_s - t.end)
    return nearest.speaker


# ──────────────────────────────────────────────
# 화자 그룹
# ──────────────────────────────────────────────


class _SpeakerGroup:
    """같은 화자의 연속 단어 그룹."""

    def __init__(self, speaker: str | None, words: list[WordSegment]) -> None:
        self.speaker = speaker
        self.words = words

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        return join_words(w.word for w in self.words)


def _group_by_speaker(words: list[WordSegment]) -> list[_SpeakerGroup]:
    """단어를 화자별 연속 그룹으로 나눈다."""
    if not words:
        return []

    groups: list[_SpeakerGroup] = []
    current_speaker = getattr(words[0], "_speaker", None)
    current_words: list[WordSegment] = [words[0]]

    for word in words[1:]:
        speaker = getattr(word, "_speaker", None)
        if speaker != current_speaker:
            groups.append(_SpeakerGroup(current_speaker, current_words))
            current_speaker = speaker
            current_words = [word]
        else:
            current_words.append(word)

    groups.append(_SpeakerGroup(current_speaker, current_words))
    return groups


# ──────────────────────────────────────────────
# 겹침 병합
# ──────────────────────────────────────────────


class _MergedGroup:
    """병합 결과 그룹 (단일 또는 다화자)."""

    def __init__(self) -> None:
        self.speakers: list[str] = []
        self.texts: list[str] = []
        self.words: list[WordSegment] = []
        self.start: float = 0.0
        self.end: float = 0.0
        # 마지막으로 더한 그룹의 화자 (같은 화자가 이어지면 텍스트를 구분자 없이 잇는다)
        self.last_speaker: str | None = None

    @property
    def speaker(self) -> str | list[str] | None:
        if not self.speakers:
            return None
        if len(self.speakers) == 1:
            return self.speakers[0]
        return self.speakers

    @property
    def text(self) -> str:
        return (
            self._separator.join(self.texts)
            if len(self.texts) > 1
            else (self.texts[0] if self.texts else "")
        )

    _separator: str = " / "


def _merge_overlapping_groups(
    groups: list[_SpeakerGroup],
    separator: str,
) -> list[_MergedGroup]:
    """겹침이 있는 그룹을 병합한다.

    케이스 1/2: gap 있음 or gap 없음 → 별도 그룹 유지
    케이스 3/4: 겹침 → 단일 그룹으로 병합
    """
    if not groups:
        return []

    result: list[_MergedGroup] = []

    for group in groups:
        if not result:
            mg = _MergedGroup()
            mg._separator = separator
            mg.start = group.start
            mg.end = group.end
            mg.words = list(group.words)
            mg.texts = [group.text]
            mg.last_speaker = group.speaker
            if group.speaker:
                mg.speakers = [group.speaker]
            result.append(mg)
            continue

        prev = result[-1]

        # 겹침 판정: 현재 그룹의 시작이 이전 그룹의 끝 이전
        if group.start < prev.end:
            # 케이스 3/4: 겹침 → 병합
            prev.end = max(prev.end, group.end)
            prev.words.extend(group.words)
            if group.speaker is not None and group.speaker == prev.last_speaker:
                # 같은 화자의 말이 이어지면 구분자 없이 잇는다 (구분자는 다른 화자 사이에만)
                prev.texts[-1] = join_words([prev.texts[-1], group.text])
            else:
                prev.texts.append(group.text)
            prev.last_speaker = group.speaker
            if group.speaker and group.speaker not in prev.speakers:
                prev.speakers.append(group.speaker)
        else:
            # 케이스 1/2: gap 있음 or gap 없음 → 새 그룹
            mg = _MergedGroup()
            mg._separator = separator
            mg.start = group.start
            mg.end = group.end
            mg.words = list(group.words)
            mg.texts = [group.text]
            mg.last_speaker = group.speaker
            if group.speaker:
                mg.speakers = [group.speaker]
            result.append(mg)

    return result


# ──────────────────────────────────────────────
# 세그먼트 구성
# ──────────────────────────────────────────────


def _build_segments(
    groups: list[_MergedGroup],
    min_gap_s: float,
) -> list[TranscriptionSegment]:
    """병합된 그룹을 TranscriptionSegment로 변환한다."""
    segments: list[TranscriptionSegment] = []

    for i, group in enumerate(groups):
        # 신뢰도 계산
        confidences = [w.confidence for w in group.words if w.confidence > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        seg = TranscriptionSegment(
            index=i + 1,
            text=group.text,
            start=group.start,
            end=group.end,
            confidence=avg_conf,
            speaker=group.speaker,
            words=group.words,
        )
        segments.append(seg)

    # min_gap_s 보장 (케이스 2)
    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]
        if curr.start < prev.end + min_gap_s:
            curr.start = prev.end + min_gap_s

    return segments
