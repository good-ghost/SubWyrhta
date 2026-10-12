"""교정한 문장의 단어 시각 다시 정렬 (WhisperX 정렬 모델, wav2vec2).

LLM 오인식 교정(``llm_correction``)이 글자를 바꾼 세그먼트는 단어 시각이 바뀐 글자와 맞지 않는다
(합친 단어는 첫·마지막 단어의 시각만 갖는다). 전처리한 음성으로 그 문장만 다시 정렬한다
(2026-10-11 사용자 결정, WI-9.018).

whisperx 가 없거나, 정렬 모델을 읽지 못하거나, 정렬 결과의 글자가 문장과 맞지 않으면 그 세그먼트는
원래 단어 시각을 둔다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from src.core.text import join_words
from src.models.word import WordSegment

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from src.models.segment import TranscriptionSegment

log = structlog.get_logger(__name__)


def realign_segments(
    segments: Sequence[TranscriptionSegment], audio_path: Path, language: str, device: str
) -> int:
    """세그먼트마다 텍스트를 음성에 다시 정렬해 단어를 바꾼다 (in-place).

    단어는 정렬 모델의 단위(일본어·중국어는 글자, 그 밖은 공백 단어)로 바뀐다. 문절 묶기 등은 호출한
    쪽이 다시 한다.

    Args:
        segments: 다시 정렬할 세그먼트 (시작·끝이 정렬 구간이다).
        audio_path: 전처리한 16kHz mono 음성.
        language: 언어 코드 (정렬 모델 선택).
        device: ``cpu`` / ``cuda`` 등.

    Returns:
        다시 정렬한 세그먼트 수.
    """
    if not segments:
        return 0
    try:
        import whisperx
    except ImportError:
        log.warning("whisperx 가 없어 교정한 문장을 다시 정렬하지 않음")
        return 0
    try:
        model, metadata = whisperx.load_align_model(language_code=language, device=device)
        audio = whisperx.load_audio(str(audio_path))
    except Exception as e:
        # 정렬 모델·음성 준비 실패(내려받기·형식 등)는 원래 시각으로 계속한다
        log.warning("정렬 모델·음성을 준비하지 못해 다시 정렬하지 않음", error=str(e))
        return 0

    realigned = 0
    try:
        for seg in segments:
            text = join_words(w.word for w in seg.words)
            words = _align_one(whisperx, model, metadata, audio, device, text, seg.start, seg.end)
            if words is None:
                continue
            seg.words = words
            seg.start, seg.end = words[0].start, max(w.end for w in words)
            realigned += 1
    finally:
        del model
    log.info("교정한 문장 다시 정렬", segments=len(segments), realigned=realigned)
    return realigned


def _align_one(
    whisperx: Any,
    model: Any,
    metadata: Any,
    audio: Any,
    device: str,
    text: str,
    start: float,
    end: float,
) -> list[WordSegment] | None:
    """한 문장을 정렬한 단어. 실패하거나 글자가 맞지 않으면 None."""
    try:
        aligned = whisperx.align(
            [{"text": text, "start": start, "end": end}],
            model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as e:
        # 한 문장 정렬 실패는 그 문장만 원래 시각으로 둔다
        log.warning("문장 다시 정렬 실패, 원래 시각을 둔다", text=text[:40], error=str(e))
        return None
    raw = [w for s in aligned.get("segments", []) for w in s.get("words", [])]
    words = _fill_times(raw, start, end)
    if not words or _compact(join_words(w.word for w in words)) != _compact(text):
        return None
    return words


def _fill_times(raw: list[dict[str, Any]], start: float, end: float) -> list[WordSegment]:
    """정렬 결과를 단어로. 시각이 없는 단어(숫자·기호 등)는 앞 단어 끝에 붙인다."""
    words: list[WordSegment] = []
    position = start
    for item in raw:
        word = str(item.get("word", ""))
        if not word.strip():
            continue
        if "start" in item and "end" in item:
            w_start, w_end = float(item["start"]), float(item["end"])
        else:
            w_start = w_end = position
        w_start = min(max(w_start, start), end)
        w_end = min(max(w_end, w_start), end)
        words.append(WordSegment(word, w_start, w_end, float(item.get("score", 0.0))))
        position = w_end
    return words


def _compact(text: str) -> str:
    return "".join(text.split())
