"""Silero VAD 통합 및 Pre-VAD 구조.

STT 이전에 VAD를 적용하여 발화 구간별 독립 추론을 가능하게 한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import VADError

if TYPE_CHECKING:
    from src.core.config import VADConfig

log = structlog.get_logger(__name__)

# Pre-VAD: 2.0초 미만 청크는 forced alignment 건너뛰기 권장
ALIGNMENT_SKIP_THRESHOLD_S = 2.0
# Silero VAD 가 받는 샘플레이트
_SILERO_RATE = 16_000


@dataclass
class SpeechSegment:
    """발화 구간.

    Attributes:
        start: 시작 시간(초).
        end: 종료 시간(초).
        skip_alignment: forced alignment 건너뛰기 여부.
    """

    start: float
    end: float
    skip_alignment: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "skip_alignment": self.skip_alignment,
        }


# ──────────────────────────────────────────────
# Silero VAD 래퍼
# ──────────────────────────────────────────────


def _load_silero_vad() -> tuple[Any, Any]:
    """Silero VAD 모델과 get_speech_timestamps 함수를 로드한다.

    설치된 silero-vad 패키지(requirements.txt)를 먼저 쓴다. 패키지가 없을 때만 torch.hub 로
    GitHub 저장소에서 받는다 (네트워크 필요).

    Raises:
        VADError: silero-vad를 불러올 수 없을 때.
    """
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        return load_silero_vad(), get_speech_timestamps
    except ImportError:
        log.info("silero-vad 패키지 없음, torch.hub 로 로드")
    try:
        import torch

        # torch.hub.load 는 타입 표기가 없다
        model, utils = torch.hub.load(  # type: ignore[no-untyped-call]
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        return model, utils[0]
    except Exception as e:
        raise VADError(
            f"Silero VAD 모델 로드 실패: {e}. pip install silero-vad torch 로 설치해 주세요."
        ) from e


def _read_wav_mono_16k(audio_path: str | Path) -> Any:
    """PCM WAV 를 16kHz mono float32 텐서로 읽는다.

    torchaudio.load 는 torchaudio 2.8 부터 별도 백엔드(torchcodec·soundfile 등)가 없으면 실패한다.
    추출기·음량 정규화가 만드는 파일은 16kHz mono pcm_s16le 이므로 표준 라이브러리 wave 로 읽는다.

    Raises:
        VADError: PCM WAV 가 아니어서 읽을 수 없을 때.
    """
    import wave

    import numpy as np
    import torch

    try:
        with wave.open(str(audio_path), "rb") as reader:
            rate = reader.getframerate()
            channels = reader.getnchannels()
            width = reader.getsampwidth()
            frames = reader.readframes(reader.getnframes())
    except (OSError, wave.Error, EOFError) as e:
        raise VADError(f"WAV 파일을 읽을 수 없습니다: {audio_path} ({e})") from e
    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}
    if width not in dtypes:
        raise VADError(f"지원하지 않는 WAV 샘플 크기입니다: {width * 8}bit ({audio_path})")
    samples = np.frombuffer(frames, dtype=dtypes[width]).astype(np.float32)
    if width == 1:
        samples = samples - 128.0
    samples /= float(2 ** (8 * width - 1))
    wav = torch.from_numpy(samples.reshape(-1, channels).mean(axis=1))
    if rate != _SILERO_RATE:
        import torchaudio.functional as audio_functional

        wav = audio_functional.resample(wav, orig_freq=rate, new_freq=_SILERO_RATE)
    return wav


def detect_speech_silero(
    audio_path: str | Path,
    config: VADConfig,
) -> list[SpeechSegment]:
    """Silero VAD로 발화 구간을 감지한다.

    Args:
        audio_path: 16kHz mono WAV 파일 경로.
        config: VAD 설정.

    Returns:
        발화 구간 목록.

    Raises:
        VADError: VAD 처리 실패 시.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise VADError(f"torch 가 설치되지 않았습니다: {e}") from e

    model, get_speech_timestamps = _load_silero_vad()
    wav = _read_wav_mono_16k(audio_path)

    try:
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=config.threshold,
            min_speech_duration_ms=config.min_speech_ms,
            min_silence_duration_ms=config.min_silence_ms,
            sampling_rate=_SILERO_RATE,
        )
    except Exception as e:
        raise VADError(f"Silero VAD 처리 실패: {e}") from e

    segments = [
        SpeechSegment(start=ts["start"] / _SILERO_RATE, end=ts["end"] / _SILERO_RATE)
        for ts in speech_timestamps
    ]
    log.info("Silero VAD 감지 완료", segments=len(segments))
    return segments


def detect_speech_webrtc(
    audio_path: str | Path,
    config: VADConfig,
) -> list[SpeechSegment]:
    """WebRTC VAD로 발화 구간을 감지한다 (Silero VAD 폴백).

    Args:
        audio_path: 16kHz mono WAV 파일 경로.
        config: VAD 설정.

    Returns:
        발화 구간 목록.

    Raises:
        VADError: WebRTC VAD 처리 실패 시.
    """
    try:
        import wave

        import webrtcvad
    except ImportError as e:
        raise VADError(
            f"webrtcvad가 설치되지 않았습니다: {e}. pip install webrtcvad 로 설치해 주세요."
        ) from e

    try:
        vad = webrtcvad.Vad()
        # threshold 0.0~1.0 → aggressiveness 0~3
        aggressiveness = min(3, max(0, int(config.threshold * 4)))
        vad.set_mode(aggressiveness)

        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            audio_data = wf.readframes(wf.getnframes())

        if sample_rate not in (8000, 16000, 32000, 48000):
            raise VADError(
                f"WebRTC VAD는 {sample_rate}Hz를 지원하지 않습니다. 16kHz WAV를 사용하세요."
            )

        frame_duration_ms = 30  # 10, 20, or 30 ms
        frame_size = int(sample_rate * frame_duration_ms / 1000) * sample_width * n_channels

        segments: list[SpeechSegment] = []
        is_speech = False
        speech_start = 0.0

        for i in range(0, len(audio_data) - frame_size + 1, frame_size):
            frame = audio_data[i : i + frame_size]
            time_s = i / (sample_rate * sample_width * n_channels)

            try:
                frame_is_speech = vad.is_speech(frame, sample_rate)
            except Exception:
                continue

            if frame_is_speech and not is_speech:
                speech_start = time_s
                is_speech = True
            elif not frame_is_speech and is_speech:
                segments.append(SpeechSegment(start=speech_start, end=time_s))
                is_speech = False

        # 마지막 발화 구간 닫기
        if is_speech:
            total_duration = len(audio_data) / (sample_rate * sample_width * n_channels)
            segments.append(SpeechSegment(start=speech_start, end=total_duration))

        log.info("WebRTC VAD 감지 완료", segments=len(segments))
        return segments

    except Exception as e:
        if isinstance(e, VADError):
            raise
        raise VADError(f"WebRTC VAD 처리 실패: {e}") from e


def detect_speech(
    audio_path: str | Path,
    config: VADConfig,
) -> list[SpeechSegment]:
    """VAD로 발화 구간을 감지한다.

    Silero VAD를 우선 사용하고, 실패 시 WebRTC VAD로 폴백한다.

    Args:
        audio_path: 16kHz mono WAV 파일 경로.
        config: VAD 설정.

    Returns:
        발화 구간 목록.
    """
    try:
        return detect_speech_silero(audio_path, config)
    except VADError as e:
        log.warning("Silero VAD 실패, WebRTC VAD로 폴백", error=str(e))
        return detect_speech_webrtc(audio_path, config)


# ──────────────────────────────────────────────
# Pre-VAD 구간 병합 및 처리
# ──────────────────────────────────────────────


def merge_speech_segments(
    segments: list[SpeechSegment],
    config: VADConfig,
) -> list[SpeechSegment]:
    """발화 구간을 병합하여 STT에 전달할 청크를 생성한다.

    병합 규칙:
        1. gap < merge_gap_s 이면 인접 구간 병합
        2. 병합 후 min_chunk_s 미만이면 인접 구간과 추가 병합
        3. max_chunk_s 초과 시 강제 분리

    Args:
        segments: 원시 발화 구간 목록.
        config: VAD 설정.

    Returns:
        병합/분리된 청크 목록.
    """
    if not segments:
        return []

    # 1단계: gap 기반 병합
    merged = _merge_by_gap(segments, config.merge_gap_s)

    # 2단계: 최소 길이 보장
    merged = _enforce_min_chunk(merged, config.min_chunk_s)

    # 3단계: 최대 길이 제한
    merged = _enforce_max_chunk(merged, config.max_chunk_s)

    # 4단계: alignment 건너뛰기 마킹
    for seg in merged:
        if seg.duration < ALIGNMENT_SKIP_THRESHOLD_S:
            seg.skip_alignment = True

    log.info(
        "Pre-VAD 구간 병합 완료",
        input_segments=len(segments),
        output_chunks=len(merged),
    )
    return merged


def _merge_by_gap(segments: list[SpeechSegment], merge_gap_s: float) -> list[SpeechSegment]:
    """gap이 merge_gap_s 미만인 인접 구간을 병합한다."""
    if not segments:
        return []

    result: list[SpeechSegment] = [SpeechSegment(start=segments[0].start, end=segments[0].end)]

    for seg in segments[1:]:
        prev = result[-1]
        gap = seg.start - prev.end

        if gap < merge_gap_s:
            prev.end = seg.end
        else:
            result.append(SpeechSegment(start=seg.start, end=seg.end))

    return result


def _enforce_min_chunk(segments: list[SpeechSegment], min_chunk_s: float) -> list[SpeechSegment]:
    """min_chunk_s 미만 구간을 인접 구간과 병합한다."""
    if len(segments) <= 1:
        return segments

    result: list[SpeechSegment] = []

    for seg in segments:
        if result and seg.duration < min_chunk_s:
            # 이전 구간과 병합
            result[-1].end = seg.end
        elif result and result[-1].duration < min_chunk_s:
            # 이전 구간이 짧으면 현재 구간과 병합
            result[-1].end = seg.end
        else:
            result.append(SpeechSegment(start=seg.start, end=seg.end))

    # 마지막 구간이 짧으면 이전과 병합
    if len(result) > 1 and result[-1].duration < min_chunk_s:
        result[-2].end = result[-1].end
        result.pop()

    return result


def _enforce_max_chunk(segments: list[SpeechSegment], max_chunk_s: float) -> list[SpeechSegment]:
    """max_chunk_s 초과 구간을 균등 분할한다."""
    result: list[SpeechSegment] = []

    for seg in segments:
        if seg.duration <= max_chunk_s:
            result.append(seg)
        else:
            # 균등 분할
            n_splits = int(seg.duration / max_chunk_s) + 1
            chunk_duration = seg.duration / n_splits
            for i in range(n_splits):
                chunk_start = seg.start + i * chunk_duration
                chunk_end = seg.start + (i + 1) * chunk_duration
                if i == n_splits - 1:
                    chunk_end = seg.end  # 마지막 청크는 정확히 끝까지
                result.append(SpeechSegment(start=chunk_start, end=chunk_end))

    return result


# ──────────────────────────────────────────────
# 타임스탬프 오프셋 보정
# ──────────────────────────────────────────────


def apply_offset(
    segments: list[dict[str, Any]],
    offset_s: float,
    min_gap_s: float = 0.04,
) -> list[dict[str, Any]]:
    """구간별 추론 결과의 타임스탬프에 오프셋을 가산한다.

    Args:
        segments: STT 추론 결과 세그먼트 (dict 형태).
        offset_s: 구간 시작 시간 오프셋(초).
        min_gap_s: 구간 경계 최소 갭.

    Returns:
        오프셋 적용된 세그먼트 목록.
    """
    result = []
    for seg in segments:
        adjusted = dict(seg)
        adjusted["start"] = seg["start"] + offset_s
        adjusted["end"] = seg["end"] + offset_s

        # 단어 타임스탬프도 보정
        if "words" in adjusted:
            adjusted["words"] = [
                {**w, "start": w["start"] + offset_s, "end": w["end"] + offset_s}
                if "start" in w and "end" in w
                else w
                for w in adjusted["words"]
            ]

        result.append(adjusted)
    return result


# ──────────────────────────────────────────────
# VAD 결과 내보내기
# ──────────────────────────────────────────────


def export_vad_timeline(
    segments: list[SpeechSegment],
    output_path: str | Path,
) -> Path:
    """VAD 결과를 JSON 타임라인으로 내보낸다.

    Args:
        segments: 발화 구간 목록.
        output_path: 출력 JSON 파일 경로.

    Returns:
        저장된 파일의 Path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "vad_segments": [seg.to_dict() for seg in segments],
        "total_segments": len(segments),
        "total_speech_s": round(sum(seg.duration for seg in segments), 3),
    }

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("VAD 타임라인 내보내기 완료", path=str(path), segments=len(segments))
    return path
