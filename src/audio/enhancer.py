"""오디오 향상: 음량 정규화, 노이즈 억제.

전처리 파이프라인의 개별 단계를 제공한다.
올바른 순서: 채널 믹스다운 → 샘플레이트 변환 → demucs → 음량 정규화 → 노이즈 억제
(채널/샘플레이트는 extractor.py에서 처리, demucs는 vocal_separator.py에서 처리)
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from pathlib import Path

import structlog

from src.audio.extractor import _find_ffmpeg, stderr_tail
from src.core.exceptions import AudioEnhancementError

log = structlog.get_logger(__name__)


# ──────────────────────────────────────────────
# 음량 정규화
# ──────────────────────────────────────────────


def normalize_loudness(
    audio_path: str | Path,
    output_path: str | Path | None = None,
    target_i: float = -16.0,
    target_lra: float = 11.0,
    target_tp: float = -1.5,
) -> Path:
    """ffmpeg loudnorm 필터로 음량을 정규화한다.

    EBU R128 표준 기반 라우드니스 정규화를 수행한다.

    Args:
        audio_path: 입력 오디오 파일 경로.
        output_path: 출력 경로. None이면 임시 파일 생성.
        target_i: 목표 통합 라우드니스 (LUFS). 기본 -16.
        target_lra: 목표 라우드니스 범위 (LU). 기본 11.
        target_tp: 목표 트루 피크 (dBTP). 기본 -1.5.

    Returns:
        정규화된 오디오 파일의 Path.

    Raises:
        AudioEnhancementError: 정규화 실패 시.
    """
    ffmpeg = _find_ffmpeg()
    input_path = Path(audio_path)

    if not input_path.exists():
        raise AudioEnhancementError(f"입력 파일이 존재하지 않습니다: {input_path}")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        # 입력 파일과 같은 디렉토리에 임시 파일 생성 (snap ffmpeg 호환)
        out = input_path.parent / f".{input_path.stem}.normalized.wav"

    loudnorm_filter = f"loudnorm=I={target_i}:LRA={target_lra}:TP={target_tp}:print_format=none"

    cmd = [
        ffmpeg,
        "-i",
        str(input_path),
        "-af",
        loudnorm_filter,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(out),
    ]

    log.info("음량 정규화 시작", input=str(input_path), target_i=target_i)

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise AudioEnhancementError(f"음량 정규화 실패: {stderr_tail(e.stderr)}") from e
    except subprocess.TimeoutExpired as e:
        raise AudioEnhancementError(f"음량 정규화 타임아웃: {input_path}") from e

    if not out.exists() or out.stat().st_size == 0:
        raise AudioEnhancementError(f"정규화된 오디오 파일이 비어 있습니다: {out}")

    log.info("음량 정규화 완료", output=str(out))
    return out


# ──────────────────────────────────────────────
# 노이즈 억제
# ──────────────────────────────────────────────


def is_noisereduce_available() -> bool:
    """noisereduce 패키지 사용 가능 여부를 반환한다."""
    try:
        import noisereduce  # noqa: F401

        return True
    except ImportError:
        return False


def reduce_noise(
    audio_path: str | Path,
    output_path: str | Path | None = None,
    model: str = "noisereduce",
) -> Path:
    """노이즈를 억제한다.

    Args:
        audio_path: 입력 오디오 파일 경로.
        output_path: 출력 경로. None이면 임시 파일 생성.
        model: 노이즈 억제 모델 ('noisereduce').

    Returns:
        노이즈 억제된 오디오 파일의 Path.

    Raises:
        AudioEnhancementError: 노이즈 억제 실패 시.
    """
    input_path = Path(audio_path)
    if not input_path.exists():
        raise AudioEnhancementError(f"입력 파일이 존재하지 않습니다: {input_path}")

    if model == "noisereduce":
        return _reduce_noise_noisereduce(input_path, output_path)

    raise AudioEnhancementError(f"지원하지 않는 노이즈 억제 모델: {model}")


def reduce_noise_safe(
    audio_path: str | Path,
    output_path: str | Path | None = None,
    model: str = "noisereduce",
) -> Path | None:
    """노이즈 억제를 시도하고, 실패 시 None을 반환한다."""
    if not is_noisereduce_available():
        log.info("noisereduce 미설치, 노이즈 억제 건너뜀")
        return None

    try:
        return reduce_noise(audio_path, output_path, model)
    except AudioEnhancementError as e:
        log.warning("노이즈 억제 실패, 원본 사용", error=str(e))
        return None


def _reduce_noise_noisereduce(
    input_path: Path,
    output_path: str | Path | None,
) -> Path:
    """noisereduce 라이브러리로 노이즈를 억제한다."""
    try:
        import noisereduce as nr
        import numpy as np
        import scipy.io.wavfile as wavfile
    except ImportError as e:
        raise AudioEnhancementError(f"noisereduce/scipy가 설치되지 않았습니다: {e}") from e

    log.info("noisereduce 노이즈 억제 시작", input=str(input_path))

    try:
        rate, data = wavfile.read(str(input_path))

        # float 변환
        if data.dtype == np.int16:
            audio_float = data.astype(np.float32) / 32768.0
        elif data.dtype == np.float32:
            audio_float = data
        else:
            audio_float = data.astype(np.float32)

        # mono 변환
        if audio_float.ndim > 1:
            audio_float = audio_float.mean(axis=1)

        # 노이즈 억제
        reduced = nr.reduce_noise(y=audio_float, sr=rate)

        # 출력 경로 결정
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out = input_path.parent / f".{input_path.stem}.denoised.wav"

        # int16 변환 + 저장
        reduced_int16 = (reduced * 32767).astype(np.int16)
        wavfile.write(str(out), rate, reduced_int16)

        log.info("noisereduce 노이즈 억제 완료", output=str(out))
        return out

    except Exception as e:
        if isinstance(e, AudioEnhancementError):
            raise
        raise AudioEnhancementError(f"noisereduce 실패: {e}") from e


# ──────────────────────────────────────────────
# 전처리 파이프라인 오케스트레이션
# ──────────────────────────────────────────────


def enhance_audio(
    audio_path: str | Path,
    output_path: str | Path | None = None,
    normalize: bool = True,
    denoise: bool = False,
    denoise_model: str = "noisereduce",
    separate_vocals: bool = False,
    device: str = "cpu",
) -> Path:
    """오디오 전처리 파이프라인을 실행한다.

    올바른 순서: demucs → 음량 정규화 → 노이즈 억제
    (채널 믹스다운/샘플레이트 변환은 extractor에서 이미 처리됨)

    Args:
        audio_path: 입력 오디오 경로 (16kHz mono WAV).
        output_path: 최종 출력 경로. None이면 임시 파일.
        normalize: 음량 정규화 활성화.
        denoise: 노이즈 억제 활성화.
        denoise_model: 노이즈 억제 모델.
        separate_vocals: demucs 보컬 분리 활성화.
        device: 디바이스 (demucs용).

    Returns:
        전처리된 오디오 파일의 Path.
    """
    current = Path(audio_path)
    temp_files: list[Path] = []

    try:
        # 1. demucs 보컬 분리
        if separate_vocals:
            from src.audio.vocal_separator import separate_vocals_safe

            result = separate_vocals_safe(current, device=device)
            if result is not None:
                temp_files.append(result)
                current = result

        # 2. 음량 정규화
        if normalize:
            result = normalize_loudness(current)
            temp_files.append(result)
            current = result

        # 3. 노이즈 억제
        if denoise:
            result = reduce_noise_safe(current, model=denoise_model)
            if result is not None:
                temp_files.append(result)
                current = result

        # 최종 출력
        if output_path is not None:
            final = Path(output_path)
            final.parent.mkdir(parents=True, exist_ok=True)
            if current != final:
                shutil.copy2(str(current), str(final))
            current = final

        return current

    finally:
        # 중간 임시 파일 정리 (최종 결과 제외)
        for tmp in temp_files:
            if tmp != current and tmp.exists():
                with contextlib.suppress(OSError):
                    tmp.unlink()
