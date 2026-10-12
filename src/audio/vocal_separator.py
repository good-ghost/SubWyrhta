"""demucs 기반 배경음/보컬 분리.

음악, 효과음과 음성을 분리하여 보컬 트랙만 추출한다.
demucs가 설치되지 않았으면 graceful하게 건너뛴다.
"""

from __future__ import annotations

import shutil
import tempfile
import wave
from pathlib import Path
from typing import Any

import structlog

from src.audio.extractor import TARGET_SAMPLE_RATE
from src.core.exceptions import AudioEnhancementError

log = structlog.get_logger(__name__)

# demucs 기본 모델
_DEFAULT_MODEL = "htdemucs"

# demucs가 분리하는 소스 이름
_VOCAL_SOURCE = "vocals"
# float -1~1 → 16-bit PCM
_PCM16_MAX = 32767


def is_demucs_available() -> bool:
    """demucs 패키지 사용 가능 여부를 반환한다."""
    try:
        import demucs  # noqa: F401

        return True
    except ImportError:
        return False


def separate_vocals(
    audio_path: str | Path,
    output_path: str | Path | None = None,
    model_name: str = _DEFAULT_MODEL,
    device: str = "cpu",
) -> Path:
    """오디오에서 보컬 트랙을 분리한다.

    Args:
        audio_path: 입력 오디오 파일 경로.
        output_path: 보컬 출력 경로. None이면 임시 파일 생성.
        model_name: demucs 모델 이름 (기본: 'htdemucs').
        device: 디바이스 ('cpu', 'cuda').

    Returns:
        보컬 트랙 파일의 Path.

    Raises:
        AudioEnhancementError: 분리 실패 시.
    """
    if not is_demucs_available():
        raise AudioEnhancementError(
            "demucs 패키지가 설치되지 않았습니다. pip install -r requirements.txt 로 설치해 주세요."
        )

    input_path = Path(audio_path)
    if not input_path.exists():
        raise AudioEnhancementError(f"입력 파일이 존재하지 않습니다: {input_path}")

    log.info("demucs 보컬 분리 시작", input=str(input_path), model=model_name, device=device)

    try:
        return _run_demucs(input_path, output_path, model_name, device)
    except AudioEnhancementError:
        raise
    except Exception as e:
        raise AudioEnhancementError(f"demucs 보컬 분리 실패: {e}") from e


def separate_vocals_safe(
    audio_path: str | Path,
    output_path: str | Path | None = None,
    model_name: str = _DEFAULT_MODEL,
    device: str = "cpu",
) -> Path | None:
    """보컬 분리를 시도하고, 실패 시 None을 반환한다.

    demucs 미설치 또는 분리 실패 시 원본 오디오 경로를 그대로 반환하지 않고
    None을 반환하여 호출자가 원본을 사용하도록 한다.

    Args:
        audio_path: 입력 오디오 파일 경로.
        output_path: 보컬 출력 경로.
        model_name: demucs 모델 이름.
        device: 디바이스.

    Returns:
        보컬 트랙 Path. 실패 시 None.
    """
    if not is_demucs_available():
        log.info("demucs 미설치, 보컬 분리 건너뜀")
        return None

    try:
        return separate_vocals(audio_path, output_path, model_name, device)
    except AudioEnhancementError as e:
        log.warning("보컬 분리 실패, 원본 사용", error=str(e))
        return None


def _run_demucs(
    input_path: Path,
    output_path: str | Path | None,
    model_name: str,
    device: str,
) -> Path:
    """demucs를 실행하여 보컬을 분리한다."""
    import demucs.api

    separator = demucs.api.Separator(model=model_name, device=device)

    # demucs는 디렉토리에 소스별 파일을 생성한다
    with tempfile.TemporaryDirectory() as tmp_dir:
        # demucs API로 분리
        _origin, separated = separator.separate_audio_file(input_path)

        # 보컬 트랙 추출
        if _VOCAL_SOURCE not in separated:
            available = list(separated.keys())
            raise AudioEnhancementError(f"보컬 트랙을 찾을 수 없습니다. 사용 가능: {available}")

        vocal_tensor = separated[_VOCAL_SOURCE]

        # 출력 경로 결정
        if output_path is not None:
            out = Path(output_path)
        else:
            out = Path(tmp_dir) / f"{input_path.stem}.vocals.wav"

        out.parent.mkdir(parents=True, exist_ok=True)

        # tensor → WAV 파일 저장
        _save_audio_tensor(vocal_tensor, out, separator.samplerate)

        # 임시 디렉토리 사용 시 영구 위치로 복사
        if output_path is None:
            import os

            fd, permanent = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            shutil.copy2(str(out), permanent)
            out = Path(permanent)

        log.info("demucs 보컬 분리 완료", output=str(out))
        return out


def _save_audio_tensor(tensor: Any, path: Path, sample_rate: int) -> None:
    """보컬 텐서를 파이프라인 입력 형식(16kHz mono 16-bit PCM WAV)으로 저장한다.

    ``torchaudio.save`` 는 쓰지 않는다: torchaudio 2.8 은 저장 백엔드(soundfile 등)가 없으면
    ``RuntimeError`` 로 실패해 보컬 분리가 조용히 건너뛰어졌다 (VAD 읽기와 같은 원인, ``vad.py``).
    정규화를 끄면 VAD 가 이 파일을 바로 읽으므로 추출 단계와 같은 형식으로 맞춘다.

    Args:
        tensor: demucs 출력 텐서 (channels, samples), 값 범위 -1~1.
        path: 저장할 WAV 경로.
        sample_rate: 텐서의 샘플레이트 (demucs 모델 기본 44100).
    """
    import torch
    import torchaudio.functional as audio_functional

    audio = tensor.detach().to("cpu", torch.float32)
    if audio.dim() > 1:
        audio = audio.mean(dim=0)  # mono 믹스다운
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = audio_functional.resample(audio, sample_rate, TARGET_SAMPLE_RATE)
    pcm = (audio.clamp(-1.0, 1.0) * _PCM16_MAX).round().to(torch.int16)

    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(TARGET_SAMPLE_RATE)
        out.writeframes(pcm.numpy().tobytes())
