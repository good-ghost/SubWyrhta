"""STT 엔진 추상 기반 클래스 및 디바이스/compute_type 감지 유틸리티."""

from __future__ import annotations

import tempfile
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import TranscriptionError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from src.audio.vad import SpeechSegment
    from src.core.config import STTConfig
    from src.models.result import TranscriptionResult
    from src.models.segment import TranscriptionSegment

log = structlog.get_logger(__name__)


@dataclass
class ChunkTranscription:
    """Pre-VAD 청크 하나의 추론 결과.

    Attributes:
        index: 청크 인덱스 (merge_speech_segments 결과 순서, 0-based).
        segments: 원본 오디오 기준 타임스탬프로 보정된 세그먼트 목록.
        language: 이 청크에서 사용·감지한 언어 코드.
        model: 결과 메타데이터에 남길 모델 이름 (예: 'whisperx/large-v3').
    """

    index: int
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = ""
    model: str = ""


def offset_segments(
    segments: Sequence[TranscriptionSegment],
    offset_s: float,
) -> list[TranscriptionSegment]:
    """세그먼트와 단어 타임스탬프에 오프셋을 더한다 (in-place 보정 후 목록 반환).

    Args:
        segments: 청크 기준 타임스탬프의 세그먼트 목록.
        offset_s: 청크 시작 시각(초).

    Returns:
        보정된 세그먼트 목록.
    """
    for seg in segments:
        seg.start += offset_s
        seg.end += offset_s
        for word in seg.words:
            word.start += offset_s
            word.end += offset_s
    return list(segments)


def write_wav_slice(source: Path, start_s: float, end_s: float, target: Path) -> Path:
    """PCM WAV 파일의 [start_s, end_s) 구간을 새 WAV 파일로 저장한다.

    추출기가 만드는 16kHz mono pcm_s16le WAV를 전제로 한다. 구간마다 ffmpeg를 다시 부르지 않기 위해
    표준 라이브러리 wave 로 프레임을 잘라 쓴다.

    Args:
        source: 원본 WAV 경로.
        start_s: 시작 시각(초).
        end_s: 종료 시각(초).
        target: 저장할 WAV 경로.

    Returns:
        저장된 WAV 경로.

    Raises:
        TranscriptionError: WAV 파일을 읽거나 쓸 수 없을 때.
    """
    try:
        with wave.open(str(source), "rb") as reader:
            rate = reader.getframerate()
            total = reader.getnframes()
            first = min(total, max(0, int(start_s * rate)))
            last = min(total, max(first, int(end_s * rate)))
            reader.setpos(first)
            frames = reader.readframes(last - first)
            params = reader.getparams()
        with wave.open(str(target), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(frames)
    except (OSError, wave.Error, EOFError) as e:
        raise TranscriptionError(f"오디오 구간을 자를 수 없습니다: {source} ({e})") from e
    return target


# ──────────────────────────────────────────────
# 디바이스 감지 + compute_type 폴백
# ──────────────────────────────────────────────


def resolve_device(device: str) -> str:
    """디바이스 문자열을 실제 사용 가능한 디바이스로 해석한다.

    Args:
        device: 'auto', 'cuda', 'cpu', 'mps' 중 하나.

    Returns:
        실제 사용할 디바이스 문자열.
    """
    if device != "auto":
        return device

    try:
        import torch

        if torch.cuda.is_available():
            log.debug("CUDA 디바이스 감지됨")
            return "cuda"
        # ROCm은 torch.cuda.is_available()로 감지됨 (HIP 백엔드)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.debug("MPS 디바이스 감지됨")
            return "mps"
    except ImportError:
        pass

    log.debug("GPU 미감지, CPU 사용")
    return "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    """compute_type을 디바이스에 맞게 해석한다.

    Args:
        compute_type: 'auto', 'float16', 'float32', 'int8' 중 하나.
        device: 실제 사용 디바이스.

    Returns:
        사용할 compute_type 문자열.
    """
    if compute_type != "auto":
        return compute_type
    if device in ("cuda", "mps"):
        return "float16"
    return "float32"


def build_load_attempts(device: str, compute_type: str) -> list[tuple[str, str]]:
    """모델 로드 시도 순서를 구성한다.

    디바이스/compute_type 조합의 폴백 체인을 반환한다.

    - CUDA: float16 → int8 → CPU/float32
    - MPS: current → CPU/float32
    - CPU float16: → CPU/float32

    Args:
        device: 현재 디바이스.
        compute_type: 현재 compute_type.

    Returns:
        (device, compute_type) 튜플의 리스트.
    """
    attempts: list[tuple[str, str]] = [(device, compute_type)]

    if device == "cuda":
        if compute_type == "float16":
            attempts.append((device, "int8"))
        attempts.append(("cpu", "float32"))
    elif device == "mps" or (device == "cpu" and compute_type == "float16"):
        attempts.append(("cpu", "float32"))

    return attempts


def get_device_info() -> dict[str, Any]:
    """현재 시스템의 디바이스 정보를 반환한다."""
    info: dict[str, Any] = {"available_devices": ["cpu"]}

    try:
        import torch

        info["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            info["available_devices"].append("cuda")
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            try:
                mem = torch.cuda.get_device_properties(0).total_memory
                info["cuda_vram_gb"] = round(mem / 1024**3, 1)
            except RuntimeError as e:
                # VRAM 은 표시용이다. 조회에 실패해도 디바이스 감지는 계속한다
                log.warning("VRAM 조회 실패", error=str(e))

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["available_devices"].append("mps")

    except ImportError:
        info["torch_version"] = None

    return info


# ──────────────────────────────────────────────
# STTEngine ABC
# ──────────────────────────────────────────────


class STTEngine(ABC):
    """STT 엔진 추상 기반 클래스.

    모든 STT 엔진은 이 ABC를 상속하여 transcribe, load_model, unload_model을 구현한다.

    Attributes:
        config: STT 엔진 설정.
        is_loaded: 모델이 로드되었는지 여부.
    """

    def __init__(self, config: STTConfig) -> None:
        self.config = config
        self.is_loaded: bool = False

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """엔진 이름 (예: 'whisperx', 'faster-whisper')."""

    @abstractmethod
    def load_model(self) -> None:
        """모델을 로드한다.

        Raises:
            ModelLoadError: 모델 로드 실패 시.
        """

    @abstractmethod
    def unload_model(self) -> None:
        """모델을 언로드하여 메모리를 해제한다."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """음성 파일을 텍스트로 변환한다.

        Args:
            audio_path: 16kHz mono WAV 파일 경로.
            language: 언어 코드. None이면 자동 감지.

        Returns:
            TranscriptionResult 인스턴스.

        Raises:
            TranscriptionError: 추론 실패 시.
        """

    def transcribe_chunks(
        self,
        audio_path: str | Path,
        chunks: Sequence[tuple[int, SpeechSegment]],
        language: str | None = None,
    ) -> Iterator[ChunkTranscription]:
        """Pre-VAD 청크별로 독립 추론하고 원본 기준 타임스탬프로 보정해 하나씩 돌려준다.

        기본 구현은 청크마다 WAV 조각을 만들어 ``transcribe``를 부른다. 언어가 None이면 처음
        텍스트가 나온 청크의 감지 언어를 이후 청크에 고정한다 (청크마다 언어가 바뀌지 않게).

        Args:
            audio_path: 16kHz mono WAV 파일 경로.
            chunks: (청크 인덱스, 발화 구간) 목록. 체크포인트로 재개할 때는 남은 청크만 준다.
            language: 언어 코드. None이면 자동 감지.

        Yields:
            청크별 추론 결과.

        Raises:
            TranscriptionError: 추론 실패 시.
        """
        source = Path(audio_path)
        lang = language
        with tempfile.TemporaryDirectory(prefix="subwyrhta-chunks-") as tmp_dir:
            for index, chunk in chunks:
                piece = write_wav_slice(
                    source, chunk.start, chunk.end, Path(tmp_dir) / f"chunk_{index}.wav"
                )
                try:
                    result = self.transcribe(piece, language=lang)
                finally:
                    piece.unlink(missing_ok=True)
                if lang is None and result.segments and result.metadata.language:
                    lang = result.metadata.language
                yield ChunkTranscription(
                    index=index,
                    segments=offset_segments(result.segments, chunk.start),
                    language=result.metadata.language,
                    model=result.metadata.model,
                )

    def get_model_info(self) -> dict[str, Any]:
        """현재 엔진/모델 정보를 반환한다."""
        return {
            "engine": self.engine_name,
            "model_size": self.config.model_size,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "is_loaded": self.is_loaded,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"engine={self.engine_name!r}, "
            f"model_size={self.config.model_size!r}, "
            f"device={self.config.device!r})"
        )
