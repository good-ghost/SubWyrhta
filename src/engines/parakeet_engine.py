"""NVIDIA NeMo Parakeet STT 엔진 구현.

NeMo Parakeet 모델은 영어 특화 고성능 ASR을 제공한다.
nemo_toolkit 미설치 시 graceful하게 에러를 발생시킨다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import ModelLoadError, TranscriptionError
from src.engines.base import STTEngine, resolve_device
from src.models.result import Metadata, TranscriptionResult
from src.models.segment import TranscriptionSegment

if TYPE_CHECKING:
    from src.core.config import STTConfig

from src.version import __version__

log = structlog.get_logger(__name__)

# Parakeet 모델 이름 매핑
_PARAKEET_MODELS: dict[str, str] = {
    "parakeet-ctc-1.1b": "nvidia/parakeet-ctc-1.1b",
    "parakeet-rnnt-1.1b": "nvidia/parakeet-rnnt-1.1b",
    "parakeet-tdt-1.1b": "nvidia/parakeet-tdt-1.1b",
    "parakeet-ctc-0.6b": "nvidia/parakeet-ctc-0.6b",
}

# model_size → Parakeet 모델 매핑 (기본값)
_DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-1.1b"


def is_nemo_available() -> bool:
    """nemo_toolkit 패키지 사용 가능 여부."""
    try:
        import nemo.collections.asr as nemo_asr  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_parakeet_model(model_size: str) -> str:
    """model_size를 Parakeet 모델 이름으로 변환한다."""
    if model_size in _PARAKEET_MODELS:
        return _PARAKEET_MODELS[model_size]
    # nvidia/ 접두사가 붙은 전체 이름
    if model_size.startswith("nvidia/parakeet"):
        return model_size
    # 기본 모델
    return _DEFAULT_PARAKEET_MODEL


class ParakeetEngine(STTEngine):
    """NVIDIA NeMo Parakeet STT 엔진.

    영어 특화 고성능 ASR을 제공한다. nemo_toolkit이 필요하다.
    """

    def __init__(self, config: STTConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._device: str = ""
        self._model_name: str = ""

    @property
    def engine_name(self) -> str:
        return "parakeet"

    def load_model(self) -> None:
        """Parakeet 모델을 로드한다.

        Raises:
            ModelLoadError: nemo_toolkit 미설치 또는 모델 로드 실패 시.
        """
        if not is_nemo_available():
            raise ModelLoadError(
                "nemo_toolkit이 설치되지 않았습니다. "
                "pip install nemo_toolkit[asr] 로 설치해 주세요."
            )

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise ModelLoadError(f"nemo_toolkit ASR 모듈을 불러올 수 없습니다: {e}") from e

        self._device = resolve_device(self.config.device)
        self._model_name = _resolve_parakeet_model(self.config.model_size)

        log.info(
            "Parakeet 모델 로드 시작",
            model=self._model_name,
            device=self._device,
        )

        try:
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self._model_name,
            )

            # 디바이스 설정
            if self._device == "cuda":
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()

            self._model.eval()
            self.is_loaded = True

            log.info(
                "Parakeet 모델 로드 완료",
                model=self._model_name,
                device=self._device,
            )

        except Exception as e:
            if self._device != "cpu":
                log.warning("GPU 로드 실패, CPU 폴백 시도", error=str(e))
                try:
                    self._model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name=self._model_name,
                    )
                    self._model = self._model.cpu()
                    self._model.eval()
                    self._device = "cpu"
                    self.is_loaded = True
                    log.info("Parakeet 모델 CPU 폴백 완료", model=self._model_name)
                    return
                except Exception as fallback_err:
                    raise ModelLoadError(
                        f"Parakeet 모델 로드 실패 (CPU 폴백 포함): {fallback_err}"
                    ) from fallback_err

            raise ModelLoadError(f"Parakeet 모델 로드 실패: {self._model_name} ({e})") from e

    def unload_model(self) -> None:
        """모델을 언로드하여 메모리를 해제한다."""
        if self._model is not None:
            del self._model
            self._model = None
            self.is_loaded = False

            if self._device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass

            log.info("Parakeet 모델 언로드 완료")

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Parakeet으로 음성을 텍스트로 변환한다.

        Args:
            audio_path: 16kHz mono WAV 파일 경로.
            language: 언어 코드 (Parakeet은 영어 전용이므로 무시됨).

        Returns:
            TranscriptionResult 인스턴스.

        Raises:
            TranscriptionError: 추론 실패 시.
        """
        if not self.is_loaded or self._model is None:
            raise TranscriptionError(
                "모델이 로드되지 않았습니다. load_model()을 먼저 호출해 주세요."
            )

        path = Path(audio_path)
        if not path.exists():
            raise TranscriptionError(f"오디오 파일이 존재하지 않습니다: {path}")

        log.info("Parakeet 추론 시작", audio=str(path))

        try:
            # NeMo transcribe
            hypotheses = self._model.transcribe([str(path)])

            # hypotheses는 리스트 또는 (texts, hypotheses) 튜플
            if isinstance(hypotheses, tuple):
                texts = hypotheses[0]
            elif isinstance(hypotheses, list):
                texts = hypotheses
            else:
                texts = [str(hypotheses)]

            full_text = texts[0] if texts else ""

            # 타임스탬프 추출 시도
            segments = self._extract_segments_with_timestamps(path, full_text)

            return TranscriptionResult(
                metadata=Metadata(
                    source_file=str(path),
                    duration_s=0.0,
                    language="en",
                    model=f"parakeet/{self._model_name}",
                    created_at=datetime.now(tz=UTC).isoformat(),
                    subwyrhta_version=__version__,
                ),
                segments=segments,
            )

        except Exception as e:
            if isinstance(e, TranscriptionError):
                raise
            raise TranscriptionError(f"Parakeet 추론 실패: {e}") from e

    def _extract_segments_with_timestamps(
        self,
        audio_path: Path,
        full_text: str,
    ) -> list[TranscriptionSegment]:
        """타임스탬프 포함 세그먼트를 추출한다.

        NeMo가 타임스탬프를 제공하면 활용하고,
        아니면 전체 텍스트를 단일 세그먼트로 반환한다.
        """
        # NeMo 모델의 타임스탬프 지원 시도
        try:
            if hasattr(self._model, "transcribe_with_timestamps"):
                result = self._model.transcribe_with_timestamps([str(audio_path)])
                return self._parse_timestamped_result(result)
        except Exception:
            pass

        # 타임스탬프 없이 단일 세그먼트 반환
        if not full_text.strip():
            return []

        return [
            TranscriptionSegment(
                index=1,
                text=full_text.strip(),
                start=0.0,
                end=0.0,
                confidence=0.9,
            )
        ]

    def _parse_timestamped_result(
        self,
        result: Any,
    ) -> list[TranscriptionSegment]:
        """NeMo 타임스탬프 결과를 파싱한다."""
        segments: list[TranscriptionSegment] = []

        # NeMo의 타임스탬프 형식은 모델에 따라 다양함
        if isinstance(result, tuple) and len(result) >= 2:
            texts, timestamps = result[0], result[1]
            if isinstance(timestamps, list) and timestamps:
                for idx, (text, ts) in enumerate(
                    zip(texts, timestamps, strict=False),
                    start=1,
                ):
                    if isinstance(ts, dict):
                        start = float(ts.get("start", 0.0))
                        end = float(ts.get("end", 0.0))
                    elif isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        start, end = float(ts[0]), float(ts[1])
                    else:
                        start, end = 0.0, 0.0

                    segments.append(
                        TranscriptionSegment(
                            index=idx,
                            text=str(text).strip(),
                            start=start,
                            end=end,
                            confidence=0.9,
                        )
                    )

        if not segments and isinstance(result, (list, tuple)):
            text = str(result[0][0]) if result and result[0] else ""
            if text.strip():
                segments.append(
                    TranscriptionSegment(
                        index=1,
                        text=text.strip(),
                        start=0.0,
                        end=0.0,
                        confidence=0.9,
                    )
                )

        return segments
