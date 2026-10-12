"""whisper-turbo STT 엔진 구현.

OpenAI whisper turbo 모델을 faster-whisper(CTranslate2) 백엔드로 실행한다.
turbo 모델은 large-v3 대비 빠른 추론 속도를 제공한다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import ModelLoadError, TranscriptionError
from src.engines.base import (
    STTEngine,
    build_load_attempts,
    resolve_compute_type,
    resolve_device,
)
from src.models.result import Metadata, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment

if TYPE_CHECKING:
    from src.core.config import STTConfig

from src.version import __version__

log = structlog.get_logger(__name__)

# turbo 모델 이름
_TURBO_MODEL = "turbo"


class WhisperTurboEngine(STTEngine):
    """whisper-turbo STT 엔진.

    OpenAI whisper turbo 모델을 faster-whisper 백엔드로 실행한다.
    config.model_size는 무시되며 항상 turbo 모델을 사용한다.
    """

    def __init__(self, config: STTConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._device: str = ""
        self._compute_type: str = ""

    @property
    def engine_name(self) -> str:
        return "whisper-turbo"

    def load_model(self) -> None:
        """whisper turbo 모델을 로드한다.

        faster-whisper 백엔드를 사용하며, GPU 실패 시 CPU로 폴백한다.

        Raises:
            ModelLoadError: 모든 폴백 시도 후에도 실패 시.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ModelLoadError(
                "faster-whisper 패키지가 설치되지 않았습니다. "
                "pip install -r requirements.txt 로 설치해 주세요."
            ) from e

        self._device = resolve_device(self.config.device)
        self._compute_type = resolve_compute_type(self.config.compute_type, self._device)

        model_dir = str(Path(self.config.model_dir).expanduser())

        log.info(
            "whisper-turbo 모델 로드 시작",
            model=_TURBO_MODEL,
            device=self._device,
            compute_type=self._compute_type,
        )

        load_attempts = build_load_attempts(self._device, self._compute_type)

        for device, compute_type in load_attempts:
            try:
                self._model = WhisperModel(
                    _TURBO_MODEL,
                    device=device,
                    compute_type=compute_type,
                    download_root=model_dir,
                )
                self._device = device
                self._compute_type = compute_type
                self.is_loaded = True
                log.info(
                    "whisper-turbo 모델 로드 완료",
                    model=_TURBO_MODEL,
                    device=device,
                    compute_type=compute_type,
                )
                return
            except Exception as e:
                log.warning(
                    "모델 로드 실패, 폴백 시도",
                    device=device,
                    compute_type=compute_type,
                    error=str(e),
                )

        raise ModelLoadError(
            f"whisper-turbo 모델 로드 실패: model={_TURBO_MODEL}, "
            f"모든 디바이스/compute_type 조합 시도 완료"
        )

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

            log.info("whisper-turbo 모델 언로드 완료")

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """whisper-turbo로 음성을 텍스트로 변환한다.

        Args:
            audio_path: 16kHz mono WAV 파일 경로.
            language: 언어 코드. None이면 자동 감지.

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

        lang = language or (self.config.source_lang if self.config.source_lang != "auto" else None)

        log.info(
            "whisper-turbo 추론 시작",
            audio=str(path),
            language=lang or "auto",
            beam_size=self.config.beam_size,
        )

        try:
            segments_iter, info = self._model.transcribe(
                str(path),
                language=lang,
                beam_size=self.config.beam_size,
                temperature=self.config.temperature,
                word_timestamps=True,
                vad_filter=False,
            )

            detected_lang = info.language if info.language else (lang or "unknown")
            duration = info.duration if info.duration else 0.0

            return self._convert_result(
                segments_iter=segments_iter,
                audio_path=path,
                language=detected_lang,
                duration=duration,
            )

        except Exception as e:
            if isinstance(e, TranscriptionError):
                raise
            raise TranscriptionError(f"whisper-turbo 추론 실패: {e}") from e

    def _convert_result(
        self,
        segments_iter: Any,
        audio_path: Path,
        language: str,
        duration: float,
    ) -> TranscriptionResult:
        """faster-whisper 결과를 TranscriptionResult로 변환한다."""

        segments: list[TranscriptionSegment] = []

        for idx, seg in enumerate(segments_iter, start=1):
            words = _extract_words(seg)

            seg_confidence = _logprob_to_confidence(seg.avg_logprob)
            if words:
                word_confs = [w.confidence for w in words if w.confidence > 0]
                if word_confs:
                    seg_confidence = sum(word_confs) / len(word_confs)

            segments.append(
                TranscriptionSegment(
                    index=idx,
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    confidence=seg_confidence,
                    words=words,
                )
            )

        return TranscriptionResult(
            metadata=Metadata(
                source_file=str(audio_path),
                duration_s=duration,
                language=language,
                model=f"whisper-turbo/{_TURBO_MODEL}",
                created_at=datetime.now(tz=UTC).isoformat(),
                subwyrhta_version=__version__,
            ),
            segments=segments,
        )


def _extract_words(segment: Any) -> list[WordSegment]:
    """세그먼트에서 단어 목록을 추출한다."""
    words: list[WordSegment] = []
    if not hasattr(segment, "words") or segment.words is None:
        return words

    for w in segment.words:
        words.append(
            WordSegment(
                word=w.word.strip(),
                start=w.start,
                end=w.end,
                confidence=w.probability if hasattr(w, "probability") else 0.0,
            )
        )
    return words


def _logprob_to_confidence(avg_log_prob: float) -> float:
    """avg_log_prob를 0~1 신뢰도로 변환한다."""
    import math

    try:
        return min(1.0, max(0.0, math.exp(avg_log_prob)))
    except (OverflowError, ValueError):
        return 0.0
