"""WhisperX STT 엔진 구현.

배치 추론 + forced alignment(phoneme-level 정밀 타임스탬프 재정렬)을 지원한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import ModelLoadError, TranscriptionError
from src.engines.base import (
    ChunkTranscription,
    STTEngine,
    build_load_attempts,
    offset_segments,
    resolve_compute_type,
    resolve_device,
)
from src.models.result import Metadata, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from src.audio.vad import SpeechSegment
    from src.core.config import STTConfig

from src.version import __version__

log = structlog.get_logger(__name__)

# whisperx.load_audio 는 16kHz mono float32 배열을 돌려준다
_SAMPLE_RATE = 16_000


@dataclass
class _Aligner:
    """로드해 둔 정렬 모델. model 이 None 이면 로드에 실패해 정렬을 건너뛴다."""

    language: str
    model: Any
    metadata: Any


class WhisperXEngine(STTEngine):
    """WhisperX 기반 STT 엔진.

    배치 추론으로 GPU 활용을 극대화하고,
    forced alignment로 단어 단위 정밀 타임스탬프를 생성한다.
    """

    def __init__(self, config: STTConfig) -> None:
        super().__init__(config)
        self._model: Any = None
        self._device: str = ""
        self._compute_type: str = ""

    @property
    def engine_name(self) -> str:
        return "whisperx"

    def load_model(self) -> None:
        """WhisperX 모델을 로드한다.

        GPU 초기화 실패 시 CPU로 폴백한다.
        compute_type 호환 실패 시 대안으로 폴백한다.

        Raises:
            ModelLoadError: 모든 폴백 시도 후에도 실패 시.
        """
        try:
            import whisperx
        except ImportError as e:
            raise ModelLoadError(
                "whisperx 패키지가 설치되지 않았습니다. "
                "pip install -r requirements.txt 로 설치해 주세요."
            ) from e

        self._device = resolve_device(self.config.device)
        self._compute_type = resolve_compute_type(self.config.compute_type, self._device)

        model_dir = str(Path(self.config.model_dir).expanduser())

        log.info(
            "WhisperX 모델 로드 시작",
            model_size=self.config.model_size,
            device=self._device,
            compute_type=self._compute_type,
        )

        attempts = build_load_attempts(self._device, self._compute_type)

        for device, compute_type in attempts:
            try:
                self._model = whisperx.load_model(
                    self.config.model_size,
                    device=device,
                    compute_type=compute_type,
                    download_root=model_dir,
                    language=self.config.source_lang if self.config.source_lang != "auto" else None,
                )
                self._device = device
                self._compute_type = compute_type
                self.is_loaded = True
                log.info(
                    "WhisperX 모델 로드 완료",
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
            f"WhisperX 모델 로드 실패: model_size={self.config.model_size}, "
            f"모든 디바이스/compute_type 조합 시도 완료"
        )

    def unload_model(self) -> None:
        """모델을 언로드하여 메모리(GPU VRAM 포함)를 해제한다."""
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

            log.info("WhisperX 모델 언로드 완료")

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """WhisperX로 음성을 텍스트로 변환한다.

        배치 추론 후 forced alignment로 단어 단위 타임스탬프를 생성한다.

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

        try:
            import whisperx
        except ImportError as e:
            raise TranscriptionError("whisperx 패키지를 불러올 수 없습니다.") from e

        path = Path(audio_path)
        if not path.exists():
            raise TranscriptionError(f"오디오 파일이 존재하지 않습니다: {path}")

        lang = language or (self.config.source_lang if self.config.source_lang != "auto" else None)

        log.info("WhisperX 추론 시작", audio=str(path), language=lang or "auto")

        try:
            # 1. 오디오 로드
            audio = whisperx.load_audio(str(path))

            # 2. 배치 추론
            result = self._model.transcribe(
                audio,
                batch_size=self.config.batch_size,
                language=lang,
            )

            detected_lang = result.get("language", lang or "unknown")

            # 3. Forced alignment
            result = self._align(whisperx, audio, result, detected_lang)

            # 4. 결과 변환
            return self._convert_result(
                raw_result=result,
                audio_path=path,
                language=detected_lang,
            )

        except Exception as e:
            if isinstance(e, TranscriptionError):
                raise
            raise TranscriptionError(f"WhisperX 추론 실패: {e}") from e

    def transcribe_chunks(
        self,
        audio_path: str | Path,
        chunks: Sequence[tuple[int, SpeechSegment]],
        language: str | None = None,
    ) -> Iterator[ChunkTranscription]:
        """Pre-VAD 청크별로 추론한다 (오디오는 한 번 읽고, 정렬 모델은 언어마다 한 번 로드).

        기본 구현처럼 청크마다 ``transcribe``를 부르면 정렬 모델을 청크마다 다시 로드한다.
        2.0초 미만 청크(``skip_alignment``)는 forced alignment를 건너뛴다.

        Args:
            audio_path: 16kHz mono WAV 파일 경로.
            chunks: (청크 인덱스, 발화 구간) 목록.
            language: 언어 코드. None이면 첫 청크에서 감지한 언어를 이후 청크에 고정한다.

        Yields:
            청크별 추론 결과.

        Raises:
            TranscriptionError: 추론 실패 시.
        """
        if not self.is_loaded or self._model is None:
            raise TranscriptionError(
                "모델이 로드되지 않았습니다. load_model()을 먼저 호출해 주세요."
            )
        try:
            import whisperx
        except ImportError as e:
            raise TranscriptionError("whisperx 패키지를 불러올 수 없습니다.") from e

        path = Path(audio_path)
        try:
            audio = whisperx.load_audio(str(path))
        except Exception as e:
            raise TranscriptionError(f"WhisperX 오디오 로드 실패: {path} ({e})") from e

        lang = language
        aligner: _Aligner | None = None
        try:
            for index, chunk in chunks:
                piece = audio[int(chunk.start * _SAMPLE_RATE) : int(chunk.end * _SAMPLE_RATE)]
                try:
                    raw = self._model.transcribe(
                        piece, batch_size=self.config.batch_size, language=lang
                    )
                except Exception as e:
                    raise TranscriptionError(f"WhisperX 청크 추론 실패 (청크 {index}): {e}") from e
                detected = str(raw.get("language") or lang or "unknown")
                if lang is None and raw.get("segments"):
                    lang = detected
                if not chunk.skip_alignment and raw.get("segments"):
                    if aligner is None or aligner.language != detected:
                        aligner = self._load_aligner(whisperx, detected)
                    raw = self._align_with(whisperx, aligner, piece, raw)
                converted = self._convert_result(raw_result=raw, audio_path=path, language=detected)
                yield ChunkTranscription(
                    index=index,
                    segments=offset_segments(converted.segments, chunk.start),
                    language=detected,
                    model=converted.metadata.model,
                )
        finally:
            aligner = None
            self._release_cuda_cache()

    def _load_aligner(self, whisperx: Any, language: str) -> _Aligner | None:
        """언어별 정렬 모델을 로드한다. 실패하면 None (기본 타임스탬프 사용)."""
        try:
            model, metadata = whisperx.load_align_model(language_code=language, device=self._device)
        except Exception as e:
            log.warning(
                "정렬 모델 로드 실패, 기본 타임스탬프 사용", language=language, error=str(e)
            )
            return _Aligner(language=language, model=None, metadata=None)
        return _Aligner(language=language, model=model, metadata=metadata)

    def _align_with(
        self,
        whisperx: Any,
        aligner: _Aligner | None,
        audio: Any,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """로드해 둔 정렬 모델로 forced alignment를 수행한다. 실패하면 원래 결과."""
        if aligner is None or aligner.model is None:
            return result
        try:
            aligned: dict[str, Any] = whisperx.align(
                result["segments"],
                aligner.model,
                aligner.metadata,
                audio,
                self._device,
                return_char_alignments=False,
            )
        except Exception as e:
            log.warning("Forced alignment 실패, 기본 타임스탬프 사용", error=str(e))
            return result
        aligned.setdefault("language", result.get("language"))
        return aligned

    def _release_cuda_cache(self) -> None:
        """CUDA 캐시를 비운다 (정렬 모델을 해제한 뒤 VRAM을 돌려준다)."""
        if self._device != "cuda":
            return
        try:
            import torch
        except ImportError:
            return
        torch.cuda.empty_cache()

    def _align(
        self,
        whisperx: Any,
        audio: Any,
        result: dict[str, Any],
        language: str,
    ) -> dict[str, Any]:
        """Forced alignment을 수행하여 단어 단위 타임스탬프를 생성한다."""
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=language,
                device=self._device,
            )
            aligned: dict[str, Any] = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                self._device,
                return_char_alignments=False,
            )
            del align_model
            if self._device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            return aligned
        except Exception as e:
            log.warning(
                "Forced alignment 실패, 기본 타임스탬프 사용",
                language=language,
                error=str(e),
            )
            return result

    def _convert_result(
        self,
        raw_result: dict[str, Any],
        audio_path: Path,
        language: str,
    ) -> TranscriptionResult:
        """WhisperX 원시 결과를 TranscriptionResult로 변환한다."""
        segments: list[TranscriptionSegment] = []

        for idx, seg in enumerate(raw_result.get("segments", []), start=1):
            words = _extract_words(seg)

            seg_confidence = 0.0
            if words:
                confidences = [w.confidence for w in words if w.confidence > 0]
                if confidences:
                    seg_confidence = sum(confidences) / len(confidences)

            segments.append(
                TranscriptionSegment(
                    index=idx,
                    text=seg.get("text", "").strip(),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    confidence=seg_confidence,
                    speaker=seg.get("speaker"),
                    words=words,
                )
            )

        model_name = f"whisperx/{self.config.model_size}"

        return TranscriptionResult(
            metadata=Metadata(
                source_file=str(audio_path),
                language=language,
                model=model_name,
                created_at=datetime.now(tz=UTC).isoformat(),
                subwyrhta_version=__version__,
            ),
            segments=segments,
        )


def _extract_words(segment: dict[str, Any]) -> list[WordSegment]:
    """WhisperX 세그먼트에서 단어 목록을 추출한다."""
    words: list[WordSegment] = []
    for w in segment.get("words", []):
        if "start" not in w or "end" not in w:
            continue
        words.append(
            WordSegment(
                word=w.get("word", ""),
                start=float(w["start"]),
                end=float(w["end"]),
                confidence=float(w.get("score", w.get("confidence", 0.0))),
            )
        )
    return words
