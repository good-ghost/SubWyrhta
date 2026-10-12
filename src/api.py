"""SubWyrhta Python API.

독립 라이브러리로 활용 가능한 프로그래밍 인터페이스.

사용 예시::

    from src.api import SubWyrhta, TranscribeConfig

    stt = SubWyrhta(model="whisperx", model_size="large-v3")
    result = stt.transcribe("video.mp4", TranscribeConfig(source_lang="en"))
    save_result(result, "output.srt")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.config import AppConfig
from src.core.logging import setup_logging
from src.pipeline import PipelineOptions, TranscriptionPipeline

if TYPE_CHECKING:
    from src.models.result import TranscriptionResult

log = structlog.get_logger(__name__)


@dataclass
class TranscribeConfig:
    """transcribe 호출 시 상세 설정.

    값이 None 인 항목은 덮어쓰지 않는다: 설정 파일(``SubWyrhta(config_path=…)``) 값, 없으면 코드 기본값을 쓴다.

    Attributes:
        source_lang: 음성 언어 ('auto' / 'en' / 'ko' / 'ja' / 'zh' / ...).
        diarize: 화자 분리 활성화 여부.
        hf_token: HuggingFace 토큰 (화자 분리용).
        num_speakers: 화자 수 ('auto' 또는 정수 문자열).
        segmentation_mode: 세그멘테이션 모드 ('basic' / 'advanced' / 'extreme' / 'llm-assisted').
        max_cps: 최대 CPS. 지정하지 않으면 설정 값(없으면 언어별 자동).
        max_duration_s: 자막 최대 표시 시간(초).
        min_duration_s: 자막 최소 표시 시간(초).
        denoise: 노이즈 억제 활성화 여부.
        keep_audio: 추출 오디오 보존 여부.
        audio_track: 오디오 트랙 인덱스.
        start: 처리 시작 시간.
        end: 처리 종료 시간.
        output_dir: 출력 디렉토리. 이것이나 output_format 을 지정하면 자막 파일도 쓴다.
        output_format: 출력 포맷 ('srt' / 'ass' / 'vtt' / 'json'). 지정하면 자막 파일도 쓴다.
    """

    source_lang: str | None = None
    diarize: bool | None = None
    hf_token: str | None = None
    num_speakers: str | None = None
    segmentation_mode: str | None = None
    max_cps: float | None = None
    max_duration_s: float | None = None
    min_duration_s: float | None = None
    denoise: bool | None = None
    keep_audio: bool = False
    audio_track: int | None = None
    start: str | None = None
    end: str | None = None
    output_dir: str | None = None
    output_format: str | None = None

    @property
    def writes_files(self) -> bool:
        """출력 위치나 포맷을 지정했으면 자막 파일도 쓴다."""
        return self.output_dir is not None or self.output_format is not None


class SubWyrhta:
    """SubWyrhta 메인 API 클래스.

    인자를 지정하지 않으면(None) 설정 파일 값, 없으면 코드 기본값을 쓴다.

    Args:
        model: STT 엔진 ('whisperx' / 'faster-whisper' / 'whisper-turbo' / 'parakeet' / 'distil-whisper').
        model_size: 모델 크기 ('large-v3' / 'small' / 'tiny' 등).
        device: 디바이스 ('auto' / 'cuda' / 'cpu' / 'mps').
        compute_type: compute_type ('auto' / 'float16' / 'float32' / 'int8').
        config_path: YAML 설정 파일 경로.
        log_level: 로깅 레벨.
    """

    def __init__(
        self,
        model: str | None = None,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        config_path: str | Path | None = None,
        log_level: str = "WARNING",
    ) -> None:
        setup_logging(level=log_level)
        self._model = model
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._config_path = config_path

    def transcribe(
        self,
        input_path: str | Path,
        config: TranscribeConfig | None = None,
    ) -> TranscriptionResult:
        """음성/영상 파일을 자막으로 변환한다.

        ``config.output_dir`` 이나 ``config.output_format`` 을 지정하면 자막 파일도 쓴다
        (파일명 규칙은 CLI 와 같다). 지정하지 않으면 결과만 돌려준다 (저장은 ``save_result``).

        Args:
            input_path: 입력 파일 경로.
            config: 상세 설정. None이면 설정 파일·코드 기본값.

        Returns:
            TranscriptionResult 인스턴스.
        """
        if config is None:
            config = TranscribeConfig()

        app_config = self._build_config(config)
        options = PipelineOptions(
            keep_audio=config.keep_audio,
            audio_track=config.audio_track,
            start=config.start,
            end=config.end,
            write=config.writes_files,
        )
        with TranscriptionPipeline(app_config) as pipeline:
            outcome = pipeline.run(input_path, options)

        log.info(
            "transcribe 완료",
            segments=outcome.result.segment_count,
            outputs=[str(path) for path in outcome.outputs],
        )
        return outcome.result

    def _build_config(self, config: TranscribeConfig) -> AppConfig:
        """지정한 값만 설정 파일 위에 덮어써 AppConfig 를 만든다."""
        candidates: dict[str, Any] = {
            "stt.model": self._model,
            "stt.model_size": self._model_size,
            "stt.device": self._device,
            "stt.compute_type": self._compute_type,
            "stt.source_lang": config.source_lang,
            "segmentation.mode": config.segmentation_mode,
            "segmentation.max_cps": config.max_cps,
            "segmentation.max_duration_s": config.max_duration_s,
            "segmentation.min_duration_s": config.min_duration_s,
            "output.dir": config.output_dir,
            "output.format": config.output_format,
            "diarization.enabled": config.diarize,
            "diarization.num_speakers": config.num_speakers,
            "diarization.hf_token": config.hf_token,
            "audio.denoise.enabled": config.denoise,
        }
        overrides = {key: value for key, value in candidates.items() if value is not None}

        return AppConfig.load(
            config_path=self._config_path,
            cli_overrides=overrides,
        )


def save_result(
    result: TranscriptionResult,
    output_path: str | Path,
) -> Path:
    """결과를 파일로 저장한다.

    확장자에 따라 적절한 Writer를 선택한다.

    Args:
        result: TranscriptionResult 인스턴스.
        output_path: 출력 파일 경로 (.srt / .ass / .vtt / .json).

    Returns:
        저장된 파일의 Path.
    """
    from src.writers.writer_factory import create_writer

    path = Path(output_path)
    ext = path.suffix.lstrip(".")

    # 확장자 → 포맷 매핑
    format_map: dict[str, str] = {
        "srt": "srt",
        "ass": "ass",
        "ssa": "ass",
        "vtt": "vtt",
        "json": "json",
    }
    fmt = format_map.get(ext, "srt")

    writer = create_writer(fmt)
    return writer.write(result, path)
