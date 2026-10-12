"""작업별 옵션: 웹 화면에서 작업을 등록할 때 설정 파일 기본값을 덮어쓰는 값.

옵션은 작업 기록에 JSON 으로 남고, 작업을 실행할 때 dot-notation 오버라이드로 바뀐다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.engines.engine_factory import get_available_engines
from src.segmentation.segmentation_factory import get_available_modes
from src.writers.writer_factory import get_available_formats

if TYPE_CHECKING:
    from src.core.config import AppConfig

MODEL_SIZES: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "turbo",
    "distil-large-v3",
    "distil-medium.en",
    "distil-small.en",
)
COMPUTE_TYPES: tuple[str, ...] = ("auto", "float16", "float32", "int8")
DEVICES: tuple[str, ...] = ("auto", "cuda", "cpu", "mps")
# 화면의 언어 선택지 ('auto' = 자동 감지). 엔진은 이 밖의 ISO 639-1 코드도 받는다
LANGUAGES: tuple[str, ...] = (
    "auto", "en", "ko", "ja", "zh", "es", "fr", "de", "it", "pt", "ru", "vi", "th", "id", "ar", "hi",
)  # fmt: skip
MAX_CPS_LIMIT = 60.0
# 오디오 트랙 번호(0-based) 상한
MAX_AUDIO_TRACK = 31
MIN_LANG_LEN = 2
MAX_LANG_LEN = 3


class JobOptions(BaseModel):
    """작업 하나의 처리 옵션."""

    engine: str
    model_size: str
    source_lang: str = "auto"
    compute_type: str = "auto"
    formats: list[str] = Field(min_length=1)
    segmentation_mode: str = "advanced"
    max_cps: float | None = Field(default=None, gt=0, le=MAX_CPS_LIMIT)
    pre_vad: bool = True
    normalize: bool = True
    denoise: bool = False
    separate_vocals: bool = False
    diarize: bool = False
    num_speakers: str = "auto"
    restore_punctuation: bool = False
    llm_correction: bool = False
    llm_punctuation: bool = False
    llm_review: bool = False
    quality_report: bool = False
    export_words: bool = False
    # 오디오 트랙 (0-based). None 이면 자동: 언어에 맞는 트랙, 없으면 기본 트랙
    audio_track: int | None = Field(default=None, ge=0, le=MAX_AUDIO_TRACK)
    # 검수 리포트 메모를 쓸 언어: 작업을 등록한 화면의 언어. None 이면 설정 파일 값 (postprocess.review_language)
    review_language: Literal["ko", "en"] | None = None

    @field_validator("engine")
    @classmethod
    def _engine(cls, value: str) -> str:
        return one_of(value, get_available_engines())

    @field_validator("model_size")
    @classmethod
    def _model_size(cls, value: str) -> str:
        return one_of(value, MODEL_SIZES)

    @field_validator("compute_type")
    @classmethod
    def _compute_type(cls, value: str) -> str:
        return one_of(value, COMPUTE_TYPES)

    @field_validator("segmentation_mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        return one_of(value, get_available_modes())

    @field_validator("source_lang")
    @classmethod
    def _language(cls, value: str) -> str:
        return language_code(value)

    @field_validator("formats")
    @classmethod
    def _formats(cls, value: list[str]) -> list[str]:
        available = get_available_formats()
        for fmt in value:
            one_of(fmt, available)
        return list(dict.fromkeys(value))

    @field_validator("num_speakers")
    @classmethod
    def _speakers(cls, value: str) -> str:
        text = value.strip().lower()
        if text != "auto" and not (text.isdigit() and int(text) >= 1):
            raise ValueError("invalid_speakers")
        return text

    def to_overrides(self) -> dict[str, Any]:
        """설정 dot-notation 오버라이드 (AppConfig.load 의 cli_overrides)."""
        return {
            "stt.model": self.engine,
            "stt.model_size": self.model_size,
            "stt.source_lang": self.source_lang,
            "stt.compute_type": self.compute_type,
            "output.format": self.formats[0],
            "output.export_words": self.export_words,
            "segmentation.mode": self.segmentation_mode,
            "segmentation.max_cps": self.max_cps,
            "audio.vad.pre_vad": self.pre_vad,
            "audio.normalize": self.normalize,
            "audio.denoise.enabled": self.denoise,
            "audio.separate_vocals": self.separate_vocals,
            "diarization.enabled": self.diarize,
            "diarization.num_speakers": self.num_speakers,
            "postprocess.restore_punctuation": self.restore_punctuation,
            "postprocess.llm_correction": self.llm_correction,
            "postprocess.llm_punctuation": self.llm_punctuation,
            "postprocess.llm_review": self.llm_review,
            "postprocess.review_language": self.review_language,
            "quality.export_report": self.quality_report,
        }

    @classmethod
    def defaults(cls, config: AppConfig) -> JobOptions:
        """설정 파일 값으로 채운 기본 옵션 (새 작업 화면의 처음 값).

        Raises:
            ValidationError: 설정 파일에 화면 선택지 밖의 값이 있을 때.
        """
        return cls.model_validate(default_values(config))


def default_values(config: AppConfig) -> dict[str, Any]:
    """설정 파일 값을 작업 옵션 모양으로 (검증하지 않는다)."""
    return {
        "engine": config.stt.model,
        "model_size": config.stt.model_size,
        "source_lang": config.stt.source_lang,
        "compute_type": config.stt.compute_type,
        "formats": [config.output.format],
        "segmentation_mode": config.segmentation.mode,
        "max_cps": config.segmentation.max_cps,
        "pre_vad": config.audio.vad.enabled and config.audio.vad.pre_vad,
        "normalize": config.audio.normalize,
        "denoise": config.audio.denoise.enabled,
        "separate_vocals": config.audio.separate_vocals,
        "diarize": config.diarization.enabled,
        "num_speakers": str(config.diarization.num_speakers),
        "restore_punctuation": config.postprocess.restore_punctuation,
        "llm_correction": config.postprocess.llm_correction,
        "llm_punctuation": config.postprocess.llm_punctuation,
        "llm_review": config.postprocess.llm_review,
        "quality_report": config.quality.export_report,
        "export_words": config.output.export_words,
        "audio_track": None,
        # 화면이 작업을 등록할 때 지금 화면 언어를 넣는다
        "review_language": None,
    }


def one_of(value: str, allowed: list[str] | tuple[str, ...]) -> str:
    """value 가 allowed 안에 있으면 그대로, 아니면 ValueError (pydantic 검증기에서 쓴다)."""
    if value not in allowed:
        raise ValueError(f"invalid_choice: {', '.join(allowed)}")
    return value


def language_code(value: str) -> str:
    """'auto' 또는 ISO 639-1/2 언어 코드로 정리한다."""
    code = value.strip().lower()
    if code != "auto" and not (code.isalpha() and MIN_LANG_LEN <= len(code) <= MAX_LANG_LEN):
        raise ValueError("invalid_language")
    return code
