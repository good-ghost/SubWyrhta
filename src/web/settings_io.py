"""설정 화면: config.yaml 의 편집 가능한 값 읽기·검증·저장.

화면에서 바꿀 수 있는 값만 pydantic 모델로 검증한다. 비밀 값(``diarization.hf_token``)과 웹 서버
바인딩(``web.host``·``web.port``·``web.static_dir``), 출력 폴더(``output.dir``, 작업마다 정해진다)는
화면에 내보내지 않고 파일에 있는 값을 그대로 둔다. 저장은 ruamel.yaml 로 주석을 유지한다.
"""

from __future__ import annotations

import dataclasses
import os
import re
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from src.core.config import AppConfig
from src.engines.engine_factory import get_available_engines
from src.segmentation.llm_client import LLM_PROVIDERS, endpoint_required, is_cli_provider
from src.segmentation.segmentation_factory import get_available_modes
from src.web.job_options import COMPUTE_TYPES, DEVICES, MODEL_SIZES, language_code, one_of
from src.writers.writer_factory import get_available_formats

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger(__name__)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SttSettings(_Section):
    """STT 엔진."""

    model: str
    model_size: str
    model_dir: str = Field(min_length=1)
    device: str
    compute_type: str
    batch_size: int = Field(ge=1, le=256)
    beam_size: int = Field(ge=1, le=20)
    temperature: float = Field(ge=0.0, le=1.0)
    source_lang: str

    @field_validator("model")
    @classmethod
    def _model(cls, value: str) -> str:
        return one_of(value, get_available_engines())

    @field_validator("model_size")
    @classmethod
    def _size(cls, value: str) -> str:
        return one_of(value, MODEL_SIZES)

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        return one_of(value, DEVICES)

    @field_validator("compute_type")
    @classmethod
    def _compute(cls, value: str) -> str:
        return one_of(value, COMPUTE_TYPES)

    @field_validator("source_lang")
    @classmethod
    def _lang(cls, value: str) -> str:
        return language_code(value)


class VadSettings(_Section):
    """VAD·Pre-VAD."""

    enabled: bool
    threshold: float = Field(gt=0.0, lt=1.0)
    min_speech_ms: int = Field(ge=0, le=10_000)
    min_silence_ms: int = Field(ge=0, le=10_000)
    pre_vad: bool
    merge_gap_s: float = Field(ge=0.0, le=10.0)
    min_chunk_s: float = Field(ge=0.0, le=60.0)
    max_chunk_s: float = Field(gt=0.0, le=600.0)

    @model_validator(mode="after")
    def _chunks(self) -> VadSettings:
        if self.min_chunk_s >= self.max_chunk_s:
            raise ValueError("min_chunk_lt_max_chunk")
        return self


class DenoiseSettings(_Section):
    """노이즈 억제."""

    enabled: bool
    model: Literal["noisereduce"]


class AudioSettings(_Section):
    """오디오 전처리."""

    normalize: bool
    separate_vocals: bool
    denoise: DenoiseSettings
    vad: VadSettings


class SegmentationSettings(_Section):
    """세그멘테이션."""

    mode: str
    max_chars_per_line: int | None = Field(default=None, ge=10, le=200)
    max_lines: int = Field(ge=1, le=4)
    max_cps: float | None = Field(default=None, gt=0.0, le=60.0)
    min_duration_s: float = Field(gt=0.0, le=10.0)
    max_duration_s: float = Field(gt=0.0, le=30.0)
    min_gap_s: float = Field(ge=0.0, le=2.0)
    force_split_after_s: float = Field(gt=0.0, le=30.0)
    dynamic_threshold: bool
    dynamic_threshold_start_s: float = Field(ge=0.0, le=30.0)
    dynamic_threshold_min_ratio: float = Field(gt=0.0, le=1.0)
    speaker_split: bool
    overlap_text_separator: str
    min_chars: int = Field(ge=0, le=50)

    @field_validator("mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        return one_of(value, get_available_modes())

    @model_validator(mode="after")
    def _durations(self) -> SegmentationSettings:
        # 시간 제약 순서: force_split_after_s < max_duration_s, min_duration_s < max_duration_s
        if self.force_split_after_s >= self.max_duration_s:
            raise ValueError("force_split_lt_max_duration")
        if self.min_duration_s >= self.max_duration_s:
            raise ValueError("min_duration_lt_max_duration")
        return self


class LlmSettings(_Section):
    """LLM (llm-assisted 세그멘테이션)."""

    provider: str
    # 구독 CLI(claude-code, codex)는 비우면 CLI 기본 모델이다
    model: str
    # 로컬 서버(ollama, local)만 쓴다. 클라우드 프로바이더는 고정 주소라 남은 값은 무시한다
    base_url: str
    timeout_s: float = Field(gt=0.0, le=600.0)
    max_retries: int = Field(ge=0, le=10)
    rpm: int = Field(ge=0, le=10_000)
    disable_thinking: bool = False
    reasoning_effort: Literal["", "low", "medium", "high"] = ""

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        return one_of(value, LLM_PROVIDERS)

    @field_validator("base_url")
    @classmethod
    def _base_url(cls, value: str) -> str:
        url = value.strip()
        if url and not re.fullmatch(r"https?://\S+", url):
            raise ValueError("invalid_url")
        return url

    @model_validator(mode="after")
    def _required(self) -> LlmSettings:
        if not self.model.strip() and not is_cli_provider(self.provider):
            raise ValueError("llm_model_required")
        if endpoint_required(self.provider) and not self.base_url:
            raise ValueError("llm_base_url_required")
        return self


class DiarizationSettings(_Section):
    """화자 분리 (토큰은 HF_TOKEN 환경 변수로만 받는다)."""

    enabled: bool
    num_speakers: str

    @field_validator("num_speakers")
    @classmethod
    def _speakers(cls, value: str) -> str:
        text = value.strip().lower()
        if text != "auto" and not (text.isdigit() and int(text) >= 1):
            raise ValueError("invalid_speakers")
        return text


class NoiseFilterSettings(_Section):
    """노이즈 필터."""

    enabled: bool
    languages: list[str]
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    remove_low_confidence: bool
    remove_interjections: bool = True
    custom_patterns: list[str]

    @field_validator("custom_patterns")
    @classmethod
    def _patterns(cls, value: list[str]) -> list[str]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"invalid_regex: {pattern}") from e
        return value


class PostprocessSettings(_Section):
    """후처리."""

    restore_punctuation: bool
    punctuation_model: bool
    llm_correction: bool = False
    llm_punctuation: bool = False
    llm_review: bool = False


class SceneSettings(_Section):
    """장면 감지."""

    enabled: bool
    gap_s: float = Field(gt=0.0, le=600.0)
    min_segments: int = Field(ge=1, le=100)
    boundary_gap_s: float = Field(ge=0.0, le=5.0)
    use_video: bool


class QualitySettings(_Section):
    """품질 관리."""

    enabled: bool
    auto_fix: bool
    export_report: bool


class OutputSettings(_Section):
    """출력 (폴더는 작업마다 정해진다)."""

    format: str
    export_words: bool

    @field_validator("format")
    @classmethod
    def _format(cls, value: str) -> str:
        return one_of(value, get_available_formats())


class LoggingSettings(_Section):
    """로그."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class WebSettings(_Section):
    """웹 화면 (호스트·포트는 시작 옵션이라 여기서 바꾸지 않는다)."""

    session_hours: int = Field(ge=1, le=24 * 30)
    max_upload_mb: int = Field(ge=1, le=1024 * 1024)
    delete_input_after_done: bool


class EditableSettings(_Section):
    """설정 화면에서 바꿀 수 있는 값 전체."""

    stt: SttSettings
    audio: AudioSettings
    segmentation: SegmentationSettings
    llm: LlmSettings
    diarization: DiarizationSettings
    noise_filter: NoiseFilterSettings
    postprocess: PostprocessSettings
    scene: SceneSettings
    quality: QualitySettings
    output: OutputSettings
    logging: LoggingSettings
    web: WebSettings


# 검증 오류 문구는 코드다 (화면이 validation.<코드> 로 번역한다). pydantic 이 "Value error, <코드>" 로 감싼다


class SettingsError(ValueError):
    """설정 저장 실패. code 는 화면이 문구로 바꾼다 (errors 는 칸별 검증 오류)."""

    def __init__(self, code: str, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.errors = errors or []


def editable_values(config: AppConfig) -> dict[str, Any]:
    """설정에서 화면에 보일 값만 뽑는다."""
    data = dataclasses.asdict(config)
    fields = EditableSettings.model_fields
    result: dict[str, Any] = {}
    for section, info in fields.items():
        model = info.annotation
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError(f"설정 섹션 모델이 아닙니다: {section}")
        result[section] = _pick(data.get(section, {}), model)
    return result


def _pick(data: dict[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    picked: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        value = data.get(name)
        nested = info.annotation
        if isinstance(nested, type) and issubclass(nested, BaseModel) and isinstance(value, dict):
            picked[name] = _pick(value, nested)
        else:
            picked[name] = value
    return picked


def validate_values(values: dict[str, Any]) -> EditableSettings:
    """화면이 보낸 값을 검증한다.

    Raises:
        SettingsError: ``invalid_settings`` (칸별 오류 포함).
    """
    try:
        return EditableSettings.model_validate(values)
    except ValidationError as e:
        errors = [
            {
                "loc": ".".join(str(part) for part in err["loc"]),
                "message": err["msg"],
                "type": err["type"],
                "ctx": {k: str(v) for k, v in (err.get("ctx") or {}).items()},
            }
            for err in e.errors()
        ]
        raise SettingsError("invalid_settings", errors) from e


def is_writable(path: Path) -> bool:
    """설정 파일(또는 없으면 그 폴더)에 쓸 수 있는지."""
    if path.exists():
        return os.access(path, os.W_OK)
    return os.access(path.parent, os.W_OK)


def save_settings(path: Path, settings: EditableSettings) -> None:
    """검증한 값을 config.yaml 에 쓴다. 화면에 없는 키와 주석은 그대로 둔다.

    Raises:
        SettingsError: ``read_only`` (쓸 수 없는 파일).
    """
    if not is_writable(path):
        raise SettingsError("read_only")
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    document: Any = CommentedMap()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            loaded = yaml.load(f)
        if isinstance(loaded, CommentedMap):
            document = loaded
    _merge(document, settings.model_dump())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.dump(document, f)
    os.replace(tmp, path)
    log.info("설정 저장", path=str(path))


def _merge(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, dict):
            node = target.get(key)
            if not isinstance(node, dict):
                node = CommentedMap()
                target[key] = node
            _merge(node, value)
        else:
            target[key] = value


def ensure_config_file(path: Path) -> None:
    """설정 파일이 없으면 기본값으로 만든다 (컨테이너 첫 실행)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = dataclasses.asdict(AppConfig())
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    document = CommentedMap(defaults)
    document.yaml_set_start_comment("SubWyrhta 설정 파일 (웹 화면의 설정에서 바꿀 수 있다)")
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(document, f)
    log.info("기본 설정 파일 생성", path=str(path))
