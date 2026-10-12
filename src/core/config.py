"""설정 로더: YAML 파일 로드 + 환경변수 오버라이드 + CLI 인자 병합."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ──────────────────────────────────────────────
# 하위 설정 데이터클래스
# ──────────────────────────────────────────────


@dataclass
class STTConfig:
    """STT 엔진 설정."""

    model: str = "whisperx"
    model_size: str = "large-v3"
    model_dir: str = "~/.cache/subwyrhta"
    device: str = "auto"
    compute_type: str = "auto"
    batch_size: int = 16
    beam_size: int = 5
    temperature: float = 0.0
    source_lang: str = "auto"


@dataclass
class VADConfig:
    """VAD 설정."""

    enabled: bool = True
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 100
    pre_vad: bool = True
    merge_gap_s: float = 0.5
    min_chunk_s: float = 1.0
    max_chunk_s: float = 30.0


@dataclass
class DenoiseConfig:
    """노이즈 억제 설정."""

    enabled: bool = False
    model: str = "noisereduce"


@dataclass
class AudioConfig:
    """오디오 전처리 설정."""

    normalize: bool = True
    vad: VADConfig = field(default_factory=VADConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    separate_vocals: bool = False


@dataclass
class SegmentationConfig:
    """세그멘테이션 설정."""

    mode: str = "advanced"
    # None 이면 언어별 기본값 (ja·zh 16, 그 밖 42. speed_optimizer.CHARS_PER_LINE_DEFAULTS)
    max_chars_per_line: int | None = None
    max_lines: int = 2
    max_cps: float | None = None
    min_duration_s: float = 1.0
    max_duration_s: float = 7.0
    min_gap_s: float = 0.04
    force_split_after_s: float = 5.0
    dynamic_threshold: bool = True
    dynamic_threshold_start_s: float = 3.0
    dynamic_threshold_min_ratio: float = 0.4
    speaker_split: bool = True
    overlap_text_separator: str = " / "
    min_chars: int = 2


@dataclass
class LLMConfig:
    """LLM 프로바이더 설정 (프로바이더 목록: src/segmentation/llm_client.py LLM_PROVIDERS)."""

    # OpenAI·Claude 는 구독(codex, claude-code)만, 키는 정해진 환경 변수(.env)로만 받는다
    provider: str = "claude-code"
    # 구독 CLI(claude-code, codex)는 비우면 CLI 기본 모델
    model: str = ""
    # 로컬 서버(ollama, local)만 쓴다. 클라우드 프로바이더는 고정 주소
    base_url: str = ""
    timeout_s: float = 30.0
    max_retries: int = 3
    # 분당 요청 수 한도 (재시도 포함). 0 이면 제한 없음
    rpm: int = 0
    # 추론(thinking) 끄기: OpenAI 호환 요청에 chat_template_kwargs.thinking=false 를 넣는다 (NIM·vLLM).
    # NIM deepseek-v4.1-flash 는 자막 20개 검토에 추론 토큰을 1만 개 넘게 써 묶음 하나가 5분을 넘었다
    disable_thinking: bool = False
    # 추론 수준 (OpenAI 호환 요청의 reasoning_effort: low / medium / high). 비우면 보내지 않는다.
    # NIM glm-5.3-flash 는 chat_template_kwargs 로 추론이 꺼지지 않고, 기본 추론으로는 문장 20개 묶음이
    # 300초를 넘었다. low 면 13초 (2026-10-11 측정)
    reasoning_effort: str = ""


@dataclass
class DiarizationConfig:
    """화자 분리 설정."""

    enabled: bool = False
    num_speakers: str = "auto"
    hf_token: str = ""
    speaker_map: dict[str, str] = field(default_factory=dict)


@dataclass
class NoiseFilterConfig:
    """노이즈 필터링 설정."""

    enabled: bool = True
    languages: list[str] = field(default_factory=lambda: ["en", "ja", "ko"])
    confidence_threshold: float = 0.7
    # 저신뢰도 단어를 지운다. 기본은 마킹만 (품질 리포트 저신뢰도 수, 2026-10-08 사용자 결정)
    remove_low_confidence: bool = False
    # 감탄사·신음만 있는 단어(ああ, んん, はぁ 등)를 지운다 (2026-10-09 사용자 결정)
    remove_interjections: bool = True
    custom_patterns: list[str] = field(default_factory=list)


@dataclass
class PostprocessConfig:
    """후처리 설정.

    구두점 복원은 STT 세그먼트 끝에 문장 부호를 붙인다. Whisper 계열은 구두점을 출력하므로
    기본값은 꺼 둔다 (구두점이 없는 엔진·언어에서 켠다, REQUIREMENT 7.3).
    """

    restore_punctuation: bool = False
    punctuation_model: bool = False
    # 세그멘테이션 전에 소리는 맞고 표기가 틀린 말을 LLM 으로 고치고 그 문장을 다시 정렬한다 (WI-9.018).
    # 용어 목록은 환경 변수 SUBWYRHTA_GLOSSARY. 기본은 끈다
    llm_correction: bool = False
    # 세그멘테이션 전에 STT 문장에 LLM 으로 문장 부호를 넣는다 (분할·줄바꿈이 부호를 쓴다). 기본은 끈다
    llm_punctuation: bool = False
    # 줄바꿈·문장 부호를 LLM(llm 섹션의 프로바이더)이 검토한다. 기본은 끈다 (호출 비용·시간)
    llm_review: bool = False
    # 검수 리포트의 메모·교정 이유를 쓸 언어 (ko, en …). 자막 언어가 아니라 리포트를 읽는 사람의 언어다.
    # 웹 작업은 작업을 등록한 화면의 언어를 넣는다 (작업 옵션 review_language)
    review_language: str = "ko"


@dataclass
class SceneConfig:
    """장면 감지 설정 (REQUIREMENT 10.2).

    자막 사이가 ``gap_s`` 이상 비면 장면 경계로 보고, 경계 앞뒤 자막 사이에 ``boundary_gap_s`` 를
    보장한다. ``min_segments`` 보다 작은 장면은 이웃과 합친다. ``use_video`` 는 PySceneDetect
    (scenedetect 패키지)로 영상의 컷도 경계로 쓴다 (영상 전체를 디코딩해 느리다).
    """

    enabled: bool = True
    gap_s: float = 6.0
    min_segments: int = 2
    boundary_gap_s: float = 0.5
    use_video: bool = False


@dataclass
class WebConfig:
    """웹 화면(serve 서브커맨드) 설정."""

    host: str = "127.0.0.1"
    port: int = 8950
    session_hours: int = 24
    max_upload_mb: int = 4096
    # 성공한 작업의 업로드 원본을 지운다 (자막만 남긴다). 실패한 작업은 다시 시도하도록 남긴다
    delete_input_after_done: bool = True
    static_dir: str = "./web/dist"


@dataclass
class QualityConfig:
    """품질 관리 설정."""

    enabled: bool = True
    auto_fix: bool = True
    export_report: bool = False


@dataclass
class OutputConfig:
    """출력 설정."""

    dir: str = "./output"
    format: str = "srt"
    export_words: bool = False


@dataclass
class LoggingConfig:
    """로깅 설정."""

    level: str = "INFO"


# ──────────────────────────────────────────────
# 최상위 설정
# ──────────────────────────────────────────────

# 비밀 값(토큰·API 키)을 담는 파일 (.gitignore 대상, 견본은 .env.example)
DEFAULT_ENV_FILE = ".env"

_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "HF_TOKEN": ("diarization.hf_token", str),
    "SUBWYRHTA_MODEL_DIR": ("stt.model_dir", str),
    "SUBWYRHTA_DEVICE": ("stt.device", str),
}

_NESTED_CONFIG_MAP: dict[str, type] = {
    "stt": STTConfig,
    "audio": AudioConfig,
    "segmentation": SegmentationConfig,
    "llm": LLMConfig,
    "diarization": DiarizationConfig,
    "noise_filter": NoiseFilterConfig,
    "postprocess": PostprocessConfig,
    "scene": SceneConfig,
    "quality": QualityConfig,
    "output": OutputConfig,
    "logging": LoggingConfig,
    "web": WebConfig,
}

_AUDIO_NESTED: dict[str, type] = {
    "vad": VADConfig,
    "denoise": DenoiseConfig,
}


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """딕셔너리를 데이터클래스 인스턴스로 변환한다. 알 수 없는 키는 무시한다."""
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


@dataclass
class AppConfig:
    """SubWyrhta 전체 설정.

    YAML 파일 → 환경변수 오버라이드 → CLI 인자 병합 순서로 적용한다.
    """

    stt: STTConfig = field(default_factory=STTConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    noise_filter: NoiseFilterConfig = field(default_factory=NoiseFilterConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    web: WebConfig = field(default_factory=WebConfig)

    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> AppConfig:
        """설정을 로드한다.

        Args:
            config_path: YAML 설정 파일 경로. None이면 기본값 사용.
            cli_overrides: CLI에서 전달된 오버라이드 딕셔너리.
                dot-notation 키 지원 (예: ``{"stt.device": "cpu"}``).

        Returns:
            병합된 AppConfig 인스턴스.
        """
        raw = _load_yaml(config_path)
        _apply_env_overrides(raw)
        if cli_overrides:
            _apply_cli_overrides(raw, cli_overrides)
        return _build_config(raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """``dataclasses.asdict`` 결과에서 설정을 되만든다 (작업 자식 프로세스에 넘길 때).

        환경변수 오버라이드는 다시 적용하지 않는다 (넘기기 전에 이미 적용됐다).

        Args:
            data: 섹션별 딕셔너리.

        Returns:
            AppConfig 인스턴스.
        """
        return _build_config(copy.deepcopy(data))


def _load_yaml(config_path: str | Path | None) -> dict[str, Any]:
    """YAML 파일을 로드하여 딕셔너리로 반환한다."""
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_env_file(path: Path | None = None) -> list[str]:
    """``.env`` 파일의 값을 환경 변수로 넣는다 (CLI·serve 시작 때 한 번).

    셸에 이미 있는 환경 변수는 바꾸지 않는다 (셸 값이 우선). 값이 빈 항목은 넣지 않는다:
    ``.env.example`` 을 복사해 일부만 채워도 ``SUBWYRHTA_DEVICE=`` 같은 빈 값이 설정을 덮어쓰지 않게 한다.
    serve 의 작업 프로세스는 서버 환경을 물려받으므로 같은 값을 쓴다.

    Args:
        path: 읽을 파일. None 이면 현재 폴더의 ``.env`` (없으면 아무것도 하지 않는다).

    Returns:
        새로 넣은 환경 변수 이름 (값은 돌려주지 않는다).
    """
    from dotenv import dotenv_values

    target = path or Path.cwd() / DEFAULT_ENV_FILE
    if not target.is_file():
        return []
    loaded: list[str] = []
    for key, value in dotenv_values(target).items():
        if value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _apply_env_overrides(raw: dict[str, Any]) -> None:
    """환경변수로 설정을 오버라이드한다."""
    for env_var, (dot_path, cast) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is not None:
            _set_nested(raw, dot_path, cast(value))


def _apply_cli_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    """CLI 인자를 dot-notation 키로 병합한다."""
    for key, value in overrides.items():
        if value is not None:
            _set_nested(raw, key, value)


def _set_nested(d: dict[str, Any], dot_path: str, value: Any) -> None:
    """dot-notation 경로에 값을 설정한다 (예: 'stt.device')."""
    keys = dot_path.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _build_config(raw: dict[str, Any]) -> AppConfig:
    """딕셔너리에서 AppConfig 인스턴스를 생성한다."""
    kwargs: dict[str, Any] = {}

    for section, cls in _NESTED_CONFIG_MAP.items():
        section_data = raw.get(section, {})
        if not isinstance(section_data, dict):
            section_data = {}

        # audio 섹션은 내부에 중첩 데이터클래스가 있다
        if section == "audio":
            for sub_key, sub_cls in _AUDIO_NESTED.items():
                sub_data = section_data.get(sub_key, {})
                if isinstance(sub_data, dict):
                    section_data[sub_key] = _dict_to_dataclass(sub_cls, sub_data)
            kwargs[section] = _dict_to_dataclass(cls, section_data)
        else:
            kwargs[section] = _dict_to_dataclass(cls, section_data)

    return AppConfig(**kwargs)
