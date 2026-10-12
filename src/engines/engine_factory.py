"""STT 엔진 팩토리.

설정에 따라 적절한 STT 엔진 인스턴스를 생성한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.core.exceptions import ConfigError

if TYPE_CHECKING:
    from src.core.config import STTConfig
    from src.engines.base import STTEngine

log = structlog.get_logger(__name__)

# 엔진 이름 → (모듈 경로, 클래스 이름) 매핑
# 지연 임포트로 불필요한 의존성 로드를 방지한다
_ENGINE_REGISTRY: dict[str, tuple[str, str]] = {
    "whisperx": ("src.engines.whisperx_engine", "WhisperXEngine"),
    "faster-whisper": ("src.engines.faster_whisper_engine", "FasterWhisperEngine"),
    "whisper-turbo": ("src.engines.whisper_turbo_engine", "WhisperTurboEngine"),
    "parakeet": ("src.engines.parakeet_engine", "ParakeetEngine"),
}

# distil-whisper는 faster-whisper 엔진으로 실행
_ENGINE_ALIASES: dict[str, str] = {
    "distil-whisper": "faster-whisper",
}


def get_available_engines() -> list[str]:
    """사용 가능한 엔진 이름 목록을 반환한다."""
    return sorted(set(_ENGINE_REGISTRY.keys()) | set(_ENGINE_ALIASES.keys()))


def create_engine(config: STTConfig) -> STTEngine:
    """설정에 따라 STT 엔진 인스턴스를 생성한다.

    Args:
        config: STT 엔진 설정. config.model로 엔진 종류를 결정한다.

    Returns:
        STTEngine 구현체 인스턴스.

    Raises:
        ConfigError: 알 수 없는 엔진 이름일 때.
    """
    engine_name = config.model.lower()

    # 별칭 해소
    engine_name = _ENGINE_ALIASES.get(engine_name, engine_name)

    if engine_name not in _ENGINE_REGISTRY:
        available = ", ".join(get_available_engines())
        raise ConfigError(f"알 수 없는 STT 엔진: {config.model!r}. 사용 가능: {available}")

    module_path, class_name = _ENGINE_REGISTRY[engine_name]

    try:
        import importlib

        module = importlib.import_module(module_path)
        engine_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ConfigError(
            f"STT 엔진 {config.model!r}을(를) 불러올 수 없습니다. "
            f"필요한 패키지가 설치되어 있는지 확인해 주세요: {e}"
        ) from e

    log.info("STT 엔진 생성", engine=engine_name, model_size=config.model_size)
    return engine_class(config)  # type: ignore[no-any-return]
