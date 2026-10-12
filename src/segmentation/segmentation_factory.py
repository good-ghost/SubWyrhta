"""세그멘테이션 팩토리."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.core.exceptions import ConfigError

if TYPE_CHECKING:
    from src.core.config import LLMConfig, SegmentationConfig
    from src.segmentation.base import SegmentationStrategy

log = structlog.get_logger(__name__)

# 모드 이름 → (모듈 경로, 클래스 이름)
_STRATEGY_REGISTRY: dict[str, tuple[str, str]] = {
    "basic": ("src.segmentation.basic", "BasicSegmentation"),
    "advanced": ("src.segmentation.advanced", "AdvancedSegmentation"),
    "extreme": ("src.segmentation.extreme", "ExtremeSegmentation"),
    "llm-assisted": ("src.segmentation.llm_assisted", "LLMAssistedSegmentation"),
}


def get_available_modes() -> list[str]:
    """사용 가능한 세그멘테이션 모드 목록을 반환한다."""
    return sorted(_STRATEGY_REGISTRY.keys())


def create_segmentation(
    config: SegmentationConfig,
    llm_config: LLMConfig | None = None,
) -> SegmentationStrategy:
    """설정에 따라 세그멘테이션 전략 인스턴스를 생성한다.

    Args:
        config: 세그멘테이션 설정. config.mode로 전략을 결정한다.
        llm_config: LLM 설정 (llm-assisted 모드 전용).

    Returns:
        SegmentationStrategy 구현체 인스턴스.

    Raises:
        ConfigError: 알 수 없는 모드일 때.
    """
    mode = config.mode.lower()

    if mode not in _STRATEGY_REGISTRY:
        available = ", ".join(get_available_modes())
        raise ConfigError(f"알 수 없는 세그멘테이션 모드: {config.mode!r}. 사용 가능: {available}")

    module_path, class_name = _STRATEGY_REGISTRY[mode]

    try:
        import importlib

        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ConfigError(f"세그멘테이션 모드 {config.mode!r}을(를) 불러올 수 없습니다: {e}") from e

    log.info("세그멘테이션 전략 생성", mode=mode)

    if mode == "llm-assisted":
        return strategy_class(config, llm_config=llm_config)  # type: ignore[no-any-return]

    return strategy_class(config)  # type: ignore[no-any-return]
