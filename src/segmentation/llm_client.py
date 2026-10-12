"""LLM 프로바이더 클라이언트.

llm-assisted 세그멘테이션이 쓰는 LLM 호출을 한 인터페이스로 묶는다 (PHASE-PLAN-08).

- 구독: ChatGPT(Codex CLI ``codex exec``), Claude(Claude Code CLI ``claude -p``).
  서버에 설치·로그인한 공식 CLI 를 비대화 모드로 호출하고, 구독 로그인 정보는 SubWyrhta 가 다루지 않는다.
  OpenAI·Anthropic 은 구독만 허용한다 (API 키 방식 없음, 2026-10-09 사용자 결정).
- API 키 (클라우드, OpenAI 호환 엔드포인트 고정): Gemini, OpenRouter, NVIDIA NIM.
  키는 정해진 환경 변수(.env)로만 받는다.
- 로컬: Ollama(기본 주소, 바꿀 수 있다), 범용 OpenAI 호환 서버(엔드포인트 필수).
- 분당 요청 수(RPM) 제한: ``llm.rpm`` (0 이면 제한 없음). 재시도도 요청으로 센다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import structlog

from src.audio.extractor import stderr_tail
from src.core.exceptions import LLMSegmentationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import openai

    from src.core.config import LLMConfig

log = structlog.get_logger(__name__)

# ──────────────────────────────────────────────
# 프로바이더 목록
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class OpenAICompatiblePreset:
    """OpenAI 호환 엔드포인트 프리셋.

    Attributes:
        base_url: 기본 엔드포인트. 빈 문자열이면 설정의 ``base_url`` 이 필수다 (``local``).
        api_key_env: API 키 환경 변수. 빈 문자열이면 키를 쓰지 않는다 (로컬).
        endpoint_editable: 설정의 ``base_url`` 로 엔드포인트를 바꿀 수 있는가.
            클라우드 프로바이더는 고정 주소만 쓴다 (NVIDIA NIM 은 클라우드만 허용).
    """

    base_url: str
    api_key_env: str
    endpoint_editable: bool


OPENAI_COMPATIBLE_PRESETS: dict[str, OpenAICompatiblePreset] = {
    "gemini": OpenAICompatiblePreset(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        endpoint_editable=False,
    ),
    "openrouter": OpenAICompatiblePreset(
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", endpoint_editable=False
    ),
    "nvidia-nim": OpenAICompatiblePreset(
        "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", endpoint_editable=False
    ),
    "ollama": OpenAICompatiblePreset("http://localhost:11434/v1", "", endpoint_editable=True),
    "local": OpenAICompatiblePreset("", "", endpoint_editable=True),
}

# 구독 CLI 프로바이더 → 실행 파일. 모델을 비우면 CLI 기본 모델을 쓴다
CLI_EXECUTABLES: dict[str, str] = {"codex": "codex", "claude-code": "claude"}
CLI_PROVIDERS: tuple[str, ...] = tuple(CLI_EXECUTABLES)

# 화면 선택지 순서: 구독, 클라우드 API 키, 로컬
LLM_PROVIDERS: tuple[str, ...] = (*CLI_PROVIDERS, *OPENAI_COMPATIBLE_PRESETS)

# 키가 필요 없는 서버에 보낼 값 (openai SDK 는 키가 없으면 OPENAI_API_KEY 를 찾다가 실패한다)
_NO_KEY_PLACEHOLDER = "not-needed"

# 재시도해도 되는 HTTP 상태 (한도 초과·서버 오류)
_RETRYABLE_STATUS_MIN = 500
_RATE_LIMIT_STATUS = 429

# RPM 을 세는 구간
_RPM_WINDOW_S = 60.0


def is_openai_available() -> bool:
    """openai 패키지 설치 여부를 반환한다 (클라우드 API 키·로컬 프로바이더가 쓴다)."""
    try:
        import openai  # noqa: F401

        return True
    except ImportError:
        return False


def is_cli_provider(provider: str) -> bool:
    """구독 CLI 프로바이더인지 반환한다 (모델 칸을 비워도 된다)."""
    return provider in CLI_EXECUTABLES


def endpoint_required(provider: str) -> bool:
    """설정에 엔드포인트(``llm.base_url``)가 반드시 있어야 하는 프로바이더인가 (``local``)."""
    preset = OPENAI_COMPATIBLE_PRESETS.get(provider)
    return preset is not None and preset.endpoint_editable and not preset.base_url


def endpoint_editable(provider: str) -> bool:
    """설정의 엔드포인트를 쓰는 프로바이더인가 (로컬 서버만)."""
    preset = OPENAI_COMPATIBLE_PRESETS.get(provider)
    return preset is not None and preset.endpoint_editable


# ──────────────────────────────────────────────
# 준비 상태 (설정 화면 배너)
# ──────────────────────────────────────────────

ReadinessKind = Literal["subscription", "api_key", "local"]


@dataclass(frozen=True)
class ProviderReadiness:
    """프로바이더를 쓸 준비가 됐는지.

    Attributes:
        kind: 구독 CLI / API 키 / 로컬 서버.
        requirement: 필요한 것의 이름 (CLI 실행 파일 또는 키 환경 변수). 로컬은 빈 문자열.
        ready: CLI 가 PATH 에 있거나 키 환경 변수가 있다. 로컬은 늘 True
            (서버가 떠 있는지·구독 로그인이 됐는지는 호출할 때 안다).
    """

    kind: ReadinessKind
    requirement: str
    ready: bool


def provider_readiness() -> dict[str, ProviderReadiness]:
    """프로바이더마다 준비 상태 (키 값은 돌려주지 않는다)."""
    result: dict[str, ProviderReadiness] = {}
    for name, executable in CLI_EXECUTABLES.items():
        result[name] = ProviderReadiness(
            "subscription", executable, shutil.which(executable) is not None
        )
    for name, preset in OPENAI_COMPATIBLE_PRESETS.items():
        if preset.api_key_env:
            result[name] = ProviderReadiness(
                "api_key", preset.api_key_env, bool(os.environ.get(preset.api_key_env))
            )
        else:
            result[name] = ProviderReadiness("local", "", True)
    return result


# ──────────────────────────────────────────────
# 응답 데이터·오류
# ──────────────────────────────────────────────


@dataclass
class LLMResponse:
    """LLM 응답 결과."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class LLMTransientError(Exception):
    """다시 시도하면 나을 수 있는 오류 (시간 초과, 한도 초과, 서버 오류).

    ``LLMSegmentationError`` 는 설정·인증 문제라 재시도하지 않는다.
    """


# ──────────────────────────────────────────────
# RPM 제한
# ──────────────────────────────────────────────


class RateLimiter:
    """분당 요청 수(RPM) 제한. 최근 60초 동안 보낸 요청 시각을 세어 한도에 닿으면 기다린다.

    작업 하나(자식 프로세스 하나) 안에서 센다. 웹 작업은 한 번에 하나씩 돌아 동시에 두 프로세스가
    같은 한도를 나눠 쓰지 않는다.
    """

    def __init__(
        self,
        rpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rpm = rpm
        self._clock = clock
        self._sleep = sleep
        self._sent: deque[float] = deque()

    def acquire(self) -> None:
        """요청 하나를 보내도 될 때까지 기다린다. rpm 이 0 이하면 바로 돌아온다."""
        if self._rpm <= 0:
            return
        while True:
            now = self._clock()
            while self._sent and now - self._sent[0] >= _RPM_WINDOW_S:
                self._sent.popleft()
            if len(self._sent) < self._rpm:
                self._sent.append(now)
                return
            wait_s = _RPM_WINDOW_S - (now - self._sent[0])
            log.info("LLM RPM 한도 대기", rpm=self._rpm, wait_s=round(wait_s, 1))
            self._sleep(wait_s)


# ──────────────────────────────────────────────
# 프로바이더 추상 인터페이스
# ──────────────────────────────────────────────


class LLMProvider(ABC):
    """LLM 프로바이더 추상 인터페이스."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """LLM에 프롬프트를 전송하고 응답을 반환한다.

        Args:
            system_prompt: 시스템 프롬프트.
            user_prompt: 사용자 프롬프트.

        Returns:
            LLMResponse 인스턴스.

        Raises:
            LLMSegmentationError: 설정·인증 문제처럼 재시도해도 안 되는 실패.
        """

    @abstractmethod
    def provider_name(self) -> str:
        """프로바이더 이름을 반환한다."""


# ──────────────────────────────────────────────
# OpenAI 호환 프로바이더 (Gemini, OpenRouter, NVIDIA NIM, Ollama, 로컬)
# ──────────────────────────────────────────────


# OpenAI 호환 요청 본문에 합칠 JSON (.env). 모델마다 추론을 끄는 방법이 달라 코드 수정 없이 맞춘다
# (2026-10-11 사용자 요청). 예: NIM glm {"reasoning_effort": "low"}, llama-server Qwen3 계열
# {"chat_template_kwargs": {"enable_thinking": false}}. {"models": {"<모델>": {...}, "*": {...}}} 이면
# 지금 모델(없으면 "*")의 항목을 쓴다
LLM_EXTRA_BODY_ENV = "SUBWYRHTA_LLM_EXTRA_BODY"


def request_extra_body(config: LLMConfig) -> dict[str, Any]:
    """OpenAI 호환 요청에 더할 본문: 설정(추론 끄기·추론 수준) 위에 환경 변수 JSON 을 합친다.

    설정이 만드는 항목은 그 서버가 모르면 거부될 수 있어 켰을 때만 넣는다. 환경 변수 값이 우선하고,
    값이 ``null`` 이면 그 항목을 뺀다 (예: 설정의 ``chat_template_kwargs.thinking`` 을 빼기).

    Args:
        config: LLM 설정 (모델 이름으로 모델별 항목을 고른다).

    Returns:
        요청 본문에 합칠 사전 (비어 있을 수 있다).

    Raises:
        LLMSegmentationError: 환경 변수가 JSON 객체가 아닐 때 (재시도하지 않는 설정 오류).
    """
    body: dict[str, Any] = {}
    if config.disable_thinking:
        body["chat_template_kwargs"] = {"thinking": False}
    if config.reasoning_effort:
        body["reasoning_effort"] = config.reasoning_effort
    return _merge_body(body, _env_extra_body(config.model))


def _env_extra_body(model: str) -> dict[str, Any]:
    raw = os.environ.get(LLM_EXTRA_BODY_ENV, "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMSegmentationError(f"{LLM_EXTRA_BODY_ENV} 가 JSON 이 아닙니다: {e}") from e
    if not isinstance(data, dict):
        raise LLMSegmentationError(f"{LLM_EXTRA_BODY_ENV} 는 JSON 객체여야 합니다")
    models = data.get("models")
    if isinstance(models, dict):
        chosen = models.get(model, models.get("*", {}))
        if not isinstance(chosen, dict):
            raise LLMSegmentationError(
                f"{LLM_EXTRA_BODY_ENV} 의 models.{model} 은 JSON 객체여야 합니다"
            )
        return chosen
    return data


def _merge_body(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """사전을 깊게 합친다 (extra 우선, 값이 None 이면 그 키를 뺀다)."""
    merged = dict(base)
    for key, value in extra.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_body(merged[key], value)
        else:
            merged[key] = value
    return merged


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI Chat Completions 호환 API 프로바이더."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._name = config.provider.lower()
        self._preset = OPENAI_COMPATIBLE_PRESETS[self._name]
        self._client: openai.OpenAI | None = None

    def _get_client(self) -> openai.OpenAI:
        if self._client is not None:
            return self._client

        try:
            import openai
        except ImportError as e:
            raise LLMSegmentationError(
                "openai 패키지가 설치되지 않았습니다: pip install -r requirements.txt"
            ) from e

        # 클라우드 프로바이더는 고정 주소만 쓴다 (설정에 남은 base_url 은 쓰지 않는다)
        base_url = self._preset.base_url
        if self._preset.endpoint_editable and self._config.base_url:
            base_url = self._config.base_url
        if not base_url:
            raise LLMSegmentationError(
                f"{self._name} 프로바이더는 엔드포인트 URL(llm.base_url)이 필요합니다"
            )

        api_key = _resolve_api_key(self._name)
        if self._preset.api_key_env and not api_key:
            raise LLMSegmentationError(
                f"{self._name} API 키가 없습니다: .env 또는 환경 변수 "
                f"{self._preset.api_key_env} 를 설정하세요"
            )

        # 재시도는 complete_with_retry 가 한다. SDK 기본 재시도(2회)를 두면 시간 초과 한 번이
        # timeout_s 의 3배가 되고 RPM 계산에서도 빠진다
        self._client = openai.OpenAI(
            api_key=api_key or _NO_KEY_PLACEHOLDER,
            base_url=base_url,
            timeout=self._config.timeout_s,
            max_retries=0,
        )
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Chat Completions API를 호출한다."""
        client = self._get_client()

        extra_body = request_extra_body(self._config)
        response = client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            extra_body=extra_body or None,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response.model or self._config.model,
        )

    def provider_name(self) -> str:
        return self._name


# ──────────────────────────────────────────────
# 구독 CLI 프로바이더 (Claude Code, Codex)
# ──────────────────────────────────────────────


class ClaudeCodeProvider(LLMProvider):
    """Claude 구독(Pro/Max)으로 Claude Code CLI 를 비대화 모드로 호출한다.

    ``--bare`` 는 구독 로그인을 읽지 않으므로 쓰지 않는다. 대신 사용자 설정·CLAUDE.md·훅·MCP 가
    끼어들지 않게 ``--safe-mode``·``--strict-mcp-config`` 와 빈 임시 폴더를 쓰고, 도구는 끈다.
    """

    EXECUTABLE = CLI_EXECUTABLES["claude-code"]
    # API 키가 있으면 CLI 가 구독 대신 API 과금으로 간다. CLAUDECODE 는 Claude Code 안에서 띄웠다는 표시다
    DROPPED_ENV: tuple[str, ...] = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE")

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def build_command(self, system_prompt: str) -> list[str]:
        """CLI 명령 (프롬프트는 stdin 으로 보낸다)."""
        command = [
            self.EXECUTABLE,
            "-p",
            "--output-format",
            "json",
            "--system-prompt",
            system_prompt,
            "--tools",
            "",
            "--strict-mcp-config",
            "--safe-mode",
            "--no-session-persistence",
        ]
        if self._config.model:
            command += ["--model", self._config.model]
        return command

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """``claude -p`` 결과 JSON 을 읽는다."""
        result = _run_cli(
            self.build_command(system_prompt), user_prompt, self._config.timeout_s, self.DROPPED_ENV
        )
        data = _parse_json_object(result.stdout)
        if data is None:
            raise LLMSegmentationError(
                f"claude 출력을 읽지 못했습니다 (종료 코드 {result.returncode}): "
                f"{stderr_tail(result.stderr) or result.stdout[:200]}"
            )
        message = str(data.get("result") or "")
        if data.get("is_error"):
            status = data.get("api_error_status")
            if isinstance(status, int) and _is_retryable_status(status):
                raise LLMTransientError(f"claude 호출 실패 (HTTP {status}): {message}")
            # 인증 만료·구독 없음 등: 재시도해도 같다
            raise LLMSegmentationError(f"claude 호출 실패: {message}")

        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        model_usage = data.get("modelUsage")
        model = next(iter(model_usage), "") if isinstance(model_usage, dict) else ""
        return LLMResponse(
            text=message,
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            model=model or self._config.model,
        )

    def provider_name(self) -> str:
        return "claude-code"


class CodexProvider(LLMProvider):
    """ChatGPT 구독으로 Codex CLI 를 비대화 모드로 호출한다.

    ``codex exec`` 는 시스템 프롬프트 옵션이 없어 지시를 사용자 프롬프트 앞에 붙인다.
    읽기 전용 샌드박스·사용자 설정 무시·세션 저장 안 함으로 실행한다.
    """

    EXECUTABLE = CLI_EXECUTABLES["codex"]
    # CODEX_API_KEY 가 있으면 codex exec 은 구독 로그인보다 API 키를 먼저 쓴다
    DROPPED_ENV: tuple[str, ...] = ("CODEX_API_KEY", "OPENAI_API_KEY")

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def build_command(self) -> list[str]:
        """CLI 명령 (``-`` : 프롬프트를 stdin 으로 읽는다)."""
        command = [
            self.EXECUTABLE,
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--ignore-user-config",
            "--ignore-rules",
        ]
        if self._config.model:
            command += ["-m", self._config.model]
        command.append("-")
        return command

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """``codex exec --json`` 이벤트에서 마지막 답과 토큰 사용량을 읽는다."""
        prompt = f"{system_prompt}\n\n{user_prompt}"
        result = _run_cli(self.build_command(), prompt, self._config.timeout_s, self.DROPPED_ENV)

        text = ""
        error = ""
        usage: Mapping[str, Any] = {}
        for event in _parse_json_lines(result.stdout):
            kind = event.get("type")
            item = event.get("item")
            if (
                kind == "item.completed"
                and isinstance(item, dict)
                and item.get("type") == "agent_message"
            ):
                text = str(item.get("text") or "")
            elif kind == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
            elif kind in ("turn.failed", "error"):
                error = _event_error_message(event)

        if error or not text:
            reason = error or stderr_tail(result.stderr) or f"종료 코드 {result.returncode}"
            raise LLMSegmentationError(f"codex 호출 실패: {reason}")
        return LLMResponse(
            text=text,
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            model=self._config.model,
        )

    def provider_name(self) -> str:
        return "codex"


# ──────────────────────────────────────────────
# 팩토리 + 재시도 래퍼
# ──────────────────────────────────────────────

_PROVIDER_MAP: dict[str, Callable[[LLMConfig], LLMProvider]] = {
    "codex": CodexProvider,
    "claude-code": ClaudeCodeProvider,
    **dict.fromkeys(OPENAI_COMPATIBLE_PRESETS, OpenAICompatibleProvider),
}


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """설정에 따라 LLM 프로바이더 인스턴스를 생성한다.

    Args:
        config: LLM 설정.

    Returns:
        LLMProvider 구현체 인스턴스.

    Raises:
        LLMSegmentationError: 알 수 없는 프로바이더이거나, API·로컬 방식인데 모델이 비었을 때.
    """
    provider_name = config.provider.lower()

    if provider_name not in _PROVIDER_MAP:
        available = ", ".join(LLM_PROVIDERS)
        raise LLMSegmentationError(
            f"알 수 없는 LLM 프로바이더: {config.provider!r}. 사용 가능: {available}"
        )
    if not config.model and not is_cli_provider(provider_name):
        raise LLMSegmentationError(f"{provider_name} 프로바이더는 모델(llm.model)이 필요합니다")

    provider_cls = _PROVIDER_MAP[provider_name]
    log.info("LLM 프로바이더 생성", provider=provider_name, model=config.model, rpm=config.rpm)
    return provider_cls(config)


def complete_with_retry(
    provider: LLMProvider,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    rate_limiter: RateLimiter | None = None,
) -> LLMResponse:
    """Exponential backoff 재시도로 LLM 호출을 수행한다.

    Args:
        provider: LLM 프로바이더 인스턴스.
        system_prompt: 시스템 프롬프트.
        user_prompt: 사용자 프롬프트.
        max_retries: 최대 재시도 횟수.
        rate_limiter: RPM 제한. 재시도를 포함해 매 호출 전에 기다린다. None 이면 제한 없음.

    Returns:
        LLMResponse 인스턴스.

    Raises:
        LLMSegmentationError: 모든 재시도 실패 시.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            response = provider.complete(system_prompt, user_prompt)
            if attempt > 0:
                log.info("LLM 재시도 성공", attempt=attempt + 1)
            return response
        except LLMSegmentationError:
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = 2**attempt
                log.warning(
                    "LLM 호출 실패, 재시도 예정",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay_s=delay,
                    error=str(e),
                )
                time.sleep(delay)
            else:
                log.error(
                    "LLM 호출 최종 실패",
                    attempts=max_retries + 1,
                    error=str(e),
                )

    raise LLMSegmentationError(f"LLM 호출이 {max_retries + 1}회 모두 실패했습니다: {last_error}")


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────


def _resolve_api_key(provider: str) -> str:
    """프로바이더의 API 키를 정해진 환경 변수(.env 포함)에서 읽는다. 없으면 빈 문자열."""
    preset = OPENAI_COMPATIBLE_PRESETS.get(provider)
    if preset is None or not preset.api_key_env:
        return ""
    return os.environ.get(preset.api_key_env, "")


def _run_cli(
    command: list[str], stdin_text: str, timeout_s: float, dropped_env: Iterable[str]
) -> subprocess.CompletedProcess[str]:
    """구독 CLI 를 빈 임시 폴더에서 실행한다.

    작업 폴더를 비워 두는 이유: CLI 가 그 폴더의 프로젝트 설정(CLAUDE.md, AGENTS.md, .mcp.json)을
    읽지 않게 한다.

    Raises:
        LLMSegmentationError: CLI 를 찾지 못했을 때.
        LLMTransientError: 시간 초과.
    """
    executable = shutil.which(command[0])
    if executable is None:
        raise LLMSegmentationError(
            f"{command[0]} CLI 를 찾을 수 없습니다: "
            "서버를 실행하는 계정의 PATH 에 설치하고 로그인하세요"
        )
    dropped = set(dropped_env)
    env = {key: value for key, value in os.environ.items() if key not in dropped}
    with tempfile.TemporaryDirectory(prefix="subwyrhta-llm-") as workdir:
        try:
            return subprocess.run(
                [executable, *command[1:]],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=workdir,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise LLMTransientError(
                f"{command[0]} 응답이 {timeout_s:g}초 안에 오지 않았습니다"
            ) from e


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """출력 전체가 JSON 객체 하나면 그 객체, 아니면 None."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _parse_json_lines(text: str) -> list[dict[str, Any]]:
    """JSON Lines 중 객체인 줄만 (진행 문구 등 다른 줄은 건너뛴다)."""
    events: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _event_error_message(event: Mapping[str, Any]) -> str:
    """codex ``turn.failed``·``error`` 이벤트의 오류 문구."""
    error = event.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or event.get("message") or event.get("type"))


def _is_retryable_status(status: int) -> bool:
    return status == _RATE_LIMIT_STATUS or status >= _RETRYABLE_STATUS_MIN


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else 0
