"""LLM 프로바이더 테스트 (PHASE-PLAN-08): 구독 CLI, 클라우드 API 키, 로컬 서버, RPM, 설정 검증."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.config import LLMConfig
from src.core.exceptions import LLMSegmentationError
from src.segmentation import llm_client
from src.segmentation.llm_client import (
    CLI_PROVIDERS,
    LLM_EXTRA_BODY_ENV,
    LLM_PROVIDERS,
    ClaudeCodeProvider,
    CodexProvider,
    LLMResponse,
    LLMTransientError,
    OpenAICompatibleProvider,
    ProviderReadiness,
    RateLimiter,
    complete_with_retry,
    create_llm_provider,
    provider_readiness,
    request_extra_body,
)

_KEY_ENVS = ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY")


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _KEY_ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """openai 패키지 대신 쓰는 모의 모듈 (클라이언트 생성 인자를 기록한다)."""
    module = MagicMock()
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[3, 7]"))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
        model="served-model",
    )
    module.OpenAI.return_value.chat.completions.create.return_value = completion
    monkeypatch.setitem(sys.modules, "openai", module)
    return module


# ══════════════════════════════════════════════
# 프로바이더 목록·팩토리
# ══════════════════════════════════════════════


class TestFactory:
    def test_provider_set(self) -> None:
        # OpenAI·Claude 는 구독만, NIM 은 클라우드만 (2026-10-09 사용자 결정)
        assert LLM_PROVIDERS == (
            "codex",
            "claude-code",
            "gemini",
            "openrouter",
            "nvidia-nim",
            "ollama",
            "local",
        )

    @pytest.mark.parametrize("provider", LLM_PROVIDERS)
    def test_every_provider_creates(self, provider: str) -> None:
        cfg = LLMConfig(provider=provider, model="m")
        assert create_llm_provider(cfg).provider_name() == provider

    @pytest.mark.parametrize("provider", ["openai", "anthropic"])
    def test_api_key_openai_and_anthropic_are_not_allowed(self, provider: str) -> None:
        with pytest.raises(LLMSegmentationError, match="알 수 없는"):
            create_llm_provider(LLMConfig(provider=provider, model="m"))

    def test_api_provider_requires_model(self) -> None:
        with pytest.raises(LLMSegmentationError, match="모델"):
            create_llm_provider(LLMConfig(provider="gemini", model=""))

    @pytest.mark.parametrize("provider", CLI_PROVIDERS)
    def test_cli_provider_allows_empty_model(self, provider: str) -> None:
        assert (
            create_llm_provider(LLMConfig(provider=provider, model="")).provider_name() == provider
        )


class TestReadiness:
    def test_reports_cli_and_keys_without_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "secret")
        monkeypatch.setattr(
            llm_client.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None
        )

        readiness = provider_readiness()

        assert set(readiness) == set(LLM_PROVIDERS)
        assert readiness["claude-code"] == ProviderReadiness("subscription", "claude", True)
        assert readiness["codex"] == ProviderReadiness("subscription", "codex", False)
        assert readiness["gemini"] == ProviderReadiness("api_key", "GEMINI_API_KEY", True)
        assert readiness["nvidia-nim"] == ProviderReadiness("api_key", "NVIDIA_API_KEY", False)
        assert readiness["ollama"] == ProviderReadiness("local", "", True)
        assert "secret" not in repr(readiness)


# ══════════════════════════════════════════════
# OpenAI 호환 (Gemini, OpenRouter, NVIDIA NIM, Ollama, local)
# ══════════════════════════════════════════════


class TestExtraBodyEnv:
    """.env 의 SUBWYRHTA_LLM_EXTRA_BODY 로 모델별 추론 끄기를 코드 수정 없이 맞춘다 (2026-10-11)."""

    def test_plain_body_merges_over_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            LLM_EXTRA_BODY_ENV, '{"chat_template_kwargs": {"enable_thinking": false}}'
        )
        body = request_extra_body(
            LLMConfig(provider="local", model="qwen3.8-27b", disable_thinking=True)
        )
        assert body == {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}

    def test_models_map_picks_current_model_or_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            LLM_EXTRA_BODY_ENV,
            json.dumps(
                {
                    "models": {
                        "z-ai/glm-5.3-flash": {"reasoning_effort": "low"},
                        "*": {"chat_template_kwargs": {"enable_thinking": False}},
                    }
                }
            ),
        )
        assert request_extra_body(LLMConfig(model="z-ai/glm-5.3-flash")) == {
            "reasoning_effort": "low"
        }
        assert request_extra_body(LLMConfig(model="qwen3.8-35b-a3b")) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_null_removes_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            LLM_EXTRA_BODY_ENV,
            '{"chat_template_kwargs": {"thinking": null, "enable_thinking": false}}',
        )
        body = request_extra_body(LLMConfig(model="m", disable_thinking=True))
        assert body == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_unset_uses_settings_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(LLM_EXTRA_BODY_ENV, raising=False)
        assert request_extra_body(LLMConfig(model="m")) == {}

    @pytest.mark.parametrize("raw", ["{not json", "[1, 2]", '{"models": {"m": 3}}'])
    def test_invalid_raises_config_error(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(LLM_EXTRA_BODY_ENV, raw)
        with pytest.raises(LLMSegmentationError, match=LLM_EXTRA_BODY_ENV):
            request_extra_body(LLMConfig(model="m"))

    def test_sent_with_request(
        self, fake_openai: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            LLM_EXTRA_BODY_ENV, '{"chat_template_kwargs": {"enable_thinking": false}}'
        )
        config = LLMConfig(
            provider="local", model="qwen3.8-27b", base_url="http://127.0.0.1:8080/v1"
        )
        OpenAICompatibleProvider(config).complete("s", "u")
        kwargs = fake_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


class TestOpenAICompatible:
    @pytest.mark.parametrize(("effort", "sent"), [("", False), ("low", True)])
    def test_reasoning_effort_only_when_set(
        self,
        fake_openai: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        effort: str,
        sent: bool,
    ) -> None:
        """reasoning_effort 는 정했을 때만 보낸다 (NIM glm-5.3-flash: low 면 묶음당 13초)."""
        monkeypatch.setenv("NVIDIA_API_KEY", "k-123")
        config = LLMConfig(provider="nvidia-nim", model="m", reasoning_effort=effort)
        OpenAICompatibleProvider(config).complete("s", "u")

        body = fake_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs[
            "extra_body"
        ]
        assert (body is not None and body.get("reasoning_effort") == "low") is sent

    @pytest.mark.parametrize(
        ("disable_thinking", "expected"),
        [(False, None), (True, {"chat_template_kwargs": {"thinking": False}})],
    )
    def test_disable_thinking(
        self,
        fake_openai: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        disable_thinking: bool,
        expected: dict[str, object] | None,
    ) -> None:
        """추론 끄기는 켰을 때만 chat_template_kwargs 를 보낸다 (NIM deepseek 묶음당 5분 → 30초)."""
        monkeypatch.setenv("NVIDIA_API_KEY", "k-123")
        config = LLMConfig(provider="nvidia-nim", model="m", disable_thinking=disable_thinking)
        OpenAICompatibleProvider(config).complete("s", "u")

        create = fake_openai.OpenAI.return_value.chat.completions.create
        assert create.call_args.kwargs["extra_body"] == expected

    @pytest.mark.parametrize(
        ("provider", "env", "base_url"),
        [
            (
                "gemini",
                "GEMINI_API_KEY",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
            ("nvidia-nim", "NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1"),
        ],
    )
    def test_cloud_endpoint_and_key(
        self,
        fake_openai: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        provider: str,
        env: str,
        base_url: str,
    ) -> None:
        monkeypatch.setenv(env, "k-123")
        response = OpenAICompatibleProvider(LLMConfig(provider=provider, model="m")).complete(
            "s", "u"
        )

        kwargs = fake_openai.OpenAI.call_args.kwargs
        assert kwargs["base_url"] == base_url
        assert kwargs["api_key"] == "k-123"
        # 재시도는 complete_with_retry 가 한다 (SDK 재시도를 끈다)
        assert kwargs["max_retries"] == 0
        assert response == LLMResponse(
            text="[3, 7]", input_tokens=11, output_tokens=4, model="served-model"
        )

    def test_cloud_ignores_configured_endpoint(
        self, fake_openai: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NVIDIA NIM 은 클라우드만: 설정에 남은 base_url(자체 호스팅 주소)은 쓰지 않는다
        monkeypatch.setenv("NVIDIA_API_KEY", "k")
        cfg = LLMConfig(provider="nvidia-nim", model="m", base_url="http://nim.lan:8000/v1")
        OpenAICompatibleProvider(cfg).complete("s", "u")

        assert (
            fake_openai.OpenAI.call_args.kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
        )

    def test_missing_key_fails_without_retry(self, fake_openai: MagicMock) -> None:
        with pytest.raises(LLMSegmentationError, match="GEMINI_API_KEY"):
            OpenAICompatibleProvider(LLMConfig(provider="gemini", model="m")).complete("s", "u")
        fake_openai.OpenAI.assert_not_called()

    def test_nim_without_key_fails(self, fake_openai: MagicMock) -> None:
        cfg = LLMConfig(provider="nvidia-nim", model="m", base_url="http://nim.lan:8000/v1")
        with pytest.raises(LLMSegmentationError, match="NVIDIA_API_KEY"):
            OpenAICompatibleProvider(cfg).complete("s", "u")

    def test_ollama_default_endpoint_without_key(self, fake_openai: MagicMock) -> None:
        OpenAICompatibleProvider(LLMConfig(provider="ollama", model="llama3")).complete("s", "u")

        kwargs = fake_openai.OpenAI.call_args.kwargs
        assert kwargs["base_url"] == "http://localhost:11434/v1"
        assert kwargs["api_key"] == "not-needed"

    def test_ollama_remote_endpoint(self, fake_openai: MagicMock) -> None:
        cfg = LLMConfig(provider="ollama", model="llama3", base_url="http://gpu.lan:11434/v1")
        OpenAICompatibleProvider(cfg).complete("s", "u")

        assert fake_openai.OpenAI.call_args.kwargs["base_url"] == "http://gpu.lan:11434/v1"

    def test_local_requires_endpoint(self, fake_openai: MagicMock) -> None:
        with pytest.raises(LLMSegmentationError, match="base_url"):
            OpenAICompatibleProvider(LLMConfig(provider="local", model="m")).complete("s", "u")

    def test_local_with_endpoint(self, fake_openai: MagicMock) -> None:
        cfg = LLMConfig(provider="local", model="m", base_url="http://127.0.0.1:8080/v1")
        OpenAICompatibleProvider(cfg).complete("s", "u")

        assert fake_openai.OpenAI.call_args.kwargs["base_url"] == "http://127.0.0.1:8080/v1"

    def test_package_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(LLMSegmentationError, match=r"requirements\.txt"):
            OpenAICompatibleProvider(LLMConfig(provider="ollama", model="m")).complete("s", "u")


# ══════════════════════════════════════════════
# 구독 CLI (Claude Code, Codex)
# ══════════════════════════════════════════════


class _FakeRun:
    """subprocess.run 대신 쓰며 받은 인자를 기록한다."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[dict[str, Any]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"args": args, **kwargs})
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


@pytest.fixture
def cli_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_client.shutil, "which", lambda name: f"/usr/bin/{name}")


def _claude_json(**fields: Any) -> str:
    data: dict[str, Any] = {
        "type": "result",
        "is_error": False,
        "result": "[2, 5]",
        "usage": {"input_tokens": 30, "output_tokens": 6},
        "modelUsage": {"claude-sonnet-5-5": {}},
        "api_error_status": None,
    }
    data.update(fields)
    return json.dumps(data)


class TestClaudeCode:
    def test_command_is_isolated(self) -> None:
        command = ClaudeCodeProvider(
            LLMConfig(provider="claude-code", model="sonnet")
        ).build_command("SYS")

        assert command[:2] == ["claude", "-p"]
        assert command[command.index("--system-prompt") + 1] == "SYS"
        assert command[command.index("--tools") + 1] == ""
        for flag in ("--safe-mode", "--strict-mcp-config", "--no-session-persistence"):
            assert flag in command
        # --bare 는 구독 로그인을 읽지 않는다
        assert "--bare" not in command
        assert command[-2:] == ["--model", "sonnet"]

    def test_empty_model_uses_cli_default(self) -> None:
        command = ClaudeCodeProvider(LLMConfig(provider="claude-code", model="")).build_command("S")
        assert "--model" not in command

    def test_success(self, cli_on_path: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "would-bill-the-api")
        monkeypatch.setenv("KEEP_ME", "1")
        run = _FakeRun(stdout=_claude_json())
        monkeypatch.setattr(llm_client.subprocess, "run", run)

        response = ClaudeCodeProvider(LLMConfig(provider="claude-code", model="")).complete(
            "S", "USER"
        )

        assert response == LLMResponse(
            text="[2, 5]", input_tokens=30, output_tokens=6, model="claude-sonnet-5-5"
        )
        call = run.calls[0]
        assert call["input"] == "USER"
        assert call["env"]["KEEP_ME"] == "1"
        # API 키를 넘기면 구독 대신 API 로 과금된다
        assert "ANTHROPIC_API_KEY" not in call["env"]
        # 빈 임시 폴더에서 실행해 프로젝트 CLAUDE.md·.mcp.json 을 읽지 않는다
        assert Path(call["cwd"]).name.startswith("subwyrhta-llm-")
        assert not Path(call["cwd"]).exists()

    def test_auth_error_is_not_retried(
        self, cli_on_path: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stdout = _claude_json(
            is_error=True, result="Failed to authenticate: OAuth session expired", usage={}
        )
        monkeypatch.setattr(llm_client.subprocess, "run", _FakeRun(stdout=stdout, returncode=1))

        with pytest.raises(LLMSegmentationError, match="OAuth session expired"):
            ClaudeCodeProvider(LLMConfig(provider="claude-code")).complete("S", "U")

    @pytest.mark.parametrize("status", [429, 529])
    def test_rate_limit_is_transient(
        self, cli_on_path: None, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        stdout = _claude_json(is_error=True, result="busy", api_error_status=status)
        monkeypatch.setattr(llm_client.subprocess, "run", _FakeRun(stdout=stdout, returncode=1))

        with pytest.raises(LLMTransientError, match=str(status)):
            ClaudeCodeProvider(LLMConfig(provider="claude-code")).complete("S", "U")

    def test_unreadable_output(self, cli_on_path: None, monkeypatch: pytest.MonkeyPatch) -> None:
        run = _FakeRun(stdout="", stderr="error: unknown option '--safe-mode'", returncode=1)
        monkeypatch.setattr(llm_client.subprocess, "run", run)

        with pytest.raises(LLMSegmentationError, match="unknown option"):
            ClaudeCodeProvider(LLMConfig(provider="claude-code")).complete("S", "U")

    def test_cli_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_client.shutil, "which", lambda _name: None)

        with pytest.raises(LLMSegmentationError, match="claude CLI"):
            ClaudeCodeProvider(LLMConfig(provider="claude-code")).complete("S", "U")

    def test_timeout_is_transient(self, cli_on_path: None, monkeypatch: pytest.MonkeyPatch) -> None:
        def slow(args: list[str], **_kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(args, 5)

        monkeypatch.setattr(llm_client.subprocess, "run", slow)

        with pytest.raises(LLMTransientError, match="5초"):
            ClaudeCodeProvider(LLMConfig(provider="claude-code", timeout_s=5)).complete("S", "U")


def _codex_events(*events: dict[str, Any]) -> str:
    return "\n".join(json.dumps(event) for event in events) + "\n"


class TestCodex:
    def test_command(self) -> None:
        command = CodexProvider(LLMConfig(provider="codex", model="gpt-5-codex")).build_command()

        assert command[:2] == ["codex", "exec"]
        assert command[command.index("--sandbox") + 1] == "read-only"
        for flag in ("--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config"):
            assert flag in command
        assert command[-3:] == ["-m", "gpt-5-codex", "-"]

    def test_success(self, cli_on_path: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CODEX_API_KEY", "would-bill-the-api")
        stdout = _codex_events(
            {"type": "thread.started", "thread_id": "t"},
            {"type": "item.completed", "item": {"id": "i1", "type": "reasoning", "text": "..."}},
            {
                "type": "item.completed",
                "item": {"id": "i2", "type": "agent_message", "text": "[4]"},
            },
            {"type": "turn.completed", "usage": {"input_tokens": 40, "output_tokens": 3}},
        )
        run = _FakeRun(stdout="progress line\n" + stdout)
        monkeypatch.setattr(llm_client.subprocess, "run", run)

        response = CodexProvider(LLMConfig(provider="codex", model="")).complete("SYS", "USER")

        assert (response.text, response.input_tokens, response.output_tokens) == ("[4]", 40, 3)
        call = run.calls[0]
        assert call["input"] == "SYS\n\nUSER"
        assert "CODEX_API_KEY" not in call["env"]

    def test_turn_failed(self, cli_on_path: None, monkeypatch: pytest.MonkeyPatch) -> None:
        stdout = _codex_events(
            {"type": "turn.failed", "error": {"message": "You've hit your usage limit"}}
        )
        monkeypatch.setattr(llm_client.subprocess, "run", _FakeRun(stdout=stdout, returncode=1))

        with pytest.raises(LLMSegmentationError, match="usage limit"):
            CodexProvider(LLMConfig(provider="codex")).complete("S", "U")

    def test_no_answer_reports_stderr(
        self, cli_on_path: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = _FakeRun(stdout="", stderr="Not logged in. Run codex login", returncode=1)
        monkeypatch.setattr(llm_client.subprocess, "run", run)

        with pytest.raises(LLMSegmentationError, match="codex login"):
            CodexProvider(LLMConfig(provider="codex")).complete("S", "U")


# ══════════════════════════════════════════════
# RPM 제한
# ══════════════════════════════════════════════


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestRateLimiter:
    def test_unlimited(self) -> None:
        clock = _Clock()
        limiter = RateLimiter(0, clock=clock, sleep=clock.sleep)
        for _ in range(100):
            limiter.acquire()
        assert clock.slept == []

    def test_waits_for_oldest_request_to_leave_window(self) -> None:
        clock = _Clock()
        limiter = RateLimiter(2, clock=clock, sleep=clock.sleep)
        limiter.acquire()
        clock.now += 10
        limiter.acquire()
        clock.now += 5
        limiter.acquire()  # 첫 요청(15초 전)이 60초 구간에서 빠질 때까지 45초

        assert clock.slept == [45.0]

    def test_window_slides(self) -> None:
        clock = _Clock()
        limiter = RateLimiter(1, clock=clock, sleep=clock.sleep)
        limiter.acquire()
        clock.now += 61
        limiter.acquire()
        assert clock.slept == []

    def test_retries_count_as_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
        provider = MagicMock()
        provider.complete.side_effect = [LLMTransientError("busy"), LLMResponse(text="ok")]
        limiter = MagicMock(spec=RateLimiter)

        response = complete_with_retry(provider, "s", "u", max_retries=2, rate_limiter=limiter)

        assert response.text == "ok"
        assert limiter.acquire.call_count == 2


# ══════════════════════════════════════════════
# 웹 설정 검증 (settings_io.LlmSettings)
# ══════════════════════════════════════════════


class TestLlmSettingsValidation:
    @staticmethod
    def _values(**llm: Any) -> dict[str, Any]:
        from src.core.config import AppConfig
        from src.web.settings_io import editable_values

        values = editable_values(AppConfig())
        values["llm"].update(llm)
        return values

    @staticmethod
    def _error_messages(values: dict[str, Any]) -> list[str]:
        from src.web.settings_io import SettingsError, validate_values

        with pytest.raises(SettingsError) as caught:
            validate_values(values)
        return [f"{err['loc']}: {err['message']}" for err in caught.value.errors]

    @pytest.mark.parametrize("provider", LLM_PROVIDERS)
    def test_every_provider_accepted(self, provider: str) -> None:
        from src.web.settings_io import validate_values

        base_url = "http://127.0.0.1:8080/v1" if provider == "local" else ""
        values = self._values(provider=provider, model="m", base_url=base_url, rpm=30)
        settings = validate_values(values)
        assert (settings.llm.provider, settings.llm.rpm) == (provider, 30)

    @pytest.mark.parametrize("provider", ["x", "openai", "anthropic"])
    def test_unknown_provider(self, provider: str) -> None:
        messages = self._error_messages(self._values(provider=provider, model="m"))
        assert any("invalid_choice" in m for m in messages)

    def test_api_key_env_field_is_gone(self) -> None:
        # 키는 정해진 환경 변수(.env)로만 받는다
        values = self._values()
        assert "api_key_env" not in values["llm"]
        values["llm"]["api_key_env"] = "MY_KEY"
        assert any(m.startswith("llm.api_key_env") for m in self._error_messages(values))

    def test_cloud_provider_keeps_stale_endpoint(self) -> None:
        # 로컬에서 바꿔 와도 저장할 수 있다 (클라우드는 남은 엔드포인트를 쓰지 않는다)
        from src.web.settings_io import validate_values

        values = self._values(provider="gemini", model="m", base_url="http://127.0.0.1:8080/v1")
        assert validate_values(values).llm.provider == "gemini"

    def test_api_provider_needs_model(self) -> None:
        messages = self._error_messages(self._values(provider="gemini", model=""))
        assert any("llm_model_required" in m for m in messages)

    def test_cli_provider_allows_empty_model(self) -> None:
        from src.web.settings_io import validate_values

        assert validate_values(self._values(provider="claude-code", model="")).llm.model == ""

    def test_local_needs_endpoint(self) -> None:
        messages = self._error_messages(self._values(provider="local", model="m", base_url=""))
        assert any("llm_base_url_required" in m for m in messages)

    def test_endpoint_must_be_http(self) -> None:
        messages = self._error_messages(self._values(base_url="ftp://host"))
        assert any(m.startswith("llm.base_url") and "invalid_url" in m for m in messages)

    def test_negative_rpm(self) -> None:
        assert any(m.startswith("llm.rpm") for m in self._error_messages(self._values(rpm=-1)))


def test_provider_list_matches_web_options() -> None:
    """화면 선택지(web/src/settings/fields.ts)와 서버 목록이 같은 순서다."""
    import re

    source = (Path(__file__).parents[2] / "web/src/settings/fields.ts").read_text(encoding="utf-8")
    match = re.search(r"export const LLM_PROVIDERS = \[(.*?)\] as const", source, re.S)
    assert match is not None
    assert tuple(re.findall(r"'([^']+)'", match.group(1))) == LLM_PROVIDERS
    assert set(LLM_PROVIDERS) == {*llm_client.OPENAI_COMPATIBLE_PRESETS, *CLI_PROVIDERS}
