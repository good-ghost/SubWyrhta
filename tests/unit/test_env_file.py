""".env 파일 지원 테스트 (CLI·serve 시작 때 환경 변수로 넣는다)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from src.core.config import AppConfig, load_env_file
from src.main import cli

if TYPE_CHECKING:
    from collections.abc import Iterator

_KEYS = (
    "SUBWYRHTA_TEST_A",
    "SUBWYRHTA_TEST_EMPTY",
    "SUBWYRHTA_TEST_SHELL",
    "HF_TOKEN",
    "SUBWYRHTA_DEVICE",
)


@pytest.fixture(autouse=True)
def _clean_env() -> Iterator[None]:
    # load_env_file 은 os.environ 에 직접 넣는다. monkeypatch.delenv 는 원래 없던 키를 기록하지 않아
    # 테스트가 넣은 값이 다음 테스트로 새므로, 원래 상태를 저장했다가 되돌린다
    saved = {key: os.environ.pop(key, None) for key in _KEYS}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadEnvFile:
    def test_no_default_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert load_env_file() == []

    def test_reads_default_file_in_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path / ".env", "SUBWYRHTA_TEST_A=one\n")
        monkeypatch.chdir(tmp_path)

        assert load_env_file() == ["SUBWYRHTA_TEST_A"]
        assert os.environ["SUBWYRHTA_TEST_A"] == "one"

    def test_shell_value_wins_and_empty_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SUBWYRHTA_TEST_SHELL", "from-shell")
        env = _write(
            tmp_path / "custom.env",
            "# 주석\nSUBWYRHTA_TEST_A='quoted value'\nSUBWYRHTA_TEST_EMPTY=\n"
            "SUBWYRHTA_TEST_SHELL=from-file\n",
        )

        assert load_env_file(env) == ["SUBWYRHTA_TEST_A"]
        assert os.environ["SUBWYRHTA_TEST_A"] == "quoted value"
        assert os.environ["SUBWYRHTA_TEST_SHELL"] == "from-shell"
        assert "SUBWYRHTA_TEST_EMPTY" not in os.environ

    def test_empty_device_does_not_override_config(self, tmp_path: Path) -> None:
        # .env.example 을 복사해 비워 둔 줄이 설정을 빈 값으로 덮어쓰지 않는다
        env = _write(tmp_path / ".env", "HF_TOKEN=hf_test\nSUBWYRHTA_DEVICE=\n")
        load_env_file(env)

        config = AppConfig.load()
        assert config.diarization.hf_token == "hf_test"
        assert config.stt.device == AppConfig().stt.device


class TestCliEnvFile:
    def test_env_file_option(self, tmp_path: Path) -> None:
        env = _write(tmp_path / "subwyrhta.env", "SUBWYRHTA_TEST_A=cli\n")

        result = CliRunner().invoke(cli, ["--env-file", str(env), "list-models"])

        assert result.exit_code == 0, result.output
        assert os.environ["SUBWYRHTA_TEST_A"] == "cli"

    def test_missing_env_file_is_an_error(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(cli, ["--env-file", str(tmp_path / "nope.env"), "list-models"])
        assert result.exit_code != 0

    def test_example_lists_every_env_override(self) -> None:
        """견본(.env.example)에 설정 덮어쓰기·LLM 키 환경 변수가 모두 있다."""
        from src.core.config import _ENV_OVERRIDES
        from src.segmentation.llm_client import OPENAI_COMPATIBLE_PRESETS

        example = (Path(__file__).parents[2] / ".env.example").read_text(encoding="utf-8")
        names = {
            line.split("=", 1)[0]
            for line in example.splitlines()
            if "=" in line and not line.startswith("#")
        }
        expected = {
            *_ENV_OVERRIDES,
            *(
                preset.api_key_env
                for preset in OPENAI_COMPATIBLE_PRESETS.values()
                if preset.api_key_env
            ),
        }
        assert expected <= names
        # OpenAI·Claude 는 구독만: API 키 견본을 두지 않는다
        assert not names & {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
