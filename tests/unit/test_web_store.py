"""웹 백엔드 저장소 테스트: 작업 큐, 계정·세션, 비밀 저장소, 설정 파일, 로그 리더, 작업 옵션."""

from __future__ import annotations

import json
import stat
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from src.core.config import AppConfig
from src.web.auth import AccountError, AccountStore, SessionSigner, hash_password, verify_password
from src.web.job_options import JobOptions, default_values
from src.web.jobs import (
    ERROR_INTERRUPTED,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    JobError,
    JobStore,
)
from src.web.logs import read_log
from src.web.secret_store import SecretStore
from src.web.settings_io import (
    SettingsError,
    editable_values,
    ensure_config_file,
    save_settings,
    validate_values,
)

if TYPE_CHECKING:
    from pathlib import Path


def _upload(tmp_path: Path, name: str = "clip.mp4", data: bytes = b"data") -> Path:
    path = tmp_path / f"upload-{name}"
    path.write_bytes(data)
    return path


def _options() -> dict[str, object]:
    return JobOptions.defaults(AppConfig()).model_dump()


class TestJobStore:
    """작업 기록 상태 흐름."""

    def test_create_claim_progress_finish(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "data")
        first = store.create("a.mp4", _upload(tmp_path, "a.mp4"), _options())
        second = store.create("b.mp4", _upload(tmp_path, "b.mp4"), _options())
        assert first.status == STATUS_QUEUED and first.size == 4
        assert store.input_path(first).read_bytes() == b"data"

        claimed = store.claim_next()
        assert claimed is not None and claimed.id == first.id
        assert claimed.status == STATUS_RUNNING and claimed.attempts == 1
        store.update_progress(first.id, "transcribe", 3, 9)
        running = store.get(first.id)
        assert (running.stage, running.progress_done, running.progress_total) == (
            "transcribe",
            3,
            9,
        )

        store.finish(first.id, STATUS_DONE, outputs=["a.en.srt"], result={"segments": 2})
        done = store.get(first.id)
        assert done.status == STATUS_DONE and done.outputs == ["a.en.srt"]
        assert done.result == {"segments": 2} and done.finished_at is not None
        assert store.counts()[STATUS_QUEUED] == 1
        assert [job.id for job in store.query([STATUS_QUEUED])] == [second.id]
        assert [job.id for job in store.query()] == [second.id, first.id]

    def test_cancel_retry_delete(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "data")
        job = store.create("a.mp4", _upload(tmp_path), _options())
        assert store.cancel_queued(job.id)
        assert store.get(job.id).status == STATUS_CANCELLED
        assert not store.cancel_queued(job.id)

        retried = store.retry(job.id)
        assert retried.status == STATUS_QUEUED
        with pytest.raises(JobError, match="job_not_retryable"):
            store.retry(job.id)

        claimed = store.claim_next()
        assert claimed is not None
        with pytest.raises(JobError, match="job_running"):
            store.delete(job.id)
        store.finish(job.id, STATUS_FAILED, error="x")
        store.delete(job.id)
        assert not store.job_dir(job.id).exists()
        with pytest.raises(JobError, match="job_not_found"):
            store.get(job.id)

    def test_retry_needs_input(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "data")
        job = store.create("a.mp4", _upload(tmp_path), _options())
        store.claim_next()
        store.finish(job.id, STATUS_FAILED, error="x")
        store.input_path(job).unlink()
        with pytest.raises(JobError, match="input_deleted"):
            store.retry(job.id)

    def test_recover_interrupted(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "data")
        job = store.create("a.mp4", _upload(tmp_path), _options())
        store.claim_next()
        store.close()
        reopened = JobStore(tmp_path / "data")
        assert reopened.recover_interrupted() == 1
        failed = reopened.get(job.id)
        assert failed.status == STATUS_FAILED and failed.error == ERROR_INTERRUPTED

    def test_output_file_only_listed_names(self, tmp_path: Path) -> None:
        store = JobStore(tmp_path / "data")
        job = store.create("a.mp4", _upload(tmp_path), _options())
        out = store.output_dir(job.id)
        out.mkdir(parents=True)
        (out / "a.en.srt").write_text("x")
        store.claim_next()
        store.finish(job.id, STATUS_DONE, outputs=["a.en.srt"])
        assert store.output_file(job.id, "a.en.srt").read_text() == "x"
        for name in ("../input/a.mp4", "other.srt"):
            with pytest.raises(JobError, match="file_not_found"):
                store.output_file(job.id, name)


class TestAuth:
    """관리 계정과 세션 서명."""

    def test_password_hash(self) -> None:
        stored = hash_password("correct horse")
        assert verify_password("correct horse", stored)
        assert not verify_password("wrong", stored)
        assert not verify_password("x", "plain$text")

    def test_account_lifecycle(self, tmp_path: Path) -> None:
        accounts = AccountStore(SecretStore(tmp_path / "secrets.json"))
        assert not accounts.exists()
        with pytest.raises(AccountError, match="password_too_short"):
            accounts.create("admin", "short")
        accounts.create(" admin ", "password1")
        assert accounts.username() == "admin"
        assert accounts.check("admin", "password1")
        assert not accounts.check("admin", "password2")
        with pytest.raises(AccountError, match="account_exists"):
            accounts.create("other", "password1")

        key = accounts.session_key()
        with pytest.raises(AccountError, match="wrong_password"):
            accounts.change("nope", username="admin", new_password=None)
        accounts.change("password1", username="root", new_password="password2")
        assert accounts.check("root", "password2")
        # 계정을 바꾸면 서명 키도 바뀌어 이전 세션이 끝난다
        assert accounts.session_key() != key

    def test_session_signer_expiry(self) -> None:
        now = [1_000_000.0]
        signer = SessionSigner(lambda: b"k" * 32, max_age_s=60, clock=lambda: now[0])
        value = signer.issue()
        assert signer.verify(value)
        assert not signer.verify(value + "0")
        assert not signer.verify(None)
        now[0] += 61
        assert not signer.verify(value)

    def test_secret_store_permissions(self, tmp_path: Path) -> None:
        store = SecretStore(tmp_path / "secrets.json")
        store.set("a", "1")
        assert store.get("a") == "1" and store.get("b") is None
        mode = stat.S_IMODE((tmp_path / "secrets.json").stat().st_mode)
        assert mode == 0o600


class TestSettingsIo:
    """설정 파일 읽기·검증·저장."""

    def test_roundtrip_preserves_comments_and_hidden_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(
            "# 머리 주석\nstt:\n  model: whisperx  # 엔진\n  model_size: large-v3\n"
            "diarization:\n  hf_token: secret-value\noutput:\n  dir: ./somewhere\n",
            encoding="utf-8",
        )
        values = editable_values(AppConfig.load(path))
        assert "hf_token" not in values["diarization"]
        assert "dir" not in values["output"]
        values["stt"]["model"] = "faster-whisper"
        values["stt"]["compute_type"] = "int8"
        save_settings(path, validate_values(values))

        text = path.read_text(encoding="utf-8")
        assert "# 머리 주석" in text and "# 엔진" in text
        reloaded = AppConfig.load(path)
        assert reloaded.stt.model == "faster-whisper" and reloaded.stt.compute_type == "int8"
        assert reloaded.diarization.hf_token == "secret-value"
        assert reloaded.output.dir == "./somewhere"

    def test_validation_errors(self) -> None:
        values = editable_values(AppConfig())
        values["stt"]["beam_size"] = 0
        values["segmentation"]["force_split_after_s"] = 9.0
        values["noise_filter"]["custom_patterns"] = ["("]
        with pytest.raises(SettingsError) as info:
            validate_values(values)
        locs = {error["loc"] for error in info.value.errors}
        assert "stt.beam_size" in locs
        assert "segmentation" in locs
        assert "noise_filter.custom_patterns" in locs

    def test_unknown_key_rejected(self) -> None:
        values = editable_values(AppConfig())
        values["web"]["port"] = 1
        with pytest.raises(SettingsError):
            validate_values(values)

    def test_ensure_config_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "config.yaml"
        ensure_config_file(path)
        assert AppConfig.load(path).stt.model == AppConfig().stt.model
        path.write_text("stt:\n  model: faster-whisper\n", encoding="utf-8")
        ensure_config_file(path)
        assert AppConfig.load(path).stt.model == "faster-whisper"


class TestLogReader:
    """JSON 로그 이어 읽기."""

    def test_cursor_level_and_partial_line(self, tmp_path: Path) -> None:
        path = tmp_path / "subwyrhta.log"
        lines = [
            {"event": "a", "level": "info", "logger": "src.x", "timestamp": "t1"},
            {"event": "b", "level": "debug", "logger": "src.y", "timestamp": "t2", "job_id": 3},
            {"event": "c", "level": "error", "logger": "src.z", "timestamp": "t3", "path": "/p"},
        ]
        path.write_text(
            "".join(json.dumps(line) + "\n" for line in lines) + '{"event": "par', encoding="utf-8"
        )

        first = read_log(path, cursor=None, level="INFO", limit=10)
        assert first.reset
        assert [entry.message for entry in first.entries] == ["a", "c path=/p"]
        debug = read_log(path, cursor=None, level="DEBUG", limit=10)
        assert debug.entries[1].job_id == 3

        with path.open("a", encoding="utf-8") as f:
            f.write('tial", "level": "warning"}\n')
        more = read_log(path, cursor=first.cursor, level="INFO", limit=10)
        assert not more.reset
        assert [entry.message for entry in more.entries] == ["partial"]
        assert more.entries[0].level == "WARNING"

    def test_missing_file_and_truncated(self, tmp_path: Path) -> None:
        path = tmp_path / "none.log"
        assert read_log(path, cursor=None, level="INFO", limit=5).entries == []
        path.write_text('{"event": "x", "level": "info"}\n', encoding="utf-8")
        # 파일이 cursor 보다 짧아졌으면 처음부터 다시 읽는다
        chunk = read_log(path, cursor="999999", level="INFO", limit=5)
        assert chunk.reset and [entry.message for entry in chunk.entries] == ["x"]


class TestJobOptions:
    """작업 옵션 검증."""

    def test_defaults_and_overrides(self) -> None:
        config = AppConfig()
        options = JobOptions.defaults(config)
        assert options.engine == config.stt.model
        assert options.formats == [config.output.format]
        overrides = options.model_copy(update={"formats": ["vtt", "srt"]}).to_overrides()
        assert overrides["output.format"] == "vtt"
        assert overrides["audio.vad.pre_vad"] is True

    def test_invalid_values(self) -> None:
        base = default_values(AppConfig())
        for field, value in (
            ("engine", "nope"),
            ("formats", []),
            ("formats", ["txt"]),
            ("source_lang", "english"),
            ("num_speakers", "0"),
            ("max_cps", -1),
        ):
            with pytest.raises(ValidationError):
                JobOptions.model_validate({**base, field: value})

    def test_normalizes_language_and_speakers(self) -> None:
        options = JobOptions.model_validate(
            {
                **default_values(AppConfig()),
                "source_lang": " KO ",
                "num_speakers": " 3 ",
                "formats": ["srt", "srt"],
            }
        )
        assert options.source_lang == "ko"
        assert options.num_speakers == "3"
        assert options.formats == ["srt"]
