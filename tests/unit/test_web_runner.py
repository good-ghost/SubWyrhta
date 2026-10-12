"""작업 자식 프로세스 진입 함수 테스트 (파이프라인은 가짜)."""

from __future__ import annotations

import dataclasses
import pickle
from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import patch

from src.core.config import AppConfig
from src.models.result import TranscriptionResult
from src.pipeline import PipelineOptions, PipelineResult
from src.web.runner import EVENT_DONE, JobPayload, run_job_process

if TYPE_CHECKING:
    from pathlib import Path


class _Events:
    def __init__(self) -> None:
        self.items: list[tuple[Any, ...]] = []

    def put(self, item: tuple[Any, ...]) -> None:
        self.items.append(item)


class _FakePipeline:
    options: ClassVar[list[PipelineOptions]] = []

    def __init__(self, config: AppConfig, progress: Any = None) -> None:
        pass

    def __enter__(self) -> _FakePipeline:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def run(self, input_path: str, options: PipelineOptions) -> PipelineResult:
        _FakePipeline.options.append(options)
        return PipelineResult(result=TranscriptionResult())


def _payload(tmp_path: Path, **overrides: Any) -> JobPayload:
    values: dict[str, Any] = {
        "job_id": 7,
        "input_path": str(tmp_path / "in.mp4"),
        "output_dir": str(tmp_path / "out"),
        "checkpoint_path": str(tmp_path / "checkpoint.json"),
        "config": dataclasses.asdict(AppConfig()),
        "formats": ["srt"],
        "resume": True,
        "log_file": str(tmp_path / "job.log"),
        "log_level": "WARNING",
        "work_dir": str(tmp_path / "work"),
    }
    values.update(overrides)
    return JobPayload(**values)


class TestRunJobProcess:
    def setup_method(self) -> None:
        _FakePipeline.options = []

    def test_passes_work_dir_and_keeps_stages(self, tmp_path: Path) -> None:
        events = _Events()
        with patch("src.web.runner.TranscriptionPipeline", _FakePipeline):
            run_job_process(_payload(tmp_path), events)  # type: ignore[arg-type]

        [options] = _FakePipeline.options
        assert options.keep_stages is True
        assert options.work_dir == str(tmp_path / "work")
        assert options.resume is True
        assert events.items[-1][0] == EVENT_DONE

    def test_without_work_dir_stages_are_not_kept(self, tmp_path: Path) -> None:
        with patch("src.web.runner.TranscriptionPipeline", _FakePipeline):
            run_job_process(_payload(tmp_path, work_dir=""), _Events())  # type: ignore[arg-type]

        [options] = _FakePipeline.options
        assert options.keep_stages is False and options.work_dir is None

    def test_payload_from_older_server_has_work_dir(self, tmp_path: Path) -> None:
        """재시작 전 서버(필드가 없던 코드)가 보낸 값을 풀어도 빈 work_dir 로 읽힌다."""
        payload = _payload(tmp_path)
        state = dict(payload.__dict__)
        del state["work_dir"]
        old = object.__new__(JobPayload)
        object.__setattr__(old, "__dict__", state)

        restored = pickle.loads(pickle.dumps(old))
        assert restored.work_dir == ""
