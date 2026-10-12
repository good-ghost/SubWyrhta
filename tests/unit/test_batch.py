"""배치 프로세서 단위 테스트."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.batch_processor import (
    build_device_assignments,
    collect_files,
    get_device_for_worker,
    process_batch,
)


class TestCollectFiles:
    def test_single_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.mp4"
        f.write_text("x")
        files = collect_files(str(f))
        assert len(files) == 1

    def test_directory_non_recursive(self, tmp_path: Path) -> None:
        (tmp_path / "a.mp4").write_text("x")
        (tmp_path / "b.wav").write_text("x")
        (tmp_path / "c.txt").write_text("x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.mp4").write_text("x")
        files = collect_files(str(tmp_path), recursive=False)
        assert len(files) == 2

    def test_directory_recursive(self, tmp_path: Path) -> None:
        (tmp_path / "a.mp4").write_text("x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.mp4").write_text("x")
        files = collect_files(str(tmp_path), recursive=True)
        assert len(files) == 2

    def test_glob_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "x.mp4").write_text("x")
        (tmp_path / "y.mp4").write_text("x")
        (tmp_path / "z.txt").write_text("x")
        import os

        files = collect_files(os.path.join(str(tmp_path), "*.mp4"))
        assert len(files) == 2

    def test_file_list(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.mp4"
        f2 = tmp_path / "b.wav"
        f1.write_text("x")
        f2.write_text("x")
        fl = tmp_path / "list.txt"
        fl.write_text(f"{f1}\n# comment\n{f2}\n")
        files = collect_files(str(tmp_path), file_list=str(fl))
        assert len(files) >= 2

    def test_unsupported_filtered(self, tmp_path: Path) -> None:
        (tmp_path / "doc.pdf").write_text("x")
        files = collect_files(str(tmp_path))
        assert len(files) == 0

    def test_dedup(self, tmp_path: Path) -> None:
        f = tmp_path / "a.mp4"
        f.write_text("x")
        fl = tmp_path / "list.txt"
        fl.write_text(str(f))
        files = collect_files(str(tmp_path), file_list=str(fl))
        # file appears in both dir scan and file list → dedup
        assert len(files) == 1


class TestProcessBatch:
    def test_sequential_success(self, tmp_path: Path) -> None:
        files = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        for f in files:
            f.write_text("x")
        ok, fail, _failed = process_batch(files, lambda p, device: True, workers=1)
        assert ok == 2
        assert fail == 0

    def test_with_failure(self, tmp_path: Path) -> None:
        def fn(p: Path, device: str | None) -> bool:
            if "bad" in p.name:
                raise ValueError("fail")
            return True

        good = tmp_path / "good.mp4"
        bad = tmp_path / "bad.mp4"
        good.write_text("x")
        bad.write_text("x")
        ok, fail, failed = process_batch([good, bad], fn, workers=1)
        assert ok == 1
        assert fail == 1
        assert len(failed) == 1

    def test_failed_list(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.mp4"
        bad.write_text("x")
        fl = str(tmp_path / "failed.txt")
        process_batch(
            [bad],
            lambda p, device: (_ for _ in ()).throw(ValueError("x")),
            workers=1,
            failed_list_path=fl,
        )
        assert Path(fl).exists()
        assert "bad.mp4" in Path(fl).read_text()

    def test_empty(self) -> None:
        ok, fail, _failed = process_batch([], lambda p, device: True)
        assert ok == 0 and fail == 0


class TestDeviceAssignments:
    def test_explicit(self) -> None:
        assert build_device_assignments(3, "cpu") == ["cpu"] * 3

    def test_auto_no_gpu(self) -> None:
        devices = build_device_assignments(4, None)
        assert all(d == "cpu" for d in devices)

    def test_auto_with_gpu(self) -> None:
        with patch("src.engines.base.resolve_device", return_value="cuda"):
            devices = build_device_assignments(3, None)
            assert devices[0] == "cuda"
            assert devices[1] == "cpu"
            assert devices[2] == "cpu"

    def test_single_gpu(self) -> None:
        with patch("src.engines.base.resolve_device", return_value="cuda"):
            assert build_device_assignments(1, None) == ["cuda"]

    def test_zero(self) -> None:
        assert build_device_assignments(0) == []

    def test_get_device(self) -> None:
        a = ["cuda", "cpu", "cpu"]
        assert get_device_for_worker(0, a) == "cuda"
        assert get_device_for_worker(1, a) == "cpu"
        assert get_device_for_worker(99, a) == "cpu"


# ══════════════════════════════════════════════
# CLI 배치 연결 (PHASE-PLAN-09 9.6)
# ══════════════════════════════════════════════


def _record_device(file: Path, device: str | None) -> bool:
    """spawn worker 에서 돈다: 처리한 파일과 그 worker 의 디바이스를 파일 옆에 남긴다."""
    import os

    Path(f"{file}.done").write_text(f"{device}|{os.getpid()}", encoding="utf-8")
    return "bad" not in file.name


class TestParallelWorkers:
    def test_each_worker_gets_its_device(self, tmp_path: Path) -> None:
        files = [tmp_path / f"{i}.wav" for i in range(4)] + [tmp_path / "bad.wav"]
        for f in files:
            f.write_text("x")

        ok, fail, failed = process_batch(
            files,
            _record_device,
            workers=2,
            devices=["cuda", "cpu"],
            failed_list_path=str(tmp_path / "failed.txt"),
        )

        assert (ok, fail, failed) == (4, 1, [tmp_path / "bad.wav"])
        records = [Path(f"{f}.done").read_text().split("|") for f in files]
        # 두 worker 의 디바이스만 쓰이고, 같은 프로세스는 늘 같은 디바이스다
        assert {device for device, _ in records} <= {"cuda", "cpu"}
        by_pid: dict[str, set[str]] = {}
        for device, pid in records:
            by_pid.setdefault(pid, set()).add(device)
        assert all(len(devices) == 1 for devices in by_pid.values())
        assert (tmp_path / "failed.txt").read_text() == str(tmp_path / "bad.wav")

    def test_sequential_uses_first_device(self, tmp_path: Path) -> None:
        seen: list[str | None] = []
        f = tmp_path / "a.wav"
        f.write_text("x")
        process_batch([f], lambda p, device: seen.append(device) or True, devices=["mps"])
        assert seen == ["mps"]


class TestResolveInputs:
    def test_single_file_is_not_batch(self, tmp_path: Path) -> None:
        from src.main import _resolve_inputs

        f = tmp_path / "a.mp4"
        f.write_text("x")
        assert _resolve_inputs(str(f), False, None) == ([f], False)

    def test_directory_glob_and_list(self, tmp_path: Path) -> None:
        from src.main import _resolve_inputs

        (tmp_path / "sub").mkdir()
        for name in ("a.mp4", "b.txt", "sub/c.wav"):
            (tmp_path / name).write_text("x")
        listed = tmp_path / "list.txt"
        listed.write_text(f"# 주석\n{tmp_path / 'a.mp4'}\n", encoding="utf-8")

        assert _resolve_inputs(str(tmp_path), False, None)[0] == [(tmp_path / "a.mp4").resolve()]
        assert len(_resolve_inputs(str(tmp_path), True, None)[0]) == 2
        assert _resolve_inputs(str(tmp_path / "**" / "*.wav"), True, None) == (
            [(tmp_path / "sub" / "c.wav").resolve()],
            True,
        )
        assert _resolve_inputs(None, False, str(listed)) == ([(tmp_path / "a.mp4").resolve()], True)

    def test_errors(self, tmp_path: Path) -> None:
        import click
        import pytest

        from src.main import _resolve_inputs

        with pytest.raises(click.UsageError, match="필요"):
            _resolve_inputs(None, False, None)
        with pytest.raises(click.UsageError, match="찾을 수 없습니다"):
            _resolve_inputs(str(tmp_path / "nope.mp4"), False, None)
        with pytest.raises(click.UsageError, match="처리할 파일이 없습니다"):
            _resolve_inputs(str(tmp_path), False, None)


class TestBatchDevices:
    @staticmethod
    def _config(device: str) -> object:
        from src.core.config import AppConfig

        config = AppConfig()
        config.stt.device = device
        return config

    def test_rules(self) -> None:
        from src.main import _batch_devices

        assert _batch_devices(self._config("cuda"), 1, None) is None  # 순차는 설정 값 그대로
        assert _batch_devices(self._config("cuda"), 3, None) == [
            "cuda",
            "cpu",
            "cpu",
        ]  # GPU 는 하나만
        assert _batch_devices(self._config("cpu"), 2, None) == ["cpu", "cpu"]
        assert _batch_devices(self._config("cpu"), 2, "mps") == ["mps", "mps"]
        assert _batch_devices(self._config("cuda"), 1, "cpu") == ["cpu"]


class TestTranscribeJob:
    def test_reuses_pipeline_and_resets_checkpoint(self, tmp_path: Path) -> None:
        from src.core.config import AppConfig
        from src.main import TranscribeJob
        from src.pipeline import PipelineOptions

        files = [tmp_path / "a.wav", tmp_path / "b.wav"]
        for f in files:
            f.write_text("x")
        stale = Path(f"{files[0]}.checkpoint.json")
        stale.write_text("{}")
        stale_stages = Path(f"{files[0]}.work")
        stale_stages.mkdir()
        (stale_stages / "audio.wav").write_text("x")
        created: list[str] = []

        class FakePipeline:
            def __init__(self, config: AppConfig, keep_engine: bool = False) -> None:
                created.append(config.stt.device)
                self.closed = False

            def run(self, path: str, options: PipelineOptions) -> object:
                from types import SimpleNamespace

                return SimpleNamespace(outputs=[])

            def close(self) -> None:
                self.closed = True

        job = TranscribeJob(config=AppConfig(), options=PipelineOptions(), reset_checkpoint=True)
        with patch("src.main.TranscriptionPipeline", FakePipeline):
            assert job(files[0], "cpu") is True
            assert job(files[1], "cpu") is True
            pipeline = job._pipeline
            job.close()

        assert created == ["cpu"]  # 파일 둘에 모델(파이프라인) 하나
        assert pipeline is not None and pipeline.closed  # type: ignore[attr-defined]
        assert not stale.exists()  # --no-resume
        assert not stale_stages.exists()


class TestTranscribeResumeOptions:
    def test_transcribe_keeps_stages_and_resumes(self, tmp_path: Path) -> None:
        """transcribe 는 단계 결과를 남기고(<입력>.work), --resume 이면 그것에서 이어서 한다."""
        from click.testing import CliRunner

        from src.main import TranscribeJob, cli

        media = tmp_path / "a.wav"
        media.write_text("x")
        seen: list[TranscribeJob] = []

        def fake_run_one(self: TranscribeJob, file: Path, device: str | None = None) -> None:
            seen.append(self)

        with patch.object(TranscribeJob, "run_one", fake_run_one):
            for flag in ("--resume", "--no-resume"):
                result = CliRunner().invoke(cli, ["transcribe", str(media), flag])
                assert result.exit_code == 0, result.output

        resumed, reset = seen
        assert resumed.options.keep_stages and resumed.options.resume
        assert resumed.options.work_dir is None  # 파일마다 입력 옆 폴더
        assert reset.options.keep_stages and not reset.options.resume
        assert reset.reset_checkpoint
