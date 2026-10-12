"""단계 결과 보관 테스트 (다시 시도할 때 끝낸 단계를 건너뛴다)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.audio.vad import SpeechSegment
from src.diarization.diarizer import DiarizationResult, SpeakerTurn
from src.stage_cache import StageCache, get_work_dir, input_fingerprint, stage_key

if TYPE_CHECKING:
    from pathlib import Path


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "enhanced.wav"
    path.write_bytes(b"RIFF fake wav")
    return path


class TestKeys:
    def test_work_dir_next_to_input(self, tmp_path: Path) -> None:
        assert get_work_dir(tmp_path / "a.mp4") == tmp_path / "a.mp4.work"

    def test_fingerprint_changes_with_content(self, tmp_path: Path) -> None:
        source = tmp_path / "a.mp4"
        source.write_bytes(b"one")
        before = input_fingerprint(source)
        source.write_bytes(b"two two")
        assert input_fingerprint(source) != before

    def test_stage_key_is_order_independent(self) -> None:
        assert stage_key({"a": 1, "b": [1, 2]}) == stage_key({"b": [1, 2], "a": 1})
        assert stage_key({"a": 1}) != stage_key({"a": 2})


class TestAudio:
    def test_save_and_load(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path / "work")
        cache.save_audio("k1", _audio(tmp_path), 12.5)

        cached = cache.load_audio("k1")
        assert cached is not None
        assert cached.path == tmp_path / "work" / "audio.wav"
        assert cached.path.read_bytes() == b"RIFF fake wav"
        assert cached.duration_s == 12.5

    def test_other_key_is_not_used(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path / "work")
        cache.save_audio("k1", _audio(tmp_path), 1.0)
        assert cache.load_audio("k2") is None

    def test_missing_file_is_not_used(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path / "work")
        cache.save_audio("k1", _audio(tmp_path), 1.0)
        (tmp_path / "work" / "audio.wav").unlink()
        assert cache.load_audio("k1") is None

    def test_save_failure_only_warns(self, tmp_path: Path) -> None:
        blocker = tmp_path / "work"
        blocker.write_text("폴더 자리에 파일", encoding="utf-8")
        cache = StageCache(blocker)
        cache.save_audio("k1", _audio(tmp_path), 1.0)
        assert cache.load_audio("k1") is None


class TestChunksAndDiarization:
    def test_chunks_roundtrip(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path / "work")
        chunks = [SpeechSegment(0.5, 3.0), SpeechSegment(4.0, 5.0, skip_alignment=True)]
        cache.save_chunks("v1", chunks)

        assert cache.load_chunks("v1") == chunks
        assert cache.load_chunks("v2") is None

    def test_diarization_roundtrip(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path / "work")
        result = DiarizationResult(
            turns=[SpeakerTurn("SPEAKER_00", 0.0, 2.0), SpeakerTurn("SPEAKER_01", 2.0, 4.0)],
            num_speakers=2,
            embeddings={"SPEAKER_00": [0.1, 0.2]},
        )
        cache.save_diarization("d1", result)

        loaded = cache.load_diarization("d1")
        assert loaded is not None
        assert loaded.turns == result.turns
        assert loaded.num_speakers == 2
        assert loaded.embeddings == {"SPEAKER_00": [0.1, 0.2]}

    def test_stages_are_kept_together(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path / "work")
        cache.save_audio("a", _audio(tmp_path), 1.0)
        cache.save_chunks("v", [SpeechSegment(0.0, 1.0)])
        assert cache.load_audio("a") is not None
        assert cache.load_chunks("v") is not None


class TestBrokenManifest:
    def test_unreadable_manifest_is_ignored(self, tmp_path: Path) -> None:
        work = tmp_path / "work"
        work.mkdir()
        (work / "stages.json").write_text("{깨진 JSON", encoding="utf-8")
        assert StageCache(work).load_chunks("v") is None

    def test_other_version_is_ignored(self, tmp_path: Path) -> None:
        work = tmp_path / "work"
        cache = StageCache(work)
        cache.save_chunks("v", [SpeechSegment(0.0, 1.0)])
        data = json.loads((work / "stages.json").read_text(encoding="utf-8"))
        data["version"] = 99
        (work / "stages.json").write_text(json.dumps(data), encoding="utf-8")
        assert cache.load_chunks("v") is None

    def test_malformed_entry_is_ignored(self, tmp_path: Path) -> None:
        work = tmp_path / "work"
        work.mkdir()
        stages = {
            "vad": {"key": "v", "chunks": [{"start": "x"}]},
            "diarize": {"key": "d", "turns": "not a list"},
        }
        (work / "stages.json").write_text(
            json.dumps({"version": 1, "stages": stages}), encoding="utf-8"
        )
        cache = StageCache(work)
        assert cache.load_chunks("v") is None
        assert cache.load_diarization("d") is None


class TestClear:
    def test_clear_removes_folder(self, tmp_path: Path) -> None:
        cache = StageCache(tmp_path / "work")
        cache.save_audio("a", _audio(tmp_path), 1.0)
        cache.clear()
        assert not (tmp_path / "work").exists()

    def test_clear_without_folder(self, tmp_path: Path) -> None:
        StageCache(tmp_path / "none").clear()
