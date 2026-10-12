"""체크포인트 단위 테스트."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.checkpoint import (
    compute_config_hash,
    delete_checkpoint,
    get_checkpoint_path,
    get_remaining_chunks,
    load_checkpoint,
    save_checkpoint,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestConfigHash:
    def test_deterministic(self) -> None:
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"b": 2, "a": 1})  # key order
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_different_configs(self) -> None:
        h1 = compute_config_hash({"model": "whisperx"})
        h2 = compute_config_hash({"model": "faster-whisper"})
        assert h1 != h2


class TestCheckpointPath:
    def test_path(self) -> None:
        assert str(get_checkpoint_path("video.mp4")) == "video.mp4.checkpoint.json"


class TestSaveAndLoad:
    def test_save_load(self, tmp_path: Path) -> None:
        cp = tmp_path / "cp.json"
        h = compute_config_hash({"x": 1})

        save_checkpoint("v.mp4", h, 10, [0, 1, 2], [{"text": "hi"}], cp)
        assert cp.exists()

        loaded = load_checkpoint("v.mp4", h, cp)
        assert loaded is not None
        assert loaded["completed_chunks"] == [0, 1, 2]
        assert loaded["total_chunks"] == 10
        assert loaded["last_completed_chunk"] == 2
        assert loaded["version"] == "1.0"

    def test_hash_mismatch(self, tmp_path: Path) -> None:
        cp = tmp_path / "cp.json"
        save_checkpoint("v.mp4", "sha256:aaa", 5, [0], [], cp)
        assert load_checkpoint("v.mp4", "sha256:bbb", cp) is None

    def test_nonexistent(self) -> None:
        assert load_checkpoint("v.mp4", "h", "/nonexistent.json") is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        cp = tmp_path / "cp.json"
        cp.write_text("not json")
        assert load_checkpoint("v.mp4", "h", cp) is None

    def test_preserves_created_at(self, tmp_path: Path) -> None:
        cp = tmp_path / "cp.json"
        h = "sha256:abc"
        save_checkpoint("v.mp4", h, 5, [0], [], cp)
        created = json.loads(cp.read_text())["created_at"]

        save_checkpoint("v.mp4", h, 5, [0, 1], [], cp)
        assert json.loads(cp.read_text())["created_at"] == created

    def test_empty_completed(self, tmp_path: Path) -> None:
        cp = tmp_path / "cp.json"
        save_checkpoint("v.mp4", "h", 3, [], [], cp)
        data = json.loads(cp.read_text())
        assert data["last_completed_chunk"] == -1


class TestDelete:
    def test_delete(self, tmp_path: Path) -> None:
        cp = tmp_path / "cp.json"
        cp.write_text("{}")
        assert delete_checkpoint("v.mp4", cp) is True
        assert not cp.exists()

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        assert delete_checkpoint("v.mp4", tmp_path / "nope.json") is False


class TestRemainingChunks:
    def test_partial(self) -> None:
        assert get_remaining_chunks({"total_chunks": 5, "completed_chunks": [0, 2, 4]}) == [1, 3]

    def test_all_done(self) -> None:
        assert get_remaining_chunks({"total_chunks": 3, "completed_chunks": [0, 1, 2]}) == []

    def test_none_done(self) -> None:
        assert get_remaining_chunks({"total_chunks": 3, "completed_chunks": []}) == [0, 1, 2]
