"""화자 분리 단위 테스트 (Mock pyannote) + 화자 매핑 + 임베딩."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.core.config import DiarizationConfig
from src.core.exceptions import DiarizationError
from src.diarization.diarizer import (
    DiarizationResult,
    Diarizer,
    SpeakerTurn,
    _convert_annotation,
    _find_best_speaker,
    assign_speakers,
)
from src.diarization.embedding_store import export_embeddings, load_embeddings
from src.diarization.speaker_mapper import (
    apply_speaker_names,
    build_speaker_map,
    load_speaker_profile,
    parse_speaker_map_string,
)
from src.models.segment import TranscriptionSegment

if TYPE_CHECKING:
    from pathlib import Path

# ──────────────────────────────────────────────
# SpeakerTurn / DiarizationResult
# ──────────────────────────────────────────────


class TestSpeakerTurn:
    def test_duration(self) -> None:
        t = SpeakerTurn("A", 1.0, 3.5)
        assert t.duration == pytest.approx(2.5)

    def test_to_dict(self) -> None:
        d = SpeakerTurn("A", 0.5, 2.0).to_dict()
        assert d == {"speaker": "A", "start": 0.5, "end": 2.0, "duration": 1.5}


class TestDiarizationResult:
    def test_speakers(self) -> None:
        r = DiarizationResult(
            turns=[SpeakerTurn("B", 0, 1), SpeakerTurn("A", 1, 2), SpeakerTurn("B", 3, 4)],
            num_speakers=2,
        )
        assert r.speakers == ["A", "B"]

    def test_empty(self) -> None:
        r = DiarizationResult()
        assert r.speakers == []
        assert r.num_speakers == 0


# ──────────────────────────────────────────────
# Diarizer
# ──────────────────────────────────────────────


class TestDiarizer:
    def test_diarize_before_load(self) -> None:
        d = Diarizer(DiarizationConfig())
        with pytest.raises(DiarizationError, match="로드되지 않았습니다"):
            d.diarize("test.wav")

    def test_load_without_token(self) -> None:
        import os

        os.environ.pop("HF_TOKEN", None)
        d = Diarizer(DiarizationConfig(hf_token=""))
        with pytest.raises(DiarizationError):
            # pyannote 미설치 시 import 에러, 설치 시 토큰 에러
            d.load()

    def test_convert_annotation(self) -> None:
        mock = MagicMock()
        t1, t2 = MagicMock(), MagicMock()
        t1.start, t1.end = 0.0, 2.0
        t2.start, t2.end = 2.5, 4.0
        mock.itertracks.return_value = [(t1, None, "A"), (t2, None, "B")]
        turns = _convert_annotation(mock)
        assert len(turns) == 2
        assert turns[0].speaker == "A"
        assert turns[1].start == 2.5


# ──────────────────────────────────────────────
# assign_speakers
# ──────────────────────────────────────────────


class TestAssignSpeakers:
    def test_basic_assignment(self) -> None:
        segs = [
            TranscriptionSegment(1, "Hello", 0.0, 1.5),
            TranscriptionSegment(2, "World", 2.5, 4.0),
        ]
        diar = DiarizationResult(
            turns=[SpeakerTurn("A", 0.0, 2.0), SpeakerTurn("B", 2.0, 5.0)],
            num_speakers=2,
        )
        assign_speakers(segs, diar)
        assert segs[0].speaker == "A"
        assert segs[1].speaker == "B"

    def test_empty_diarization(self) -> None:
        segs = [TranscriptionSegment(1, "Hi", 0.0, 1.0)]
        assign_speakers(segs, DiarizationResult())
        assert segs[0].speaker is None

    def test_no_overlap(self) -> None:
        segs = [TranscriptionSegment(1, "Hi", 10.0, 11.0)]
        diar = DiarizationResult(turns=[SpeakerTurn("A", 0.0, 2.0)], num_speakers=1)
        assign_speakers(segs, diar)
        assert segs[0].speaker is None


class TestFindBestSpeaker:
    def test_max_overlap(self) -> None:
        turns = [SpeakerTurn("A", 0.0, 2.0), SpeakerTurn("B", 1.0, 4.0)]
        assert _find_best_speaker(0.0, 1.5, turns) == "A"
        assert _find_best_speaker(1.5, 4.0, turns) == "B"

    def test_no_match(self) -> None:
        assert _find_best_speaker(10.0, 11.0, [SpeakerTurn("A", 0.0, 1.0)]) is None


# ──────────────────────────────────────────────
# Speaker Mapper
# ──────────────────────────────────────────────


class TestParseMapString:
    def test_basic(self) -> None:
        m = parse_speaker_map_string("SPEAKER_00:John,SPEAKER_01:Jane")
        assert m == {"SPEAKER_00": "John", "SPEAKER_01": "Jane"}

    def test_empty(self) -> None:
        assert parse_speaker_map_string("") == {}

    def test_invalid_pair(self) -> None:
        assert parse_speaker_map_string("nocolon") == {}

    def test_mixed(self) -> None:
        m = parse_speaker_map_string("A:B,bad,C:D")
        assert m == {"A": "B", "C": "D"}


class TestLoadProfile:
    def test_detailed_format(self, tmp_path: Path) -> None:
        f = tmp_path / "p.json"
        f.write_text(json.dumps({"speakers": {"A": {"name": "Alice"}}}))
        assert load_speaker_profile(f) == {"A": "Alice"}

    def test_simple_format(self, tmp_path: Path) -> None:
        f = tmp_path / "p.json"
        f.write_text(json.dumps({"A": "Alice", "B": "Bob"}))
        assert load_speaker_profile(f) == {"A": "Alice", "B": "Bob"}

    def test_nonexistent(self) -> None:
        assert load_speaker_profile("/nonexistent.json") == {}


class TestBuildSpeakerMap:
    def test_priority(self, tmp_path: Path) -> None:
        f = tmp_path / "p.json"
        f.write_text(json.dumps({"A": "Profile"}))
        m = build_speaker_map(
            config_map={"A": "Config"},
            cli_map_string="A:CLI",
            profile_path=f,
        )
        assert m["A"] == "CLI"  # CLI wins


class TestApplyNames:
    def test_single(self) -> None:
        segs = [TranscriptionSegment(1, "x", 0, 1, speaker="A")]
        apply_speaker_names(segs, {"A": "Alice"})
        assert segs[0].speaker == "Alice"

    def test_multi(self) -> None:
        segs = [TranscriptionSegment(1, "x", 0, 1, speaker=["A", "B"])]
        apply_speaker_names(segs, {"A": "Alice", "B": "Bob"})
        assert segs[0].speaker == ["Alice", "Bob"]

    def test_unmapped(self) -> None:
        segs = [TranscriptionSegment(1, "x", 0, 1, speaker="Z")]
        apply_speaker_names(segs, {"A": "Alice"})
        assert segs[0].speaker == "Z"

    def test_empty_map(self) -> None:
        segs = [TranscriptionSegment(1, "x", 0, 1, speaker="A")]
        apply_speaker_names(segs, {})
        assert segs[0].speaker == "A"


# ──────────────────────────────────────────────
# Embedding Store
# ──────────────────────────────────────────────


class TestEmbeddingStore:
    def test_export_and_load(self, tmp_path: Path) -> None:
        emb = {"A": [0.1, 0.2], "B": [0.3, 0.4]}
        out = export_embeddings(emb, ["A", "B"], tmp_path / "e.json", "src.wav")
        data = load_embeddings(out)
        assert data["speakers"] == ["A", "B"]
        assert data["embeddings"]["A"] == [0.1, 0.2]

    def test_load_nonexistent(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_embeddings("/nonexistent.json")

    def test_load_invalid(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json")
        with pytest.raises(ValueError):
            load_embeddings(f)

    def test_export_none(self, tmp_path: Path) -> None:
        out = export_embeddings(None, ["A"], tmp_path / "e.json")
        data = load_embeddings(out)
        assert data["embeddings"] == {}


class TestCheckDiarizationAccess:
    """화자 분리 파이프라인을 음성 인식 전에 불러와 본다 (작업 #1: 1시간 인식 뒤 401·403 실패)."""

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)

    @staticmethod
    def _fake_pipeline(monkeypatch: pytest.MonkeyPatch, from_pretrained: MagicMock) -> None:
        import sys
        import types

        module = types.ModuleType("pyannote.audio")
        module.Pipeline = MagicMock(from_pretrained=from_pretrained)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
        monkeypatch.setitem(sys.modules, "pyannote.audio", module)

    def test_missing_token(self) -> None:
        from src.diarization.diarizer import check_diarization_access

        with pytest.raises(DiarizationError, match="토큰이 필요"):
            check_diarization_access(DiarizationConfig(enabled=True))

    def test_loads_then_unloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.diarization.diarizer import DIARIZATION_MODEL, Diarizer, check_diarization_access

        from_pretrained = MagicMock(return_value=object())
        self._fake_pipeline(monkeypatch, from_pretrained)
        unloads: list[bool] = []
        original_unload = Diarizer.unload

        def track_unload(self: Diarizer) -> None:
            unloads.append(self._is_loaded)
            original_unload(self)

        monkeypatch.setattr(Diarizer, "unload", track_unload)

        check_diarization_access(DiarizationConfig(enabled=True, hf_token="hf_x"))

        assert from_pretrained.call_args.args[0] == DIARIZATION_MODEL
        # 메모리를 붙잡지 않게 바로 내린다 (받은 모델은 캐시에 남는다)
        assert unloads == [True]

    def test_gated_dependency_fails_with_every_model_to_accept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.diarization.diarizer import GATED_MODELS, check_diarization_access

        gated = RuntimeError(
            "403 Client Error.\nAccess to model pyannote/speaker-diarization-community-1 "
            "is restricted and you are not in the authorized list."
        )
        self._fake_pipeline(monkeypatch, MagicMock(side_effect=gated))

        with pytest.raises(DiarizationError) as caught:
            check_diarization_access(DiarizationConfig(enabled=True, hf_token="hf_x"))

        message = str(caught.value)
        assert all(repo in message for repo in GATED_MODELS)
        assert message.endswith("is restricted and you are not in the authorized list.")

    def test_pyannote_missing_keeps_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from src.diarization.diarizer import check_diarization_access

        monkeypatch.setitem(sys.modules, "pyannote.audio", None)
        with pytest.raises(DiarizationError, match=r"requirements\.txt"):
            check_diarization_access(DiarizationConfig(enabled=True, hf_token="hf_x"))


class TestSpeakerEmbeddings:
    """화자 임베딩 채우기·재사용 (--export-embeddings, --speaker-embeddings)."""

    def test_embeddings_from_pyannote4_output(self) -> None:
        import numpy as np

        from src.diarization.diarizer import _speaker_embeddings

        annotation = MagicMock()
        annotation.labels.return_value = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
        output = MagicMock(spec=["speaker_embeddings"])
        output.speaker_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]])

        result = _speaker_embeddings(output, annotation)

        # 임베딩을 못 만든 화자(NaN)는 뺀다
        assert result == {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}

    def test_no_embeddings_in_output(self) -> None:
        from src.diarization.diarizer import _speaker_embeddings

        assert _speaker_embeddings(object(), MagicMock()) is None

    def test_match_reuses_known_ids_and_adds_new(self) -> None:
        from src.diarization.embedding_store import match_known_speakers

        known = {"SPEAKER_00": [1.0, 0.0, 0.0], "SPEAKER_01": [0.0, 1.0, 0.0]}
        current = {
            "SPEAKER_00": [0.1, 0.99, 0.0],  # 이전 SPEAKER_01 과 같은 사람
            "SPEAKER_01": [0.98, 0.05, 0.0],  # 이전 SPEAKER_00 과 같은 사람
            "SPEAKER_02": [0.0, 0.0, 1.0],  # 새 사람
        }

        mapping = match_known_speakers(current, known)

        assert mapping == {
            "SPEAKER_00": "SPEAKER_01",
            "SPEAKER_01": "SPEAKER_00",
            "SPEAKER_02": "SPEAKER_02",
        }

    def test_new_speaker_does_not_take_known_id(self) -> None:
        from src.diarization.embedding_store import match_known_speakers

        known = {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}
        current = {"SPEAKER_00": [-1.0, 0.0]}  # 누구와도 닮지 않음

        assert match_known_speakers(current, known) == {"SPEAKER_00": "SPEAKER_02"}

    def test_one_known_speaker_matched_once(self) -> None:
        from src.diarization.embedding_store import match_known_speakers

        known = {"SPEAKER_00": [1.0, 0.0]}
        current = {"SPEAKER_00": [0.9, 0.1], "SPEAKER_01": [0.95, 0.05]}

        mapping = match_known_speakers(current, known)

        assert sorted(mapping.values()) == ["SPEAKER_00", "SPEAKER_01"]
        assert mapping["SPEAKER_01"] == "SPEAKER_00"  # 더 닮은 쪽이 이전 ID 를 받는다

    def test_known_embeddings_skips_bad_rows(self) -> None:
        from src.diarization.embedding_store import known_embeddings

        data = {"embeddings": {"SPEAKER_00": [1, 2.5], "SPEAKER_01": "x", "SPEAKER_02": []}}
        assert known_embeddings(data) == {"SPEAKER_00": [1.0, 2.5]}
        assert known_embeddings({"speakers": []}) == {}

    def test_relabel_updates_turns_and_embeddings(self) -> None:
        result = DiarizationResult(
            turns=[SpeakerTurn("SPEAKER_00", 0.0, 1.0), SpeakerTurn("SPEAKER_01", 1.0, 2.0)],
            num_speakers=2,
            embeddings={"SPEAKER_00": [1.0], "SPEAKER_01": [2.0]},
        )
        result.relabel({"SPEAKER_00": "SPEAKER_05"})
        assert [t.speaker for t in result.turns] == ["SPEAKER_05", "SPEAKER_01"]
        assert result.embeddings == {"SPEAKER_05": [1.0], "SPEAKER_01": [2.0]}

    def test_export_then_reuse_roundtrip(self, tmp_path: Path) -> None:
        from src.diarization.embedding_store import export_embeddings
        from src.pipeline import _reuse_speaker_ids

        path = export_embeddings(
            {"SPEAKER_00": [1.0, 0.0]}, ["SPEAKER_00"], tmp_path / "ep1.embeddings.json"
        )
        episode2 = DiarizationResult(
            turns=[SpeakerTurn("SPEAKER_00", 0.0, 1.0), SpeakerTurn("SPEAKER_01", 1.0, 2.0)],
            num_speakers=2,
            embeddings={"SPEAKER_00": [0.0, 1.0], "SPEAKER_01": [0.99, 0.01]},
        )

        _reuse_speaker_ids(episode2, str(path))

        assert [t.speaker for t in episode2.turns] == ["SPEAKER_01", "SPEAKER_00"]

    def test_reuse_unreadable_file(self, tmp_path: Path) -> None:
        from src.pipeline import _reuse_speaker_ids

        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(DiarizationError, match="화자 임베딩 파일"):
            _reuse_speaker_ids(DiarizationResult(embeddings={"SPEAKER_00": [1.0]}), str(bad))
