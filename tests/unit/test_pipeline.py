"""전체 파이프라인 테스트 (ffmpeg·모델 없이 추출기·VAD·엔진을 가짜로 바꾼다)."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from src.audio.vad import SpeechSegment
from src.core.config import AppConfig, STTConfig
from src.core.exceptions import DiarizationError
from src.diarization.diarizer import DiarizationResult, SpeakerTurn
from src.engines.base import STTEngine, write_wav_slice
from src.models.result import Metadata, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.pipeline import (
    STAGE_LOAD_MODEL,
    STAGE_TRANSCRIBE,
    PipelineOptions,
    TranscriptionPipeline,
    checkpoint_config_hash,
    ensure_words,
    restore_sentence_end,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

SAMPLE_RATE = 16_000
AUDIO_SECONDS = 8


def _write_wav(path: Path, seconds: float = AUDIO_SECONDS) -> Path:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(b"\x00\x00" * int(SAMPLE_RATE * seconds))
    return path


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


class FakeEngine(STTEngine):
    """WAV 길이에 맞춰 단어 두 개짜리 세그먼트 하나를 돌려주는 엔진."""

    def __init__(self, config: STTConfig, language: str = "en") -> None:
        super().__init__(config)
        self.language = language
        self.calls: list[tuple[float, str | None]] = []
        self.loads = 0
        self.unloads = 0

    @property
    def engine_name(self) -> str:
        return "fake"

    def load_model(self) -> None:
        self.loads += 1
        self.is_loaded = True

    def unload_model(self) -> None:
        self.unloads += 1
        self.is_loaded = False

    def transcribe(
        self, audio_path: str | Path, language: str | None = None
    ) -> TranscriptionResult:
        seconds = _wav_seconds(Path(audio_path))
        self.calls.append((seconds, language))
        index = len(self.calls)
        end = max(0.6, seconds - 0.1)
        words = [
            WordSegment(word=f"hello{index}", start=0.1, end=end / 2, confidence=0.95),
            WordSegment(word="world.", start=end / 2 + 0.05, end=end, confidence=0.95),
        ]
        segment = TranscriptionSegment(
            index=1, text=f"hello{index} world.", start=0.1, end=end, confidence=0.95, words=words
        )
        return TranscriptionResult(
            metadata=Metadata(language=self.language, model="fake/small"), segments=[segment]
        )


class FakeExtractor:
    """ffmpeg 대신 미리 만든 WAV 를 돌려준다."""

    def __init__(self, wav: Path) -> None:
        self.wav = wav
        self.extracted: list[dict[str, Any]] = []

    def __call__(self, keep_audio: bool = False) -> FakeExtractor:
        return self

    def __enter__(self) -> FakeExtractor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def extract(self, input_path: str | Path, **kwargs: Any) -> Path:
        self.extracted.append({"input": str(input_path), **kwargs})
        return self.wav


@pytest.fixture()
def wav(tmp_path: Path) -> Path:
    return _write_wav(tmp_path / "audio.wav")


@pytest.fixture()
def source(tmp_path: Path) -> Path:
    path = tmp_path / "movie.mp4"
    path.write_bytes(b"not really a movie")
    return path


@pytest.fixture()
def config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.output.dir = str(tmp_path / "out")
    cfg.audio.normalize = False
    cfg.stt.source_lang = "auto"
    return cfg


@pytest.fixture()
def env(wav: Path) -> Iterator[dict[str, Any]]:
    """추출기·길이·VAD·엔진을 가짜로 바꾼다."""
    extractor = FakeExtractor(wav)
    engine = FakeEngine(STTConfig())
    chunks = [SpeechSegment(start=0.5, end=3.0), SpeechSegment(start=4.0, end=7.5)]
    with (
        patch("src.audio.extractor.AudioExtractor", extractor),
        patch("src.audio.extractor.get_audio_duration", return_value=float(AUDIO_SECONDS)),
        patch("src.audio.vad.detect_speech", return_value=chunks) as detect,
        patch("src.engines.engine_factory.create_engine", return_value=engine),
    ):
        yield {"extractor": extractor, "engine": engine, "detect": detect, "chunks": chunks}


class TestPreVadPipeline:
    """Pre-VAD 청크별 추론 경로."""

    def test_chunks_offset_and_written(
        self, config: AppConfig, source: Path, env: dict[str, Any]
    ) -> None:
        events: list[tuple[str, int, int]] = []
        with TranscriptionPipeline(config, progress=lambda s, d, t: events.append((s, d, t))) as p:
            outcome = p.run(source)

        engine: FakeEngine = env["engine"]
        # 청크마다 한 번씩 추론하고 모델은 한 번 로드·언로드한다
        assert len(engine.calls) == 2
        assert engine.loads == 1 and engine.unloads == 1
        # 첫 청크에서 감지한 언어를 다음 청크에 고정한다
        assert engine.calls[0][1] is None and engine.calls[1][1] == "en"
        words = [w for seg in outcome.result.segments for w in seg.words]
        assert words[0].start == pytest.approx(0.6)
        assert any(w.start >= 4.0 for w in words)
        assert outcome.result.metadata.language == "en"
        assert outcome.result.metadata.duration_s == AUDIO_SECONDS
        assert [s.index for s in outcome.result.segments] == list(
            range(1, len(outcome.result.segments) + 1)
        )
        assert (STAGE_TRANSCRIBE, 2, 2) in events
        assert len(outcome.outputs) == 1 and outcome.outputs[0].suffix == ".srt"
        # 텍스트 정규화가 문장 첫 글자를 대문자로 바꾼다
        assert "hello1" in outcome.outputs[0].read_text(encoding="utf-8").lower()
        assert not Path(str(source) + ".checkpoint.json").exists()

    def test_resume_skips_finished_chunks(
        self, config: AppConfig, source: Path, env: dict[str, Any], tmp_path: Path
    ) -> None:
        from src.checkpoint import save_checkpoint

        checkpoint = tmp_path / "ck.json"
        options = PipelineOptions(resume=True, checkpoint_path=str(checkpoint))
        config_hash = checkpoint_config_hash(config, options, env["chunks"])
        done_segment = TranscriptionSegment(
            index=1,
            text="from checkpoint",
            start=0.6,
            end=2.0,
            words=[
                WordSegment(word="from", start=0.6, end=1.2),
                WordSegment(word="checkpoint", start=1.3, end=2.0),
            ],
        )
        save_checkpoint(
            str(source),
            config_hash,
            2,
            [0],
            [
                {
                    "chunk": 0,
                    "language": "ko",
                    "model": "fake/small",
                    "segments": [done_segment.to_dict()],
                }
            ],
            checkpoint,
        )
        with TranscriptionPipeline(config) as p:
            outcome = p.run(source, options)

        engine: FakeEngine = env["engine"]
        assert len(engine.calls) == 1
        # 체크포인트의 언어를 남은 청크에 그대로 쓴다
        assert engine.calls[0][1] == "ko"
        text = " ".join(seg.text for seg in outcome.result.segments).lower()
        assert "checkpoint" in text and "hello1" in text
        assert not checkpoint.exists()

    def test_checkpoint_saved_per_chunk(
        self, config: AppConfig, source: Path, env: dict[str, Any], tmp_path: Path
    ) -> None:
        checkpoint = tmp_path / "ck.json"
        engine: FakeEngine = env["engine"]
        original = engine.transcribe
        saved: list[list[int]] = []

        def transcribe_and_peek(audio_path: str | Path, language: str | None = None) -> Any:
            if checkpoint.exists():
                saved.append(json.loads(checkpoint.read_text())["completed_chunks"])
            return original(audio_path, language)

        engine.transcribe = transcribe_and_peek  # type: ignore[method-assign]
        with TranscriptionPipeline(config) as p:
            p.run(source, PipelineOptions(checkpoint_path=str(checkpoint)))
        assert saved == [[0]]


class TestWholeFilePipeline:
    """Pre-VAD 를 끈 경로와 출력 옵션."""

    def test_no_pre_vad_transcribes_once(
        self, config: AppConfig, source: Path, env: dict[str, Any]
    ) -> None:
        config.audio.vad.pre_vad = False
        with TranscriptionPipeline(config) as p:
            outcome = p.run(source)
        engine: FakeEngine = env["engine"]
        assert len(engine.calls) == 1
        assert engine.calls[0][0] == pytest.approx(AUDIO_SECONDS)
        env["detect"].assert_not_called()
        assert outcome.result.segment_count >= 1

    def test_write_false_returns_result_only(
        self, config: AppConfig, source: Path, env: dict[str, Any]
    ) -> None:
        with TranscriptionPipeline(config) as p:
            outcome = p.run(source, PipelineOptions(write=False))
        assert outcome.outputs == []
        assert not Path(config.output.dir).exists()

    def test_multiple_formats_words_and_report(
        self, config: AppConfig, source: Path, env: dict[str, Any]
    ) -> None:
        config.output.export_words = True
        config.quality.export_report = True
        with TranscriptionPipeline(config) as p:
            outcome = p.run(source, PipelineOptions(formats=["srt", "vtt"], vad_export=True))
        names = sorted(path.name for path in outcome.outputs)
        assert names == sorted(
            [
                "movie.vad.json",
                "movie.en.srt",
                "movie.en.vtt",
                "movie.en.json",
                "movie.quality.json",
            ]
        )
        assert outcome.report is not None

    def test_time_offset_shifts_segments(
        self, config: AppConfig, source: Path, env: dict[str, Any]
    ) -> None:
        with TranscriptionPipeline(config) as p:
            base = p.run(source, PipelineOptions(write=False)).result.segments[0].start
        env["engine"].calls.clear()
        with TranscriptionPipeline(config) as p:
            moved = p.run(source, PipelineOptions(write=False, time_offset_s=100.0))
        assert moved.result.segments[0].start == pytest.approx(base + 100.0)

    def test_keep_engine_reuses_model(
        self, config: AppConfig, source: Path, env: dict[str, Any]
    ) -> None:
        with TranscriptionPipeline(config, keep_engine=True) as p:
            p.run(source, PipelineOptions(write=False))
            p.run(source, PipelineOptions(write=False))
        engine: FakeEngine = env["engine"]
        assert engine.loads == 1 and engine.unloads == 1


class TestEnhanceAndDiarize:
    """전처리·화자 분리 단계 연결."""

    def test_enhanced_audio_feeds_stt_and_is_removed(
        self, config: AppConfig, source: Path, env: dict[str, Any], tmp_path: Path
    ) -> None:
        config.audio.normalize = True
        config.audio.vad.pre_vad = False
        enhanced = _write_wav(tmp_path / "enhanced.wav", seconds=5)
        with (
            patch("src.audio.enhancer.enhance_audio", return_value=enhanced) as enhance,
            TranscriptionPipeline(config) as p,
        ):
            p.run(source)
        enhance.assert_called_once()
        assert enhance.call_args.kwargs["normalize"] is True
        engine: FakeEngine = env["engine"]
        assert engine.calls[0][0] == pytest.approx(5)
        assert not enhanced.exists()

    def test_diarization_assigns_speakers(
        self, config: AppConfig, source: Path, env: dict[str, Any]
    ) -> None:
        config.diarization.enabled = True
        config.diarization.speaker_map = {"SPEAKER_00": "Alice"}
        result = DiarizationResult(
            turns=[
                SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=3.5),
                SpeakerTurn(speaker="SPEAKER_01", start=3.5, end=8.0),
            ],
            num_speakers=2,
        )

        class FakeDiarizer:
            def __init__(self, cfg: Any) -> None:
                pass

            def __enter__(self) -> FakeDiarizer:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def diarize(self, audio_path: Any) -> DiarizationResult:
                return result

        with (
            patch("src.diarization.diarizer.Diarizer", FakeDiarizer),
            # 모델 접근 확인(네트워크)은 아래 테스트에서 따로 본다
            patch("src.diarization.diarizer.check_diarization_access"),
            TranscriptionPipeline(config) as p,
        ):
            outcome = p.run(source)
        speakers = {seg.speaker for seg in outcome.result.segments}
        assert "Alice" in speakers and "SPEAKER_01" in speakers
        assert outcome.num_speakers == 2
        assert ".diarized." in outcome.outputs[0].name

    def test_diarization_access_checked_before_transcription(
        self, config: AppConfig, source: Path, env: dict[str, Any]
    ) -> None:
        # 토큰 문제는 긴 음성 인식 전에 알린다
        config.diarization.enabled = True
        with (
            patch(
                "src.diarization.diarizer.check_diarization_access",
                side_effect=DiarizationError("접근 불가"),
            ),
            TranscriptionPipeline(config) as p,
            pytest.raises(DiarizationError, match="접근 불가"),
        ):
            p.run(source)
        engine: FakeEngine = env["engine"]
        assert engine.calls == []

    def test_no_access_check_without_diarization(
        self, config: AppConfig, source: Path, env: dict[str, Any]
    ) -> None:
        with (
            patch("src.diarization.diarizer.check_diarization_access") as check,
            TranscriptionPipeline(config) as p,
        ):
            p.run(source, PipelineOptions(write=False))
        check.assert_not_called()


class CountingDiarizer:
    """호출 수를 세는 가짜 화자 분리기 (``patch`` 로 ``Diarizer`` 자리에 넣는다)."""

    calls = 0
    result = DiarizationResult(
        turns=[
            SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=3.5),
            SpeakerTurn(speaker="SPEAKER_01", start=3.5, end=8.0),
        ],
        num_speakers=2,
        embeddings={"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]},
    )

    def __init__(self, cfg: Any) -> None:
        pass

    def __enter__(self) -> CountingDiarizer:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def diarize(self, audio_path: Any) -> DiarizationResult:
        CountingDiarizer.calls += 1
        return DiarizationResult(
            turns=list(self.result.turns),
            num_speakers=self.result.num_speakers,
            embeddings=dict(self.result.embeddings),
        )


class TestStageResume:
    """다시 시도할 때 끝낸 단계(전처리·Pre-VAD·화자 분리)를 건너뛴다."""

    @pytest.fixture()
    def work_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "work"

    @pytest.fixture(autouse=True)
    def _reset_diarizer(self) -> None:
        CountingDiarizer.calls = 0

    def _options(self, work_dir: Path, tmp_path: Path, resume: bool) -> PipelineOptions:
        return PipelineOptions(
            resume=resume,
            checkpoint_path=str(tmp_path / "ck.json"),
            keep_stages=True,
            work_dir=str(work_dir),
        )

    def _fail_after_transcribe(self, config: AppConfig) -> Any:
        """음성 인식 뒤(세그멘테이션)에서 실패시킨다: 서버가 중간에 멈춘 상황."""
        return patch.object(
            TranscriptionPipeline, "_segment", side_effect=RuntimeError("서버 중단")
        )

    def test_resume_skips_finished_stages(
        self,
        config: AppConfig,
        source: Path,
        env: dict[str, Any],
        work_dir: Path,
        tmp_path: Path,
    ) -> None:
        config.diarization.enabled = True
        with (
            patch("src.diarization.diarizer.Diarizer", CountingDiarizer),
            patch("src.diarization.diarizer.check_diarization_access"),
        ):
            with (
                self._fail_after_transcribe(config),
                TranscriptionPipeline(config) as p,
                pytest.raises(RuntimeError, match="서버 중단"),
            ):
                p.run(source, self._options(work_dir, tmp_path, resume=False))
            # 실패하면 단계 결과와 체크포인트가 남는다
            assert (work_dir / "audio.wav").is_file()
            assert (tmp_path / "ck.json").is_file()
            extractor: FakeExtractor = env["extractor"]
            engine: FakeEngine = env["engine"]
            assert len(extractor.extracted) == 1 and env["detect"].call_count == 1
            assert len(engine.calls) == 2 and CountingDiarizer.calls == 1

            with TranscriptionPipeline(config) as p:
                outcome = p.run(source, self._options(work_dir, tmp_path, resume=True))

        # 추출·VAD·음성 인식·화자 분리를 다시 하지 않는다
        assert len(extractor.extracted) == 1
        assert env["detect"].call_count == 1
        assert len(engine.calls) == 2
        assert CountingDiarizer.calls == 1
        assert outcome.num_speakers == 2
        assert outcome.result.metadata.duration_s == AUDIO_SECONDS
        assert {seg.speaker for seg in outcome.result.segments} >= {"SPEAKER_00"}
        # 끝나면 지운다
        assert not work_dir.exists()
        assert not (tmp_path / "ck.json").exists()

    def test_without_resume_stages_are_redone(
        self,
        config: AppConfig,
        source: Path,
        env: dict[str, Any],
        work_dir: Path,
        tmp_path: Path,
    ) -> None:
        with (
            self._fail_after_transcribe(config),
            TranscriptionPipeline(config) as p,
            pytest.raises(RuntimeError),
        ):
            p.run(source, self._options(work_dir, tmp_path, resume=False))
        with TranscriptionPipeline(config) as p:
            p.run(source, self._options(work_dir, tmp_path, resume=False))

        assert len(env["extractor"].extracted) == 2
        assert env["detect"].call_count == 2

    def test_changed_input_invalidates_audio(
        self,
        config: AppConfig,
        source: Path,
        env: dict[str, Any],
        work_dir: Path,
        tmp_path: Path,
    ) -> None:
        with (
            self._fail_after_transcribe(config),
            TranscriptionPipeline(config) as p,
            pytest.raises(RuntimeError),
        ):
            p.run(source, self._options(work_dir, tmp_path, resume=False))
        source.write_bytes(b"a different movie with another size")
        with TranscriptionPipeline(config) as p:
            p.run(source, self._options(work_dir, tmp_path, resume=True))

        assert len(env["extractor"].extracted) == 2

    def test_changed_vad_settings_redetect_only(
        self,
        config: AppConfig,
        source: Path,
        env: dict[str, Any],
        work_dir: Path,
        tmp_path: Path,
    ) -> None:
        with (
            self._fail_after_transcribe(config),
            TranscriptionPipeline(config) as p,
            pytest.raises(RuntimeError),
        ):
            p.run(source, self._options(work_dir, tmp_path, resume=False))
        config.audio.vad.threshold = 0.3
        with TranscriptionPipeline(config) as p:
            p.run(source, self._options(work_dir, tmp_path, resume=True))

        # 전처리 음성은 그대로 쓰고 발화 구간만 다시 찾는다
        assert len(env["extractor"].extracted) == 1
        assert env["detect"].call_count == 2

    def test_finished_checkpoint_does_not_load_model(
        self,
        config: AppConfig,
        source: Path,
        env: dict[str, Any],
        work_dir: Path,
        tmp_path: Path,
    ) -> None:
        """모든 청크가 체크포인트에 있으면 STT 모델을 불러오지 않는다."""
        events: list[str] = []
        with (
            self._fail_after_transcribe(config),
            TranscriptionPipeline(config) as p,
            pytest.raises(RuntimeError),
        ):
            p.run(source, self._options(work_dir, tmp_path, resume=False))
        engine: FakeEngine = env["engine"]
        assert engine.loads == 1
        with TranscriptionPipeline(config, progress=lambda s, d, t: events.append(s)) as p:
            p.run(source, self._options(work_dir, tmp_path, resume=True))

        assert engine.loads == 1 and len(engine.calls) == 2
        assert STAGE_LOAD_MODEL not in events

    def test_checkpoint_with_other_chunk_boundaries_is_ignored(
        self,
        config: AppConfig,
        source: Path,
        env: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """청크 수가 같아도 경계가 다르면 체크포인트를 쓰지 않는다."""
        from src.checkpoint import save_checkpoint

        checkpoint = tmp_path / "ck.json"
        options = PipelineOptions(resume=True, checkpoint_path=str(checkpoint))
        other = [SpeechSegment(start=0.2, end=2.0), SpeechSegment(start=5.0, end=7.0)]
        stale = TranscriptionSegment(index=1, text="stale chunk", start=0.2, end=1.0)
        save_checkpoint(
            str(source),
            checkpoint_config_hash(config, options, other),
            2,
            [0, 1],
            [
                {"chunk": i, "language": "en", "model": "fake/small", "segments": [stale.to_dict()]}
                for i in (0, 1)
            ],
            checkpoint,
        )
        with TranscriptionPipeline(config) as p:
            outcome = p.run(source, options)

        assert len(env["engine"].calls) == 2
        assert "stale" not in " ".join(seg.text for seg in outcome.result.segments).lower()

    def test_speaker_reuse_applies_to_cached_diarization(
        self,
        config: AppConfig,
        source: Path,
        env: dict[str, Any],
        work_dir: Path,
        tmp_path: Path,
    ) -> None:
        """보관한 화자 분리 결과는 화자 ID 를 바꾸기 전 값이다 (다시 시도해도 같은 대조)."""
        config.diarization.enabled = True
        known = tmp_path / "known.json"
        known.write_text(
            json.dumps({"speakers": ["SPEAKER_07"], "embeddings": {"SPEAKER_07": [1.0, 0.0]}}),
            encoding="utf-8",
        )
        options = self._options(work_dir, tmp_path, resume=False)
        options.speaker_embeddings = str(known)
        with (
            patch("src.diarization.diarizer.Diarizer", CountingDiarizer),
            patch("src.diarization.diarizer.check_diarization_access"),
        ):
            with (
                self._fail_after_transcribe(config),
                TranscriptionPipeline(config) as p,
                pytest.raises(RuntimeError),
            ):
                p.run(source, options)
            saved = json.loads((work_dir / "stages.json").read_text(encoding="utf-8"))
            assert {t["speaker"] for t in saved["stages"]["diarize"]["turns"]} == {
                "SPEAKER_00",
                "SPEAKER_01",
            }
            options.resume = True
            with TranscriptionPipeline(config) as p:
                outcome = p.run(source, options)

        assert CountingDiarizer.calls == 1
        assert "SPEAKER_07" in {seg.speaker for seg in outcome.result.segments}


class LanguageByLengthEngine(FakeEngine):
    """언어를 지정하지 않으면 3.2초 청크는 pt, 나머지는 ja 로 감지하는 엔진 (첫 청크가 잡음인 상황)."""

    def transcribe(
        self, audio_path: str | Path, language: str | None = None
    ) -> TranscriptionResult:
        result = super().transcribe(audio_path, language)
        seconds = _wav_seconds(Path(audio_path))
        detected = language or ("pt" if abs(seconds - 3.2) < 0.05 else "ja")
        result.metadata.language = detected
        return result


class TestLanguageDetection:
    """언어 자동 감지: 고르게 퍼진 청크 몇 개의 다수결 (첫 청크 하나로 정하지 않는다)."""

    @pytest.fixture()
    def long_env(self, tmp_path: Path) -> Iterator[dict[str, Any]]:
        wav = _write_wav(tmp_path / "long.wav", seconds=40)
        engine = LanguageByLengthEngine(STTConfig())
        # 첫 청크(3.2초)만 pt 로 감지된다
        chunks = [SpeechSegment(0.0, 3.2)] + [
            SpeechSegment(5.0 * i, 5.0 * i + 4.0) for i in range(1, 7)
        ]
        with (
            patch("src.audio.extractor.AudioExtractor", FakeExtractor(wav)),
            patch("src.audio.extractor.get_audio_duration", return_value=40.0),
            patch("src.audio.vad.detect_speech", return_value=chunks),
            patch("src.engines.engine_factory.create_engine", return_value=engine),
        ):
            yield {"engine": engine, "chunks": chunks}

    def test_majority_language_used_for_all_chunks(
        self, config: AppConfig, source: Path, long_env: dict[str, Any]
    ) -> None:
        config.audio.vad.merge_gap_s = 0.1
        with TranscriptionPipeline(config) as p:
            outcome = p.run(source, PipelineOptions(write=False))

        engine: LanguageByLengthEngine = long_env["engine"]
        assert outcome.result.metadata.language == "ja"
        # 표본 5개를 각자 감지한 뒤 나머지 청크는 ja 로 고정해 인식한다
        assert [lang for _, lang in engine.calls[:5]] == [None] * 5
        assert all(lang == "ja" for _, lang in engine.calls[5:])
        # 감지 언어와 같은 표본 4개는 다시 인식하지 않고, pt 로 잡힌 첫 청크만 ja 로 다시 인식한다:
        # 표본 5 + 나머지 2 + 다시 1
        assert len(engine.calls) == len(long_env["chunks"]) + 1
        assert engine.calls[5:].count((pytest.approx(3.2), "ja")) == 1

    def test_configured_language_skips_detection(
        self, config: AppConfig, source: Path, long_env: dict[str, Any]
    ) -> None:
        config.stt.source_lang = "ko"
        config.audio.vad.merge_gap_s = 0.1
        with TranscriptionPipeline(config) as p:
            p.run(source, PipelineOptions(write=False))

        engine: LanguageByLengthEngine = long_env["engine"]
        assert {lang for _, lang in engine.calls} == {"ko"}


class TestScenes:
    """장면 감지 연결 (REQUIREMENT 10.2): 경계 감지 → 경계 자막 간격 보장."""

    @staticmethod
    def _segments() -> list[TranscriptionSegment]:
        # 0~4초 자막 둘(이어짐) → 12초 공백 → 자막 둘
        return [
            TranscriptionSegment(1, "a", 0.0, 2.0, 0.9),
            TranscriptionSegment(2, "b", 2.0, 4.0, 0.9),
            TranscriptionSegment(3, "c", 16.0, 18.0, 0.9),
            TranscriptionSegment(4, "d", 18.0, 20.0, 0.9),
        ]

    def test_video_cut_gets_gap(self, config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.quality import scene_detector

        config.scene.use_video = True
        monkeypatch.setattr(
            scene_detector,
            "detect_scenes_by_video",
            lambda path: [scene_detector.SceneBoundary(time_s=2.0, source="pyscenedetect")],
        )
        segments = self._segments()
        TranscriptionPipeline(config)._apply_scenes(segments, "movie.mp4", PipelineOptions())
        # 컷(2.0초) 앞뒤로 boundary_gap_s(0.5초)가 생긴다
        assert segments[0].end == pytest.approx(1.75)
        assert segments[1].start == pytest.approx(2.25)

    def test_video_detection_skipped_for_audio_and_trimmed_input(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.quality import scene_detector

        calls: list[str] = []
        monkeypatch.setattr(
            scene_detector, "detect_scenes_by_video", lambda path: calls.append(path) or []
        )
        config.scene.use_video = True
        pipeline = TranscriptionPipeline(config)
        pipeline._apply_scenes(self._segments(), "talk.wav", PipelineOptions())
        pipeline._apply_scenes(self._segments(), "movie.mp4", PipelineOptions(start="00:01:00"))
        assert calls == []

    def test_disabled_leaves_segments(
        self, config: AppConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.quality import scene_detector

        config.scene.enabled = False
        monkeypatch.setattr(
            scene_detector, "detect_scenes_by_gap", lambda *a: pytest.fail("감지하지 않아야 한다")
        )
        segments = self._segments()
        TranscriptionPipeline(config)._apply_scenes(segments, "movie.mp4", PipelineOptions())
        assert [(s.start, s.end) for s in segments] == [
            (0.0, 2.0),
            (2.0, 4.0),
            (16.0, 18.0),
            (18.0, 20.0),
        ]

    def test_gap_boundary_runs_in_pipeline(
        self, config: AppConfig, source: Path, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.quality import scene_detector

        seen: list[float] = []
        original = scene_detector.detect_scenes_by_gap

        def spy(segments: list[TranscriptionSegment], gap_s: float, min_segments: int) -> Any:
            seen.append(gap_s)
            return original(segments, gap_s, min_segments)

        monkeypatch.setattr(scene_detector, "detect_scenes_by_gap", spy)
        with TranscriptionPipeline(config) as p:
            p.run(source, PipelineOptions(write=False))
        assert seen == [config.scene.gap_s]


class TestHelpers:
    """단어 보완·구두점·WAV 자르기."""

    def test_ensure_words_spreads_by_length(self) -> None:
        seg = TranscriptionSegment(index=1, text="ab cdef", start=1.0, end=4.0, confidence=0.5)
        ensure_words(seg, "en")
        assert [w.word for w in seg.words] == ["ab", "cdef"]
        assert seg.words[0].start == 1.0 and seg.words[-1].end == 4.0
        assert seg.words[0].end == pytest.approx(2.0)
        # 추정 단어는 신뢰도 미제공(0.0): 노이즈 필터가 지우지 않는다
        assert all(w.confidence == 0.0 for w in seg.words)

    def test_ensure_words_cjk_and_existing(self) -> None:
        seg = TranscriptionSegment(index=1, text="こんにちは", start=0.0, end=1.0)
        ensure_words(seg, "ja")
        assert [w.word for w in seg.words] == ["こんにちは"]
        kept = TranscriptionSegment(
            index=1, text="x", start=0, end=1, words=[WordSegment(word="x", start=0, end=1)]
        )
        ensure_words(kept, "en")
        assert len(kept.words) == 1

    def test_restore_sentence_end_updates_last_word(self) -> None:
        seg = TranscriptionSegment(
            index=1,
            text="hello there",
            start=0,
            end=1,
            words=[
                WordSegment(word="hello", start=0, end=0.4),
                WordSegment(word="there", start=0.5, end=1),
            ],
        )
        done = TranscriptionSegment(
            index=2, text="ok.", start=1, end=2, words=[WordSegment(word="ok.", start=1, end=2)]
        )
        restore_sentence_end([seg, done], "en")
        assert seg.text == "hello there."
        assert seg.words[-1].word == "there."
        assert done.text == "ok." and done.words[-1].word == "ok."

    def test_write_wav_slice(self, wav: Path, tmp_path: Path) -> None:
        piece = write_wav_slice(wav, 1.0, 2.5, tmp_path / "piece.wav")
        assert _wav_seconds(piece) == pytest.approx(1.5)
        clipped = write_wav_slice(wav, 7.0, 99.0, tmp_path / "end.wav")
        assert _wav_seconds(clipped) == pytest.approx(1.0)
