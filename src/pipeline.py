"""전체 자막 생성 파이프라인.

입력 → AudioExtractor(ffmpeg) → AudioEnhancer(demucs→normalize→denoise) → Pre-VAD(Silero)
→ STTEngine → NoiseFilter → (구두점 복원) → Diarization → Segmentation(화자 분할 → 모드별 분할
→ CPS 최적화 → 최소 표시 시간 보장) → 텍스트 정규화 → QualityChecker → SubtitleWriter 순서로 실행한다.

CLI(transcribe·diarize), Python API, 웹 작업 워커가 모두 이 모듈을 쓴다.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import AudioExtractionError
from src.models.result import Metadata, TranscriptionResult
from src.models.segment import TranscriptionSegment
from src.models.word import WordSegment
from src.postprocess.llm_text import LLMBreaker
from src.segmentation.llm_client import LLM_EXTRA_BODY_ENV
from src.version import __version__

if TYPE_CHECKING:
    from src.audio.extractor import AudioExtractor
    from src.audio.vad import SpeechSegment
    from src.core.config import AppConfig
    from src.diarization.diarizer import DiarizationResult
    from src.engines.base import ChunkTranscription, STTEngine
    from src.quality.checker import QualityReport
    from src.stage_cache import StageCache

log = structlog.get_logger(__name__)

SUBWYRHTA_VERSION = __version__

# 진행 단계 이름 (웹 화면이 문구로 바꾼다)
STAGE_EXTRACT = "extract"
STAGE_ENHANCE = "enhance"
STAGE_VAD = "vad"
STAGE_LOAD_MODEL = "load_model"
STAGE_TRANSCRIBE = "transcribe"
STAGE_DIARIZE = "diarize"
STAGE_SEGMENT = "segment"
STAGE_QUALITY = "quality"
STAGE_WRITE = "write"
STAGES: tuple[str, ...] = (
    STAGE_EXTRACT,
    STAGE_ENHANCE,
    STAGE_VAD,
    STAGE_LOAD_MODEL,
    STAGE_TRANSCRIBE,
    STAGE_DIARIZE,
    STAGE_SEGMENT,
    STAGE_QUALITY,
    STAGE_WRITE,
)

# 진행 알림: (단계, 끝낸 수, 전체 수). 전체를 모르면 0
ProgressCallback = Callable[[str, int, int], None]

# 문장 끝 부호 (구두점 복원 결과에서 마지막 단어에 옮길 문자)
_SENTENCE_END = frozenset(".!?。！？…")
_CJK_LANGUAGES = frozenset({"ja", "zh", "ko"})
_WORD_SPLIT = re.compile(r"\S+")
# 단어 사이가 이만큼 쉬면 발화가 바뀐 것으로 본다 (자막 안 발화 구분). 화자 분리가 잘 안 되는
# 애니메이션에서 Whisper 가 여러 사람의 대사를 한 세그먼트로 묶어도 쉼으로 나눈다
_UTTERANCE_PAUSE_S = 0.5
# 언어 자동 감지 표본 청크 수와 최소 길이(초)
_LANGUAGE_PROBE_CHUNKS = 5
_LANGUAGE_PROBE_MIN_S = 3.0


@dataclass
class PipelineOptions:
    """실행마다 달라지는 입력·출력 옵션 (설정 파일 값이 아닌 것).

    Attributes:
        output_name: 출력 파일명 직접 지정. None이면 규칙에 따른다.
        keep_audio: 추출·전처리한 오디오를 지우지 않는다.
        audio_track: 오디오 트랙 번호 (0-based).
        start: 처리 시작 시각 (HH:MM:SS).
        end: 처리 종료 시각 (HH:MM:SS).
        time_offset_s: 결과 타임스탬프에 더할 값 (챕터 처리에서 챕터 시작 시각).
        formats: 출력 포맷 목록. None이면 ``config.output.format`` 하나.
        resume: 체크포인트가 있으면 남은 Pre-VAD 청크만 추론한다.
        vad_export: VAD 타임라인 JSON을 출력 폴더에 저장한다.
        speaker_map_str: 화자 이름 매핑 문자열 (``SPEAKER_00:John,...``).
        speaker_profile: 화자 프로파일 JSON 경로.
        speaker_embeddings: 이전 파일의 화자 임베딩 JSON. 같은 화자에게 같은 화자 ID 를 붙인다.
        export_embeddings: 화자 임베딩을 출력 폴더에 저장한다.
        checkpoint_path: 체크포인트 파일 경로. None이면 입력 파일 옆.
        keep_stages: 전처리 음성·Pre-VAD·화자 분리 결과를 남겨, ``resume`` 일 때 그 단계를 건너뛴다.
        work_dir: 단계 결과 폴더. None이면 입력 파일 옆 (``<입력>.work``).
        write: False면 파일을 저장하지 않고 결과만 돌려준다 (Python API).
    """

    output_name: str | None = None
    keep_audio: bool = False
    audio_track: int | None = None
    start: str | None = None
    end: str | None = None
    time_offset_s: float = 0.0
    formats: list[str] | None = None
    resume: bool = False
    vad_export: bool = False
    speaker_map_str: str | None = None
    speaker_profile: str | None = None
    speaker_embeddings: str | None = None
    export_embeddings: bool = False
    checkpoint_path: str | None = None
    keep_stages: bool = False
    work_dir: str | None = None
    write: bool = True


@dataclass
class PipelineResult:
    """파이프라인 실행 결과.

    Attributes:
        result: 최종 자막 결과.
        outputs: 저장한 파일 목록 (자막, 품질 리포트, VAD 타임라인, 임베딩).
        report: 품질 검사 리포트 (검사를 껐으면 None).
        num_speakers: 화자 분리를 했으면 화자 수.
    """

    result: TranscriptionResult
    outputs: list[Path] = field(default_factory=list)
    report: QualityReport | None = None
    num_speakers: int | None = None


class TranscriptionPipeline:
    """설정에 따라 전체 자막 생성 단계를 실행한다.

    STT 모델은 ``run``마다 로드하고 추론 직후 언로드한다 (화자 분리 모델과 동시에 메모리에 두지 않는다).
    ``keep_engine=True``면 여러 번 ``run``하는 동안 모델을 유지하고 ``close``에서 언로드한다 (챕터 처리).

    Args:
        config: 전체 설정.
        progress: 단계 진행 알림. None이면 알리지 않는다.
        keep_engine: True면 run 사이에 STT 모델을 유지한다.
    """

    def __init__(
        self,
        config: AppConfig,
        progress: ProgressCallback | None = None,
        keep_engine: bool = False,
    ) -> None:
        self.config = config
        self._progress = progress
        self._keep_engine = keep_engine
        self._engine: STTEngine | None = None
        # LLM 마지막 검토의 검수 리포트 항목 (run 마다 비우고, 쓰기 단계에서 <이름>.review.json 으로 남긴다)
        self._review_issues: list[dict[str, Any]] = []
        # LLM 오인식 교정에서 반영한 교정과 반영하지 않은 제안 (검수 리포트의 corrected, suggested)
        self._review_corrections: list[dict[str, Any]] = []
        self._review_suggestions: list[dict[str, Any]] = []
        # 작업 단위 LLM 차단기 (TO-DO 20): run 마다 새로 만든다. 한 LLM 단계가 재시도까지 실패하면 이후 단계는
        # 호출하지 않는다
        self._llm_breaker = LLMBreaker()

    def __enter__(self) -> TranscriptionPipeline:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """유지하던 STT 모델을 언로드한다."""
        if self._engine is not None:
            self._engine.unload_model()
            self._engine = None

    # ──────────────────────────────────────────
    # 실행
    # ──────────────────────────────────────────

    def run(self, input_path: str | Path, options: PipelineOptions | None = None) -> PipelineResult:
        """입력 파일 하나를 자막으로 만든다.

        Args:
            input_path: 입력 영상/음성 파일 경로.
            options: 실행 옵션. None이면 기본값.

        Returns:
            PipelineResult.
        """
        from src.audio.extractor import AudioExtractor

        opts = self._resolve_audio_track(str(input_path), options or PipelineOptions())
        source = str(input_path)
        log.info("파이프라인 시작", input=source)
        self._check_diarization_access()

        cache = self._stage_cache(source, opts)
        audio_key = self._audio_key(source, opts) if cache is not None else ""
        with AudioExtractor(keep_audio=opts.keep_audio) as extractor:
            cached = cache.load_audio(audio_key) if cache is not None and opts.resume else None
            enhanced: list[Path] = []
            if cached is not None:
                stt_audio, duration = cached.path, cached.duration_s
            else:
                self._notify(STAGE_EXTRACT)
                audio_path = extractor.extract(
                    input_path=source, audio_track=opts.audio_track, start=opts.start, end=opts.end
                )
                duration = _audio_duration(audio_path)
                stt_audio, enhanced = self._enhance(extractor, audio_path, opts.keep_audio)
                if cache is not None:
                    cache.save_audio(audio_key, stt_audio, duration)
            try:
                chunks = self._chunks(stt_audio, cache, audio_key, opts)
                result = self._transcribe(stt_audio, chunks, source, opts)
                outputs: list[Path] = []
                if chunks is not None and opts.vad_export:
                    outputs.append(self._export_vad(chunks, source, opts.output_name))
                result.metadata.source_file = source
                result.metadata.duration_s = duration
                diarization = self._diarize(stt_audio, opts, cache, audio_key)
                # STT 후처리는 전처리 음성이 남아 있는 동안 한다 (오인식 교정 뒤 다시 정렬에 쓴다)
                self._notify(STAGE_SEGMENT)
                language = result.metadata.language
                self._review_issues, self._review_corrections, self._review_suggestions = [], [], []
                self._llm_breaker = LLMBreaker()
                segments = self._postprocess_stt(
                    result.segments, language, cache, opts, source, stt_audio
                )
            finally:
                if not opts.keep_audio:
                    for tmp in enhanced:
                        tmp.unlink(missing_ok=True)

        segments = self._segment(segments, language, diarization, cache, opts)
        if diarization is not None:
            self._apply_speaker_names(segments, opts)
        if opts.time_offset_s:
            _shift(segments, opts.time_offset_s)
        self._apply_scenes(segments, source, opts)
        result.segments = segments
        result.reindex()

        self._notify(STAGE_QUALITY)
        report = self._check_quality(result)

        if not opts.write:
            self._clear_resume_state(source, opts, cache)
            return PipelineResult(
                result=result,
                report=report,
                num_speakers=diarization.num_speakers if diarization is not None else None,
            )

        self._notify(STAGE_WRITE)
        outputs.extend(self._write(result, opts, diarized=diarization is not None))
        if report is not None and self.config.quality.export_report:
            outputs.append(self._save_report(report, result, opts.output_name))
        if (
            self._review_issues
            or self._review_corrections
            or self._review_suggestions
            or self._llm_breaker.is_open
        ):
            outputs.append(self._save_review(result, opts.output_name))
        if diarization is not None and opts.export_embeddings:
            embedding_path = self._export_embeddings(diarization, source, opts.output_name)
            if embedding_path is not None:
                outputs.append(embedding_path)

        self._clear_resume_state(source, opts, cache)
        log.info(
            "파이프라인 완료",
            input=source,
            segments=result.segment_count,
            outputs=[str(p) for p in outputs],
        )
        return PipelineResult(
            result=result,
            outputs=outputs,
            report=report,
            num_speakers=diarization.num_speakers if diarization is not None else None,
        )

    # ──────────────────────────────────────────
    # 오디오 단계
    # ──────────────────────────────────────────

    def _resolve_audio_track(self, source: str, opts: PipelineOptions) -> PipelineOptions:
        """트랙을 지정하지 않았고 언어를 정했으면 그 언어의 오디오 트랙을 고른다."""
        language = self.config.stt.source_lang
        if opts.audio_track is not None or language == "auto":
            return opts

        from src.audio.extractor import find_audio_track_for_language

        track = find_audio_track_for_language(source, language)
        if track is None:
            return opts
        log.info("오디오 트랙 자동 선택", language=language, audio_track=track)
        return dataclasses.replace(opts, audio_track=track)

    def _enhance(
        self, extractor: AudioExtractor, audio_path: Path, keep_audio: bool
    ) -> tuple[Path, list[Path]]:
        """전처리(demucs → 음량 정규화 → 노이즈 억제)를 적용한다.

        Returns:
            (STT에 넣을 16kHz mono WAV, 이 단계가 만든 파일 목록).
        """
        audio_cfg = self.config.audio
        if not (audio_cfg.separate_vocals or audio_cfg.normalize or audio_cfg.denoise.enabled):
            return audio_path, []

        from src.audio.enhancer import enhance_audio
        from src.engines.base import resolve_device

        self._notify(STAGE_ENHANCE)
        enhanced = enhance_audio(
            audio_path,
            normalize=audio_cfg.normalize,
            denoise=audio_cfg.denoise.enabled,
            denoise_model=audio_cfg.denoise.model,
            separate_vocals=audio_cfg.separate_vocals,
            device=resolve_device(self.config.stt.device),
        )
        if enhanced == audio_path:
            return audio_path, []
        created = [enhanced]
        if audio_cfg.separate_vocals and not audio_cfg.normalize:
            # demucs 출력은 원본 샘플레이트·스테레오다. 청크를 자르는 STT 단계는 16kHz mono PCM을 전제한다
            converted = extractor.extract(input_path=enhanced)
            if not keep_audio:
                enhanced.unlink(missing_ok=True)
                created = []
            return converted, created
        return enhanced, created

    def _detect_chunks(self, audio_path: Path) -> list[SpeechSegment] | None:
        """Pre-VAD 발화 구간을 만든다. Pre-VAD를 쓰지 않으면 None (전체 추론)."""
        vad_cfg = self.config.audio.vad
        if not (vad_cfg.enabled and vad_cfg.pre_vad):
            return None

        from src.audio.vad import detect_speech, merge_speech_segments

        self._notify(STAGE_VAD)
        raw = detect_speech(audio_path, vad_cfg)
        chunks = merge_speech_segments(raw, vad_cfg)
        log.info("Pre-VAD 청크", speech_segments=len(raw), chunks=len(chunks))
        return chunks

    def _chunks(
        self, audio_path: Path, cache: StageCache | None, audio_key: str, opts: PipelineOptions
    ) -> list[SpeechSegment] | None:
        """Pre-VAD 발화 구간. 보관한 결과가 있으면 다시 감지하지 않는다."""
        vad_cfg = self.config.audio.vad
        if cache is None or not (vad_cfg.enabled and vad_cfg.pre_vad):
            return self._detect_chunks(audio_path)

        from src.stage_cache import stage_key

        key = stage_key({"audio": audio_key, "vad": dataclasses.asdict(vad_cfg)})
        chunks = cache.load_chunks(key) if opts.resume else None
        if chunks is None:
            chunks = self._detect_chunks(audio_path)
            if chunks is not None:
                cache.save_chunks(key, chunks)
        return chunks

    def _export_vad(
        self, chunks: list[SpeechSegment], source: str, output_name: str | None
    ) -> Path:
        from src.audio.vad import export_vad_timeline

        path = Path(self.config.output.dir) / f"{output_name or Path(source).stem}.vad.json"
        return export_vad_timeline(chunks, path)

    # ──────────────────────────────────────────
    # STT 단계
    # ──────────────────────────────────────────

    def _engine_loaded(self) -> STTEngine:
        if self._engine is None:
            from src.engines.engine_factory import create_engine

            self._notify(STAGE_LOAD_MODEL)
            engine = create_engine(self.config.stt)
            engine.load_model()
            self._engine = engine
        return self._engine

    def _transcribe(
        self,
        audio_path: Path,
        chunks: list[SpeechSegment] | None,
        source: str,
        opts: PipelineOptions,
    ) -> TranscriptionResult:
        """STT 추론. Pre-VAD 청크가 있으면 청크별 추론 + 체크포인트, 없으면 전체 추론."""
        language = self.config.stt.source_lang if self.config.stt.source_lang != "auto" else None
        try:
            if chunks is None:
                engine = self._engine_loaded()
                self._notify(STAGE_TRANSCRIBE, 0, 1)
                result = engine.transcribe(audio_path, language=language)
                self._notify(STAGE_TRANSCRIBE, 1, 1)
                return result
            return self._transcribe_chunks(audio_path, chunks, language, source, opts)
        finally:
            if not self._keep_engine:
                self.close()

    def _transcribe_chunks(
        self,
        audio_path: Path,
        chunks: list[SpeechSegment],
        language: str | None,
        source: str,
        opts: PipelineOptions,
    ) -> TranscriptionResult:
        from src.checkpoint import load_checkpoint, save_checkpoint

        total = len(chunks)
        config_hash = checkpoint_config_hash(self.config, opts, chunks)
        checkpoint_path = opts.checkpoint_path
        done: dict[int, ChunkRecord] = {}
        if opts.resume:
            saved = load_checkpoint(source, config_hash, checkpoint_path)
            if saved is not None and saved.get("total_chunks") == total:
                done = _records_from_checkpoint(saved.get("partial_segments", []))
                if language is None:
                    language = next((r.language for r in done.values() if r.segments), None)

        pending = [(i, chunk) for i, chunk in enumerate(chunks) if i not in done]
        if language is None and pending:
            language, probed = self._detect_language(audio_path, pending)
            for item in probed:
                # 감지한 언어와 같은 청크(또는 말이 없는 청크)는 다시 인식하지 않는다
                if item.language == language or not item.segments:
                    done[item.index] = ChunkRecord(item.language, item.model, item.segments)
            if probed:
                self._save_checkpoint(
                    save_checkpoint, source, config_hash, total, done, checkpoint_path
                )
            pending = [(i, chunk) for i, chunk in pending if i not in done]
        # 체크포인트에 모든 청크가 있으면 모델을 불러오지 않는다 (큰 모델은 로드에 시간·메모리가 든다)
        items = (
            self._engine_loaded().transcribe_chunks(audio_path, pending, language=language)
            if pending
            else iter(())
        )
        self._notify(STAGE_TRANSCRIBE, len(done), total)
        for item in items:
            done[item.index] = ChunkRecord(item.language, item.model, item.segments)
            self._save_checkpoint(
                save_checkpoint, source, config_hash, total, done, checkpoint_path
            )
            self._notify(STAGE_TRANSCRIBE, len(done), total)

        segments: list[TranscriptionSegment] = []
        for index in sorted(done):
            segments.extend(done[index].segments)
        records = [done[i] for i in sorted(done)]
        detected = next((r.language for r in records if r.segments and r.language), language or "")
        model = next((r.model for r in records if r.model), self.config.stt.model)
        for i, seg in enumerate(segments, start=1):
            seg.index = i
        return TranscriptionResult(
            metadata=Metadata(model=model, language=detected, subwyrhta_version=SUBWYRHTA_VERSION),
            segments=segments,
        )

    def _detect_language(
        self, audio_path: Path, pending: list[tuple[int, SpeechSegment]]
    ) -> tuple[str | None, list[ChunkTranscription]]:
        """음성 전체에 고루 퍼진 청크 몇 개를 따로 인식해 다수결로 언어를 정한다.

        엔진은 처음 텍스트가 나온 청크의 언어를 이후 청크에 고정한다. 그 청크가 신음·잡음이면
        엉뚱한 언어로 전체를 인식한다 (일본어 영상 10분 구간을 포르투갈어로 인식한 사례).

        Returns:
            (감지한 언어. 정하지 못하면 None, 표본 청크의 인식 결과).
        """
        probe = _language_probe_chunks(pending)
        if len(probe) <= 1:
            return None, []
        engine = self._engine_loaded()
        results: list[ChunkTranscription] = []
        for item in probe:
            # 청크마다 따로 불러 각자 언어를 감지하게 한다 (한 번에 주면 첫 청크 언어로 고정된다)
            results.extend(engine.transcribe_chunks(audio_path, [item], language=None))
        votes = Counter(r.language for r in results if r.segments and r.language)
        if not votes:
            return None, []
        language = votes.most_common(1)[0][0]
        log.info("언어 감지 (다수결)", language=language, votes=dict(votes))
        return language, results

    def _save_checkpoint(
        self,
        save: Callable[..., Path],
        source: str,
        config_hash: str,
        total: int,
        done: dict[int, ChunkRecord],
        checkpoint_path: str | None,
    ) -> None:
        """청크를 끝낼 때마다 저장한다. 저장하지 못해도 추론은 계속한다 (재개만 못 한다)."""
        partial = [
            {
                "chunk": index,
                "language": record.language,
                "model": record.model,
                "segments": [seg.to_dict() for seg in record.segments],
            }
            for index, record in sorted(done.items())
        ]
        try:
            save(source, config_hash, total, sorted(done), partial, checkpoint_path)
        except OSError as e:
            log.warning("체크포인트 저장 실패, 재개 없이 계속", source=source, error=str(e))

    def _clear_resume_state(
        self, source: str, opts: PipelineOptions, cache: StageCache | None
    ) -> None:
        """끝낸 작업의 체크포인트와 단계 결과를 지운다."""
        from src.checkpoint import delete_checkpoint

        delete_checkpoint(source, opts.checkpoint_path)
        if cache is not None:
            cache.clear()

    # ──────────────────────────────────────────
    # 단계 결과 보관
    # ──────────────────────────────────────────

    def _stage_cache(self, source: str, opts: PipelineOptions) -> StageCache | None:
        if not opts.keep_stages:
            return None

        from src.stage_cache import StageCache, get_work_dir

        return StageCache(opts.work_dir or get_work_dir(source))

    def _audio_key(self, source: str, opts: PipelineOptions) -> str:
        """전처리 음성 키: 입력 파일, 전처리 설정, 처리 범위 (VAD 설정은 넣지 않는다)."""
        from src.stage_cache import input_fingerprint, stage_key

        audio_cfg = self.config.audio
        return stage_key(
            {
                "input": input_fingerprint(source),
                "normalize": audio_cfg.normalize,
                "separate_vocals": audio_cfg.separate_vocals,
                "denoise": dataclasses.asdict(audio_cfg.denoise),
                "range": [opts.start, opts.end, opts.audio_track],
            }
        )

    # ──────────────────────────────────────────
    # 후처리·화자·분할 단계
    # ──────────────────────────────────────────

    def _postprocess_stt(
        self,
        segments: list[TranscriptionSegment],
        language: str,
        cache: StageCache | None = None,
        opts: PipelineOptions | None = None,
        source: str = "",
        audio_path: Path | None = None,
    ) -> list[TranscriptionSegment]:
        """STT 세그먼트 단위 후처리.

        단어 보완 → 일본어 문절 묶기 → 늘어진 단어 시간 자르기 → 몰린 단어 시각 다시 나누기 → 노이즈 필터
        → (LLM 오인식 교정 → 다시 정렬) → (LLM 부호 넣기) → 단어 가운데 부호 뒤 나누기 → (구두점 복원).
        """
        from src.postprocess.noise_filter import filter_noise
        from src.postprocess.word_units import split_at_inner_punctuation
        from src.segmentation.speed_optimizer import get_cps_for_language

        max_cps = get_cps_for_language(language, self.config.segmentation)
        clamped = spread = 0
        for seg in segments:
            ensure_words(seg, language)
            c, r = _prepare_words(seg, language, max_cps)
            clamped += c
            spread += r
        if clamped:
            log.info("늘어진 단어 시간 자름", words=clamped)
        if spread:
            log.info("몰린 단어 시각 다시 나눔", runs=spread)
        filtered = filter_noise(segments, self.config.noise_filter)
        if self.config.postprocess.llm_correction:
            self._llm_correct(filtered, language, cache, opts, source, audio_path, max_cps)
        if self.config.postprocess.llm_punctuation:
            self._llm_punctuate(filtered, language, cache, opts)
        for seg in filtered:
            seg.words = split_at_inner_punctuation(seg.words)
        if self.config.postprocess.restore_punctuation:
            filtered = restore_sentence_end(
                filtered, language, use_model=self.config.postprocess.punctuation_model
            )
        return filtered

    def _llm_correct(
        self,
        segments: list[TranscriptionSegment],
        language: str,
        cache: StageCache | None,
        opts: PipelineOptions | None,
        source: str,
        audio_path: Path | None,
        max_cps: float,
    ) -> None:
        """LLM 오인식 교정 뒤 교정한 문장을 다시 정렬한다 (REQUIREMENT 6.2.3, WI-9.018).

        용어 목록은 환경 변수 ``SUBWYRHTA_GLOSSARY`` 에서 읽는다. 용어 목록이 없으면 반영하지 않고 제안을
        검수 리포트의 ``suggested`` 로 남긴다 (수동 교정, TO-DO 19). LLM 제안은 단계 캐시에 남긴다
        (키: 프로바이더·모델·추론 끄기·언어·프롬프트·용어 목록).
        """
        from src.engines.base import resolve_device
        from src.postprocess.glossary import load_glossary
        from src.postprocess.llm_correction import CORRECTION_PROMPT, correct_transcript
        from src.postprocess.realign import realign_segments
        from src.stage_cache import stage_key

        llm = self.config.llm
        glossary = load_glossary(source or None)
        key = stage_key(
            {
                "provider": llm.provider,
                "model": llm.model,
                "disable_thinking": llm.disable_thinking,
                "reasoning_effort": llm.reasoning_effort,
                "extra_body": os.environ.get(LLM_EXTRA_BODY_ENV, ""),
                "language": language,
                "prompt": CORRECTION_PROMPT,
                "glossary": glossary,
                "note_language": self.config.postprocess.review_language,
            }
        )
        resume = cache is not None and opts is not None and opts.resume
        known = cache.load_correction(key) if cache is not None and resume else None
        result = correct_transcript(
            segments,
            language,
            llm,
            glossary,
            known,
            breaker=self._llm_breaker,
            note_language=self.config.postprocess.review_language,
        )
        if cache is not None:
            cache.save_correction(key, result.edits)
        self._review_corrections = result.applied
        self._review_suggestions = result.suggested
        if result.corrected and audio_path is not None:
            realign_segments(
                result.corrected, audio_path, language, resolve_device(self.config.stt.device)
            )
            for seg in result.corrected:
                _prepare_words(seg, language, max_cps)

    def _llm_punctuate(
        self,
        segments: list[TranscriptionSegment],
        language: str,
        cache: StageCache | None,
        opts: PipelineOptions | None,
    ) -> None:
        """세그멘테이션 전에 LLM 으로 문장 부호를 넣는다 (REQUIREMENT 6.2.2).

        결과(원문 → 부호를 넣은 문장)는 단계 캐시에 남긴다. 다시 시도하거나 세그멘테이션만 다시 돌릴 때
        같은 문장을 다시 보내지 않는다. 키에는 프로바이더·모델·프롬프트를 넣는다 (바뀌면 다시 보낸다).
        """
        from src.postprocess.llm_punctuation import PUNCTUATION_PROMPT, punctuate_transcript
        from src.stage_cache import stage_key

        llm = self.config.llm
        key = stage_key(
            {
                "provider": llm.provider,
                "model": llm.model,
                "disable_thinking": llm.disable_thinking,
                "reasoning_effort": llm.reasoning_effort,
                "extra_body": os.environ.get(LLM_EXTRA_BODY_ENV, ""),
                "language": language,
                "prompt": PUNCTUATION_PROMPT,
            }
        )
        resume = cache is not None and opts is not None and opts.resume
        known = cache.load_punctuation(key) if cache is not None and resume else None
        result = punctuate_transcript(segments, language, llm, known, breaker=self._llm_breaker)
        if cache is not None:
            cache.save_punctuation(key, result.texts)

    def _check_diarization_access(self) -> None:
        """화자 분리를 켰으면 모델 접근을 음성 인식 전에 확인한다 (긴 인식 뒤 토큰 실패 방지)."""
        if not self.config.diarization.enabled:
            return

        from src.diarization.diarizer import check_diarization_access

        check_diarization_access(self.config.diarization)

    def _diarize(
        self,
        audio_path: Path,
        opts: PipelineOptions,
        cache: StageCache | None = None,
        audio_key: str = "",
    ) -> DiarizationResult | None:
        if not self.config.diarization.enabled:
            return None

        from src.diarization.diarizer import Diarizer
        from src.stage_cache import stage_key

        self._notify(STAGE_DIARIZE)
        key = stage_key({"audio": audio_key, "num_speakers": self.config.diarization.num_speakers})
        result = cache.load_diarization(key) if cache is not None and opts.resume else None
        if result is None:
            with Diarizer(self.config.diarization) as diarizer:
                result = diarizer.diarize(audio_path)
            if cache is not None:
                cache.save_diarization(key, result)
        if opts.speaker_embeddings is not None:
            _reuse_speaker_ids(result, opts.speaker_embeddings)
        return result

    def _segment(
        self,
        segments: list[TranscriptionSegment],
        language: str,
        diarization: DiarizationResult | None,
        cache: StageCache | None = None,
        opts: PipelineOptions | None = None,
    ) -> list[TranscriptionSegment]:
        """화자 분할 → 모드별 분할 → CPS 최적화 → 최소 표시 시간 보장 → 읽기 시간 확보 → 발화 구분 → 텍스트 정규화
        → 앞에 붙은 문장 부호 옮기기 → 줄바꿈 → (선택) LLM 마지막 검토."""
        from src.diarization.diarizer import assign_speakers
        from src.postprocess.normalizer import normalize_segments
        from src.postprocess.subtitle_text import (
            mark_utterance_breaks,
            move_leading_punctuation,
            wrap_lines,
        )
        from src.segmentation.min_display_guard import enforce_min_duration
        from src.segmentation.segmentation_factory import create_segmentation
        from src.segmentation.speaker_splitter import split_by_speaker
        from src.segmentation.speed_optimizer import (
            extend_for_reading,
            get_chars_per_line,
            optimize_cps,
        )

        seg_cfg = self.config.segmentation
        lang = language or "en"
        utterance_ends = _utterance_ends(segments)
        strategy = create_segmentation(seg_cfg, self.config.llm)

        if diarization is not None:
            assign_speakers(segments, diarization)
            groups = split_by_speaker(segments, diarization, seg_cfg)
            result: list[TranscriptionSegment] = []
            for group in groups:
                if not group.words:
                    result.append(group)
                    continue
                for piece in strategy.segment(group.words, lang):
                    piece.speaker = group.speaker
                    result.append(piece)
        else:
            words = [word for seg in segments for word in seg.words]
            result = strategy.segment(words, lang) if words else list(segments)

        result = optimize_cps(result, lang, seg_cfg)
        chars_per_line = get_chars_per_line(lang, seg_cfg)
        result = enforce_min_duration(
            result, seg_cfg, chars_per_line * seg_cfg.max_lines, chars_per_line
        )
        extend_for_reading(result, lang, seg_cfg)
        mark_utterance_breaks(result, utterance_ends)
        result = move_leading_punctuation(normalize_segments(result))
        wrap_lines(result, lang, chars_per_line, seg_cfg.max_lines, seg_cfg.overlap_text_separator)
        if self.config.postprocess.llm_review:
            self._llm_finalize(result, lang, chars_per_line, cache, opts)
        return result

    def _llm_finalize(
        self,
        segments: list[TranscriptionSegment],
        language: str,
        chars_per_line: int,
        cache: StageCache | None,
        opts: PipelineOptions | None,
    ) -> None:
        """LLM 마지막 검토: 경계 옮기기·나누기·줄바꿈을 반영하고 검수 리포트 항목을 모은다 (REQUIREMENT 6.2.1).

        경계를 옮기거나 나눈 자막은 읽기 시간 확보를 다시 한다. 응답은 단계 캐시에 남긴다 (묶음 내용이 같으면
        다시 보내지 않는다).
        """
        from src.postprocess.llm_finalize import finalize_subtitles
        from src.segmentation.speed_optimizer import extend_for_reading
        from src.stage_cache import stage_key

        llm = self.config.llm
        key = stage_key(
            {
                "provider": llm.provider,
                "model": llm.model,
                "disable_thinking": llm.disable_thinking,
                "reasoning_effort": llm.reasoning_effort,
                "extra_body": os.environ.get(LLM_EXTRA_BODY_ENV, ""),
            }
        )
        resume = cache is not None and opts is not None and opts.resume
        known = cache.load_finalize(key) if cache is not None and resume else None
        result = finalize_subtitles(
            segments,
            language,
            chars_per_line,
            self.config.segmentation,
            llm,
            known,
            breaker=self._llm_breaker,
            note_language=self.config.postprocess.review_language,
        )
        if result.boundaries or result.splits:
            extend_for_reading(segments, language, self.config.segmentation)
        self._review_issues = result.issues
        if cache is not None:
            cache.save_finalize(key, result.records)

    def _apply_scenes(
        self, segments: list[TranscriptionSegment], source: str, opts: PipelineOptions
    ) -> None:
        """장면 경계를 찾아 경계 앞뒤 자막 사이에 간격을 보장한다 (in-place, REQUIREMENT 10.2).

        챕터 처리의 시각 보정 뒤에 돌려, 영상 컷(원본 기준 시각)과 자막 시각이 맞는다.
        ``--start`` 로 앞을 잘랐으면(챕터 처리처럼 시각을 원본 기준으로 보정하지 않으면) 영상 컷
        시각과 어긋나 영상 기반 감지는 하지 않는다.
        """
        scene_cfg = self.config.scene
        if not scene_cfg.enabled or not segments:
            return

        from src.audio.extractor import VIDEO_EXTENSIONS
        from src.quality.scene_detector import (
            detect_scenes_by_gap,
            detect_scenes_by_video,
            ensure_scene_boundary_gaps,
        )

        boundaries = detect_scenes_by_gap(segments, scene_cfg.gap_s, scene_cfg.min_segments)
        if scene_cfg.use_video:
            if opts.start and not opts.time_offset_s:
                log.warning("--start 로 앞을 잘라 영상 기반 장면 감지를 건너뜀", start=opts.start)
            elif Path(source).suffix.lower() in VIDEO_EXTENSIONS:
                boundaries.extend(detect_scenes_by_video(source))
        if boundaries:
            boundaries.sort(key=lambda b: b.time_s)
            ensure_scene_boundary_gaps(segments, boundaries, scene_cfg.boundary_gap_s)

    def _apply_speaker_names(
        self, segments: list[TranscriptionSegment], opts: PipelineOptions
    ) -> None:
        from src.diarization.speaker_mapper import apply_speaker_names, build_speaker_map

        speaker_map = build_speaker_map(
            config_map=self.config.diarization.speaker_map,
            cli_map_string=opts.speaker_map_str,
            profile_path=opts.speaker_profile,
        )
        apply_speaker_names(segments, speaker_map)

    def _check_quality(self, result: TranscriptionResult) -> QualityReport | None:
        if not self.config.quality.enabled:
            return None

        from src.quality.checker import check_quality

        return check_quality(result, self.config.quality, self.config.segmentation)

    # ──────────────────────────────────────────
    # 출력 단계
    # ──────────────────────────────────────────

    def _write(
        self, result: TranscriptionResult, opts: PipelineOptions, diarized: bool
    ) -> list[Path]:
        from src.writers.writer_factory import create_writer

        formats = list(dict.fromkeys(opts.formats or [self.config.output.format]))
        # 단어 타임스탬프 내보내기: JSON 출력이 없으면 JSON(단어 포함)을 함께 저장한다
        if self.config.output.export_words and "json" not in formats:
            formats.append("json")
        paths: list[Path] = []
        for fmt in formats:
            writer = create_writer(fmt)
            paths.append(
                writer.write_auto(
                    result=result,
                    output_dir=self.config.output.dir,
                    output_name=opts.output_name,
                    diarized=diarized,
                )
            )
        return paths

    def _save_report(
        self, report: QualityReport, result: TranscriptionResult, output_name: str | None
    ) -> Path:
        from src.quality.reporter import report_to_dict

        stem = output_name or Path(result.metadata.source_file).stem
        path = Path(self.config.output.dir) / f"{stem}.quality.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = report_to_dict(report, source_file=result.metadata.source_file)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("품질 리포트 저장", path=str(path))
        return path

    def _save_review(self, result: TranscriptionResult, output_name: str | None) -> Path:
        """LLM 검수 리포트를 남긴다 (자막은 바꾸지 않은, 사람이 확인할 항목)."""
        stem = output_name or Path(result.metadata.source_file).stem
        path = Path(self.config.output.dir) / f"{stem}.review.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "source_file": result.metadata.source_file,
            "provider": self.config.llm.provider,
            "model": self.config.llm.model,
            "corrected": self._review_corrections,
            "suggested": self._review_suggestions,
            # LLM 호출이 실패해 이후 LLM 단계를 건너뛰었으면 그 원인 (빈 값이면 정상)
            "llm_error": self._llm_breaker.reason,
            "issues": self._review_issues,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(
            "LLM 검수 리포트 저장",
            path=str(path),
            corrected=len(self._review_corrections),
            suggested=len(self._review_suggestions),
            issues=len(self._review_issues),
        )
        return path

    def _export_embeddings(
        self, diarization: DiarizationResult, source: str, output_name: str | None
    ) -> Path | None:
        if diarization.embeddings is None:
            log.warning("화자 임베딩이 없어 내보내지 않음", source=source)
            return None

        from src.diarization.embedding_store import export_embeddings

        path = Path(self.config.output.dir) / f"{output_name or Path(source).stem}.embeddings.json"
        return export_embeddings(
            diarization.embeddings, diarization.speakers, path, source_file=source
        )

    def _notify(self, stage: str, done: int = 0, total: int = 0) -> None:
        if self._progress is not None:
            self._progress(stage, done, total)


# ──────────────────────────────────────────────
# 보조 함수
# ──────────────────────────────────────────────


@dataclass
class ChunkRecord:
    """끝낸 Pre-VAD 청크 (체크포인트 저장 단위)."""

    language: str
    model: str
    segments: list[TranscriptionSegment]


def _utterance_ends(segments: list[TranscriptionSegment]) -> set[int]:
    """발화가 끝나는 단어의 ``id``: STT 세그먼트의 마지막 단어, 그리고 다음 단어까지 오래 쉬는 단어.

    한 자막에 발화가 둘 이상 합쳐지면 이 자리에 구분을 넣는다 (``mark_utterance_breaks``).
    """
    ends: set[int] = set()
    for seg in segments:
        if not seg.words:
            continue
        ends.add(id(seg.words[-1]))
        for word, following in zip(seg.words, seg.words[1:], strict=False):
            if following.start - word.end >= _UTTERANCE_PAUSE_S:
                ends.add(id(word))
    return ends


def checkpoint_config_hash(
    config: AppConfig, opts: PipelineOptions, chunks: list[SpeechSegment]
) -> str:
    """음성 인식 체크포인트 키: STT·오디오 설정, 처리 범위, 청크 경계.

    청크 경계가 바뀌면(전처리 결과가 달라지면) 같은 청크 번호가 다른 구간을 가리킨다.
    개수만 비교하면 다른 구간의 인식 결과를 이어 붙일 수 있어 경계를 키에 넣는다.
    """
    from src.checkpoint import compute_config_hash

    return compute_config_hash(
        {
            "stt": dataclasses.asdict(config.stt),
            "audio": dataclasses.asdict(config.audio),
            "range": [opts.start, opts.end, opts.audio_track],
            "chunks": [[round(c.start, 3), round(c.end, 3)] for c in chunks],
        }
    )


def _language_probe_chunks(
    pending: list[tuple[int, SpeechSegment]],
) -> list[tuple[int, SpeechSegment]]:
    """언어 감지 표본: 3초 이상인 청크 중 시간 순서로 고르게 최대 5개 (짧은 청크는 감탄사·잡음이 많다)."""
    candidates = [
        item for item in pending if item[1].end - item[1].start >= _LANGUAGE_PROBE_MIN_S
    ] or pending
    count = min(_LANGUAGE_PROBE_CHUNKS, len(candidates))
    return [candidates[(2 * k + 1) * len(candidates) // (2 * count)] for k in range(count)]


def _records_from_checkpoint(partial: list[dict[str, Any]]) -> dict[int, ChunkRecord]:
    """체크포인트의 partial_segments를 청크 기록으로 되돌린다. 형식이 다른 항목은 버린다."""
    records: dict[int, ChunkRecord] = {}
    for item in partial:
        try:
            index = int(item["chunk"])
            segments = [TranscriptionSegment.from_dict(s) for s in item.get("segments", [])]
        except (KeyError, TypeError, ValueError) as e:
            log.warning("체크포인트 항목 무시", error=str(e))
            continue
        records[index] = ChunkRecord(
            language=str(item.get("language", "")),
            model=str(item.get("model", "")),
            segments=segments,
        )
    return records


def _audio_duration(audio_path: Path) -> float:
    from src.audio.extractor import get_audio_duration

    try:
        return get_audio_duration(audio_path)
    except AudioExtractionError as e:
        log.warning("오디오 길이 확인 실패", audio=str(audio_path), error=str(e))
        return 0.0


def _shift(segments: list[TranscriptionSegment], offset_s: float) -> None:
    for seg in segments:
        seg.start += offset_s
        seg.end += offset_s
        for word in seg.words:
            word.start += offset_s
            word.end += offset_s


def _prepare_words(seg: TranscriptionSegment, language: str, max_cps: float) -> tuple[int, int]:
    """STT 단어를 세그멘테이션용으로 다듬는다: 문절 묶기, 늘어진 시간 자르기, 몰린 시각 다시 나누기.

    처음 후처리와 오인식 교정 뒤 다시 정렬한 세그먼트가 같이 쓴다.

    Returns:
        (자른 단어 수, 다시 나눈 구간 수).
    """
    from src.postprocess.word_units import (
        clamp_word_durations,
        group_words_into_phrases,
        spread_collapsed_words,
    )

    # 원문 공백(일본어 속 영어 단어)을 단어 경계로 지킨다. 교정한 세그먼트도 텍스트를 단어로 다시 만들어 둔다
    seg.words = group_words_into_phrases(seg.words, language, seg.text)
    clamped = clamp_word_durations(seg.words)
    spread = spread_collapsed_words(seg.words, max_cps)
    if seg.words:
        seg.start, seg.end = seg.words[0].start, max(w.end for w in seg.words)
    return clamped, spread


def ensure_words(segment: TranscriptionSegment, language: str) -> None:
    """단어 타임스탬프가 없는 세그먼트에 글자 수 비례로 추정한 단어를 채운다 (in-place).

    forced alignment를 건너뛴 짧은 청크나 단어 타임스탬프를 주지 않는 엔진의 세그먼트도
    단어 기반 세그멘테이션을 거치게 한다. 공백이 없는 CJK 텍스트는 세그먼트 전체를 한 단어로 둔다.
    추정한 단어의 신뢰도는 0.0(미제공)으로 둔다: 노이즈 필터가 저신뢰도 단어로 지우지 않게 한다.

    Args:
        segment: 대상 세그먼트.
        language: 언어 코드.
    """
    if segment.words or not segment.text.strip():
        return
    tokens = _WORD_SPLIT.findall(segment.text)
    if language in _CJK_LANGUAGES and len(tokens) <= 1:
        tokens = [segment.text.strip()]
    total_chars = sum(len(t) for t in tokens) or 1
    span = max(segment.end - segment.start, 0.0)
    cursor = segment.start
    words: list[WordSegment] = []
    for token in tokens:
        length = span * len(token) / total_chars
        words.append(
            WordSegment(
                word=token,
                start=cursor,
                end=cursor + length,
                confidence=0.0,
            )
        )
        cursor += length
    if words:
        words[-1].end = segment.end
    segment.words = words


def restore_sentence_end(
    segments: list[TranscriptionSegment],
    language: str,
    use_model: bool = False,
) -> list[TranscriptionSegment]:
    """STT 세그먼트의 문장 끝 부호를 복원하고 마지막 단어에도 반영한다.

    세그멘테이션은 단어로 텍스트를 다시 만들므로, 복원한 문장 끝 부호를 마지막 단어에 붙여야
    자막에 남는다. 쉼표처럼 문장 안에 넣는 부호는 단어 경계와 맞출 수 없어 반영하지 않는다.

    Args:
        segments: STT 세그먼트 목록.
        language: 언어 코드.
        use_model: deepmultilingualpunctuation 모델을 먼저 시도한다.

    Returns:
        구두점이 반영된 세그먼트 목록.
    """
    from src.postprocess.punctuation import restore_punctuation

    restored = restore_punctuation(segments, language=language, use_model=use_model)
    for original, fixed in zip(segments, restored, strict=True):
        mark = fixed.text.rstrip()[-1:] if fixed.text.strip() else ""
        if mark not in _SENTENCE_END or original.text.rstrip()[-1:] in _SENTENCE_END:
            continue
        original.text = original.text.rstrip() + mark
        if original.words and original.words[-1].word.rstrip()[-1:] not in _SENTENCE_END:
            original.words[-1].word = original.words[-1].word.rstrip() + mark
    return segments


def _reuse_speaker_ids(diarization: DiarizationResult, embeddings_path: str) -> None:
    """이전 파일의 화자 임베딩과 대조해 같은 화자에게 같은 화자 ID 를 붙인다 (in-place).

    Raises:
        DiarizationError: 임베딩 파일을 읽을 수 없을 때.
    """
    from src.core.exceptions import DiarizationError
    from src.diarization.embedding_store import (
        known_embeddings,
        load_embeddings,
        match_known_speakers,
    )

    try:
        known = known_embeddings(load_embeddings(embeddings_path))
    except (OSError, ValueError) as e:
        raise DiarizationError(
            f"화자 임베딩 파일을 읽을 수 없습니다: {embeddings_path} ({e})"
        ) from e
    current = diarization.embeddings if isinstance(diarization.embeddings, dict) else {}
    if not known or not current:
        log.warning(
            "화자 임베딩이 없어 이전 화자와 대조하지 않음",
            known=len(known),
            current=len(current),
            path=embeddings_path,
        )
        return
    diarization.relabel(match_known_speakers(current, known))
