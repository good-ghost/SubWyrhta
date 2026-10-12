"""단계 결과 보관 (다시 시도할 때 끝낸 단계를 건너뛴다).

음성 인식은 청크 체크포인트(``src/checkpoint.py``)로 이어서 한다. 이 모듈은 그 앞뒤의
오래 걸리는 단계 결과를 작업 폴더에 남긴다.

- 전처리한 음성 (추출 → demucs → 음량 정규화 → 노이즈 억제 결과, 16kHz mono WAV)
- Pre-VAD 발화 구간
- 화자 분리 결과 (발화 구간, 화자 수, 화자 임베딩)

단계마다 입력과 그 단계에 영향을 주는 설정으로 만든 키를 함께 저장한다. 키가 다르면 쓰지 않는다
(입력 파일이나 설정이 바뀌면 다시 만든다). 파이프라인이 끝나면 폴더를 지운다.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.checkpoint import compute_config_hash

if TYPE_CHECKING:
    from src.audio.vad import SpeechSegment
    from src.diarization.diarizer import DiarizationResult

log = structlog.get_logger(__name__)

_CACHE_VERSION = 1
_WORK_DIR_SUFFIX = ".work"
_MANIFEST_FILE = "stages.json"
_AUDIO_FILE = "audio.wav"

STAGE_AUDIO = "audio"
STAGE_VAD = "vad"
STAGE_DIARIZE = "diarize"
STAGE_PUNCTUATION = "punctuation"
STAGE_FINALIZE = "finalize"
STAGE_CORRECTION = "correction"

# 목록 파일의 값 모양이 다를 때 (손으로 고쳤거나 저장 도중 끊긴 경우)
_MALFORMED = (KeyError, TypeError, ValueError, AttributeError)


def get_work_dir(source_file: str | Path) -> Path:
    """입력 파일에 대응하는 단계 결과 폴더 (체크포인트 파일과 같은 규칙: 입력 파일 옆)."""
    return Path(str(source_file) + _WORK_DIR_SUFFIX)


def input_fingerprint(source_file: str | Path) -> dict[str, int]:
    """입력 파일이 바뀌었는지 가리는 값 (크기, 수정 시각).

    긴 영상의 내용 해시는 오래 걸려 크기와 수정 시각(ns)을 쓴다.
    """
    stat = Path(source_file).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def stage_key(parts: dict[str, Any]) -> str:
    """단계 키. 입력·설정이 같으면 같은 값이다."""
    return compute_config_hash(parts)


@dataclass(frozen=True)
class CachedAudio:
    """보관한 전처리 음성.

    Attributes:
        path: 16kHz mono WAV.
        duration_s: 추출한 오디오 길이(초). 결과 메타데이터에 쓴다.
    """

    path: Path
    duration_s: float


class StageCache:
    """작업 폴더 하나의 단계 결과.

    저장에 실패하면 경고만 남기고 처리를 계속한다 (다시 시도할 때 그 단계를 다시 할 뿐이다).

    Args:
        work_dir: 단계 결과 폴더. 없으면 저장할 때 만든다.
    """

    def __init__(self, work_dir: str | Path) -> None:
        self.work_dir = Path(work_dir)

    # ── 전처리 음성 ──

    def load_audio(self, key: str) -> CachedAudio | None:
        """키가 같은 전처리 음성이 있으면 돌려준다."""
        entry = self._entry(STAGE_AUDIO, key)
        path = self.work_dir / _AUDIO_FILE
        if entry is None or not path.is_file():
            return None
        try:
            cached = CachedAudio(path=path, duration_s=float(entry["duration_s"]))
        except _MALFORMED as e:
            _warn_malformed(STAGE_AUDIO, e)
            return None
        log.info("전처리 결과 재사용", path=str(path))
        return cached

    def save_audio(self, key: str, audio: Path, duration_s: float) -> None:
        """전처리 음성을 폴더에 복사해 둔다 (원본은 임시 파일이라 처리가 끝나면 지워진다)."""
        target = self.work_dir / _AUDIO_FILE
        partial = target.with_name(target.name + ".part")
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audio, partial)
            os.replace(partial, target)
        except OSError as e:
            log.warning("전처리 결과 저장 실패, 재개 없이 계속", path=str(target), error=str(e))
            # 위에서 실패를 남겼다. 반쯤 쓴 파일 정리는 폴더 자체를 못 만든 경우 등에 실패할 수 있다
            with contextlib.suppress(OSError):
                partial.unlink(missing_ok=True)
            return
        self._put(STAGE_AUDIO, key, {"duration_s": duration_s})

    # ── Pre-VAD 발화 구간 ──

    def load_chunks(self, key: str) -> list[SpeechSegment] | None:
        """키가 같은 발화 구간 목록이 있으면 돌려준다."""
        from src.audio.vad import SpeechSegment

        entry = self._entry(STAGE_VAD, key)
        if entry is None:
            return None
        try:
            chunks = [
                SpeechSegment(
                    start=float(c["start"]),
                    end=float(c["end"]),
                    skip_alignment=bool(c.get("skip_alignment", False)),
                )
                for c in entry["chunks"]
            ]
        except _MALFORMED as e:
            _warn_malformed(STAGE_VAD, e)
            return None
        log.info("Pre-VAD 결과 재사용", chunks=len(chunks))
        return chunks

    def save_chunks(self, key: str, chunks: list[SpeechSegment]) -> None:
        """발화 구간 목록을 남긴다."""
        data = [
            {"start": c.start, "end": c.end, "skip_alignment": c.skip_alignment} for c in chunks
        ]
        self._put(STAGE_VAD, key, {"chunks": data})

    # ── 화자 분리 ──

    def load_diarization(self, key: str) -> DiarizationResult | None:
        """키가 같은 화자 분리 결과가 있으면 돌려준다."""
        from src.diarization.diarizer import DiarizationResult, SpeakerTurn

        entry = self._entry(STAGE_DIARIZE, key)
        if entry is None:
            return None
        try:
            result = DiarizationResult(
                turns=[
                    SpeakerTurn(
                        speaker=str(t["speaker"]), start=float(t["start"]), end=float(t["end"])
                    )
                    for t in entry["turns"]
                ],
                num_speakers=int(entry["num_speakers"]),
                embeddings=entry.get("embeddings"),
            )
        except _MALFORMED as e:
            _warn_malformed(STAGE_DIARIZE, e)
            return None
        log.info("화자 분리 결과 재사용", speakers=result.num_speakers, turns=len(result.turns))
        return result

    def save_diarization(self, key: str, result: DiarizationResult) -> None:
        """화자 분리 결과를 남긴다 (화자 ID 를 바꾸기 전 값)."""
        embeddings = result.embeddings if isinstance(result.embeddings, dict) else None
        data = {
            "turns": [{"speaker": t.speaker, "start": t.start, "end": t.end} for t in result.turns],
            "num_speakers": result.num_speakers,
            "embeddings": embeddings,
        }
        self._put(STAGE_DIARIZE, key, data)

    # ── LLM 부호 넣기 ──

    def load_punctuation(self, key: str) -> dict[str, str] | None:
        """키가 같은 LLM 부호 넣기 결과(원문 → 부호를 넣은 문장)가 있으면 돌려준다."""
        entry = self._entry(STAGE_PUNCTUATION, key)
        if entry is None:
            return None
        try:
            texts = {str(k): str(v) for k, v in entry["texts"].items()}
        except _MALFORMED as e:
            _warn_malformed(STAGE_PUNCTUATION, e)
            return None
        log.info("LLM 부호 넣기 결과 재사용", lines=len(texts))
        return texts

    def save_punctuation(self, key: str, texts: dict[str, str]) -> None:
        """LLM 부호 넣기 결과를 남긴다 (다시 시도할 때 같은 문장을 다시 보내지 않는다)."""
        self._put(STAGE_PUNCTUATION, key, {"texts": texts})

    # ── LLM 오인식 교정 ──

    def load_correction(self, key: str) -> dict[str, list[list[str]]] | None:
        """키가 같은 LLM 오인식 교정 결과(원문 → [[from, to, 이유]])가 있으면 돌려준다."""
        entry = self._entry(STAGE_CORRECTION, key)
        if entry is None:
            return None
        try:
            edits = {
                str(line): [[str(x) for x in edit][:3] for edit in items]
                for line, items in entry["edits"].items()
            }
        except _MALFORMED as e:
            _warn_malformed(STAGE_CORRECTION, e)
            return None
        log.info("LLM 오인식 교정 결과 재사용", lines=len(edits))
        return edits

    def save_correction(self, key: str, edits: dict[str, list[list[str]]]) -> None:
        """LLM 오인식 교정 결과를 남긴다 (다시 시도할 때 같은 문장을 다시 보내지 않는다)."""
        self._put(STAGE_CORRECTION, key, {"edits": edits})

    # ── LLM 마지막 검토 ──

    def load_finalize(self, key: str) -> dict[str, dict[str, Any]] | None:
        """키가 같은 LLM 마지막 검토 응답(묶음 키 → 응답)이 있으면 돌려준다."""
        entry = self._entry(STAGE_FINALIZE, key)
        if entry is None:
            return None
        answers = entry.get("answers")
        if not isinstance(answers, dict):
            _warn_malformed(STAGE_FINALIZE, TypeError("answers"))
            return None
        log.info("LLM 마지막 검토 응답 재사용", windows=len(answers))
        return {str(k): v for k, v in answers.items() if isinstance(v, dict)}

    def save_finalize(self, key: str, answers: dict[str, dict[str, Any]]) -> None:
        """LLM 마지막 검토 응답을 남긴다 (자막이 같은 묶음은 다시 보내지 않는다)."""
        self._put(STAGE_FINALIZE, key, {"answers": answers})

    # ── 정리 ──

    def clear(self) -> None:
        """폴더를 지운다 (처리가 끝났거나 처음부터 다시 할 때)."""
        if not self.work_dir.exists():
            return
        try:
            shutil.rmtree(self.work_dir)
        except OSError as e:
            log.warning("단계 결과 폴더 삭제 실패", path=str(self.work_dir), error=str(e))
            return
        log.info("단계 결과 폴더 삭제", path=str(self.work_dir))

    # ── 목록 파일 ──

    def _entry(self, stage: str, key: str) -> dict[str, Any] | None:
        entry = self._read().get(stage)
        if not isinstance(entry, dict) or entry.get("key") != key:
            return None
        return entry

    def _put(self, stage: str, key: str, data: dict[str, Any]) -> None:
        stages = self._read()
        stages[stage] = {"key": key, **data}
        manifest = self.work_dir / _MANIFEST_FILE
        partial = manifest.with_name(manifest.name + ".part")
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            partial.write_text(
                json.dumps({"version": _CACHE_VERSION, "stages": stages}, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(partial, manifest)
        except OSError as e:
            log.warning("단계 결과 저장 실패, 재개 없이 계속", stage=stage, error=str(e))

    def _read(self) -> dict[str, Any]:
        manifest = self.work_dir / _MANIFEST_FILE
        if not manifest.is_file():
            return {}
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("단계 결과 목록을 읽지 못함, 무시", path=str(manifest), error=str(e))
            return {}
        if not isinstance(data, dict) or data.get("version") != _CACHE_VERSION:
            return {}
        stages = data.get("stages")
        return stages if isinstance(stages, dict) else {}


def _warn_malformed(stage: str, error: Exception) -> None:
    log.warning("단계 결과 형식 오류, 다시 만든다", stage=stage, error=repr(error))
