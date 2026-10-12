"""pyannote.audio 기반 화자 분리.

HuggingFace 토큰 기반 인증, 화자 수 자동 추정/수동 지정,
화자-세그먼트 매핑을 담당한다.
"""

from __future__ import annotations

import contextlib
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import DiarizationError

if TYPE_CHECKING:
    from src.core.config import DiarizationConfig
    from src.models.segment import TranscriptionSegment

log = structlog.get_logger(__name__)

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
# 사용 조건에 동의해야 받을 수 있는 모델 (안내용). pyannote.audio 4.x 는 3.1 파이프라인을 불러올 때
# segmentation-3.0 과 speaker-diarization-community-1(PLDA)도 받는다 (2026-10-09 확인)
GATED_MODELS: tuple[str, ...] = (
    DIARIZATION_MODEL,
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-community-1",
)
_HF_TOKEN_HELP = (
    "HF_TOKEN(.env)이 유효한 Hugging Face 토큰(hf_…)인지, 다음 모델 페이지에서 사용 조건에 "
    f"동의했는지 확인하세요: {', '.join(GATED_MODELS)}"
)


def check_diarization_access(config: DiarizationConfig) -> None:
    """화자 분리 파이프라인을 음성 인식 전에 한 번 불러와 본다.

    화자 분리는 음성 인식 뒤에 돌아, 토큰·사용 조건 문제를 그때 알면 긴 인식 시간을 버린다
    (작업 #1, 2026-10-09). 모델 이름 목록으로 접근을 확인하면 pyannote 버전마다 추가로 받는 모델
    (4.x 의 speaker-diarization-community-1)을 놓치므로, 실제로 불러온 뒤 바로 내린다.
    받은 모델은 캐시에 남아 화자 분리 단계의 로드도 빨라진다.

    Args:
        config: 화자 분리 설정 (``hf_token`` 이 비면 ``HF_TOKEN`` 환경 변수).

    Raises:
        DiarizationError: 토큰이 없거나, 모델을 받을 수 없을 때 (무효 토큰·사용 조건 미동의·미설치).
    """
    diarizer = Diarizer(config)
    try:
        diarizer.load()
    except DiarizationError as e:
        cause = e.__cause__
        # 토큰 없음(원인 없음)·pyannote 미설치(ImportError)는 그 안내를 그대로 낸다
        if cause is None or isinstance(cause, ImportError):
            raise
        detail = next(
            (line for line in reversed(str(cause).splitlines()) if line.strip()), str(cause)
        )
        raise DiarizationError(
            f"화자 분리 모델을 불러올 수 없습니다. {_HF_TOKEN_HELP}. 원인: {detail}"
        ) from e
    finally:
        diarizer.unload()
    log.info("화자 분리 모델 확인", model=DIARIZATION_MODEL)


@dataclass
class SpeakerTurn:
    """화자 발화 구간.

    Attributes:
        speaker: 화자 ID (예: 'SPEAKER_00').
        start: 시작 시간(초).
        end: 종료 시간(초).
    """

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
        }


@dataclass
class DiarizationResult:
    """화자 분리 결과.

    Attributes:
        turns: 화자 발화 구간 목록.
        num_speakers: 감지된 화자 수.
        embeddings: 화자 임베딩 (내보내기용). None이면 미생성.
    """

    turns: list[SpeakerTurn] = field(default_factory=list)
    num_speakers: int = 0
    embeddings: Any = None

    @property
    def speakers(self) -> list[str]:
        """고유 화자 ID 목록."""
        return sorted(set(t.speaker for t in self.turns))

    def relabel(self, mapping: dict[str, str]) -> None:
        """화자 ID 를 바꾼다 (in-place, 임베딩 키도 함께). mapping 에 없는 ID 는 그대로 둔다.

        Args:
            mapping: ``{지금 ID: 새 ID}`` (``match_known_speakers`` 결과).
        """
        for turn in self.turns:
            turn.speaker = mapping.get(turn.speaker, turn.speaker)
        if isinstance(self.embeddings, dict):
            self.embeddings = {mapping.get(k, k): v for k, v in self.embeddings.items()}


class Diarizer:
    """pyannote.audio 기반 화자 분리기.

    HuggingFace 토큰으로 인증하여 pyannote 화자 분리 파이프라인을 실행한다.
    """

    def __init__(self, config: DiarizationConfig) -> None:
        self.config = config
        self._pipeline: Any = None
        self._is_loaded: bool = False

    def load(self) -> None:
        """pyannote 파이프라인을 로드한다.

        Raises:
            DiarizationError: pyannote 로드 실패 시.
        """
        try:
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise DiarizationError(
                "pyannote.audio 패키지가 설치되지 않았습니다. "
                "pip install -r requirements.txt 로 설치해 주세요."
            ) from e

        hf_token = self.config.hf_token or os.environ.get("HF_TOKEN", "")
        if not hf_token:
            raise DiarizationError(
                "HuggingFace 토큰이 필요합니다. "
                "--hf-token 옵션 또는 HF_TOKEN 환경변수를 설정해 주세요."
            )

        try:
            # pyannote.audio 4.x 는 token=, 3.x 는 use_auth_token= 으로 토큰을 받는다
            # (whisperx 3.8 이 pyannote.audio 4 를 함께 설치한다)
            token_arg = (
                "token"
                if "token" in inspect.signature(Pipeline.from_pretrained).parameters
                else "use_auth_token"
            )
            self._pipeline = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                **{token_arg: hf_token},
            )
            self._is_loaded = True
            log.info("pyannote 파이프라인 로드 완료")
        except Exception as e:
            raise DiarizationError(f"pyannote 파이프라인 로드 실패: {e}") from e

    def unload(self) -> None:
        """파이프라인을 언로드한다."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._is_loaded = False
            log.info("pyannote 파이프라인 언로드 완료")

    def diarize(self, audio_path: str | Path) -> DiarizationResult:
        """오디오 파일에서 화자 분리를 수행한다.

        Args:
            audio_path: 오디오 파일 경로 (WAV 권장).

        Returns:
            DiarizationResult.

        Raises:
            DiarizationError: 화자 분리 실패 시.
        """
        if not self._is_loaded or self._pipeline is None:
            raise DiarizationError("파이프라인이 로드되지 않았습니다. load()를 먼저 호출해 주세요.")

        path = Path(audio_path)
        if not path.exists():
            raise DiarizationError(f"오디오 파일이 존재하지 않습니다: {path}")

        log.info("화자 분리 시작", audio=str(path))

        try:
            # 화자 수 설정
            kwargs: dict[str, Any] = {}
            if self.config.num_speakers != "auto":
                with contextlib.suppress(ValueError):
                    kwargs["num_speakers"] = int(self.config.num_speakers)

            diarization = self._pipeline(str(path), **kwargs)

            # 결과 변환
            # pyannote.audio 4.x 는 Annotation 을 speaker_diarization 속성에 담아 돌려준다
            annotation = getattr(diarization, "speaker_diarization", diarization)
            turns = _convert_annotation(annotation)
            speakers = sorted(set(t.speaker for t in turns))

            result = DiarizationResult(
                turns=turns,
                num_speakers=len(speakers),
                embeddings=_speaker_embeddings(diarization, annotation),
            )

            log.info(
                "화자 분리 완료",
                speakers=result.num_speakers,
                turns=len(turns),
            )
            return result

        except Exception as e:
            if isinstance(e, DiarizationError):
                raise
            raise DiarizationError(f"화자 분리 실패: {e}") from e

    def __enter__(self) -> Diarizer:
        self.load()
        return self

    def __exit__(self, *args: object) -> None:
        self.unload()


def _speaker_embeddings(output: Any, annotation: Any) -> dict[str, list[float]] | None:
    """화자별 임베딩 ``{화자 ID: 벡터}``. 없으면 None.

    pyannote.audio 4.x 는 결과의 ``speaker_embeddings`` (화자 수, 차원 배열) 를 ``annotation.labels()``
    순서로 준다. 3.x 는 임베딩을 돌려주지 않는다 (``return_embeddings`` 를 따로 줘야 한다).
    값이 NaN 인 행(임베딩을 못 만든 화자)은 뺀다.
    """
    raw = getattr(output, "speaker_embeddings", None)
    if raw is None:
        return None
    labels = list(annotation.labels())
    result: dict[str, list[float]] = {}
    for label, row in zip(labels, raw, strict=False):
        vector = [float(x) for x in row]
        if vector and all(x == x for x in vector):  # NaN 은 자기 자신과 같지 않다
            result[str(label)] = vector
    return result or None


def _convert_annotation(annotation: Any) -> list[SpeakerTurn]:
    """pyannote Annotation 객체를 SpeakerTurn 목록으로 변환한다."""
    turns: list[SpeakerTurn] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            )
        )
    # 시간순 정렬
    turns.sort(key=lambda t: t.start)
    return turns


def assign_speakers(
    segments: list[TranscriptionSegment],
    diarization: DiarizationResult,
) -> None:
    """STT 세그먼트에 화자 ID를 할당한다 (in-place).

    각 세그먼트의 시간 범위와 가장 많이 겹치는 화자를 할당한다.

    Args:
        segments: STT 세그먼트 목록.
        diarization: 화자 분리 결과.
    """
    if not diarization.turns:
        return

    for seg in segments:
        best_speaker = _find_best_speaker(seg.start, seg.end, diarization.turns)
        if best_speaker is not None:
            seg.speaker = best_speaker

        # 단어 레벨 화자 할당은 세그먼트 화자를 상속
        # (단어별 독립 할당은 speaker_splitter에서 수행)


def _find_best_speaker(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
) -> str | None:
    """주어진 시간 범위와 가장 많이 겹치는 화자를 찾는다."""
    overlap_by_speaker: dict[str, float] = {}

    for turn in turns:
        overlap_start = max(start, turn.start)
        overlap_end = min(end, turn.end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > 0:
            overlap_by_speaker[turn.speaker] = overlap_by_speaker.get(turn.speaker, 0.0) + overlap

    if not overlap_by_speaker:
        return None

    return max(overlap_by_speaker, key=overlap_by_speaker.get)  # type: ignore[arg-type]
