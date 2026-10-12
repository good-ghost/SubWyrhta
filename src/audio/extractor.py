"""ffmpeg 기반 오디오 추출기.

영상/음성 파일에서 16kHz mono WAV를 추출한다.
트랙 선택, 시간 범위 추출, 챕터 메타데이터, 임시 파일 관리를 담당한다.
"""

from __future__ import annotations

import json as _json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import structlog

from src.core.exceptions import AudioExtractionError

log = structlog.get_logger(__name__)

# 지원 입력 확장자
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".m4v", ".mkv", ".avi", ".mov", ".webm", ".mpeg", ".mpg", ".wmv", ".flv"}
)
# .mka 는 Matroska 오디오 (컨테이너는 .mkv 와 같아 ffmpeg 가 그대로 읽는다, TO-DO 8)
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".mka"}
)
SUPPORTED_EXTENSIONS: frozenset[str] = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

# STT 표준 출력 설정
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_FORMAT = "wav"
TARGET_CODEC = "pcm_s16le"


def _find_ffmpeg() -> str:
    """ffmpeg 실행 파일 경로를 찾는다.

    snap 설치 ffmpeg도 자동 감지한다.

    Raises:
        AudioExtractionError: ffmpeg를 찾을 수 없을 때.
    """
    # 일반 경로
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        return ffmpeg_path

    # snap 설치 경로
    snap_path = Path("/snap/bin/ffmpeg")
    if snap_path.exists():
        return str(snap_path)

    raise AudioExtractionError(
        "ffmpeg를 찾을 수 없습니다. ffmpeg를 설치해 주세요: "
        "apt install ffmpeg 또는 snap install ffmpeg"
    )


def _find_ffprobe() -> str:
    """ffprobe 실행 파일 경로를 찾는다.

    Raises:
        AudioExtractionError: ffprobe를 찾을 수 없을 때.
    """
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is not None:
        return ffprobe_path

    # snap 설치: ffmpeg.ffprobe 또는 ffprobe
    for snap_name in ("/snap/bin/ffmpeg.ffprobe", "/snap/bin/ffprobe"):
        snap_path = Path(snap_name)
        if snap_path.exists():
            return str(snap_path)

    raise AudioExtractionError("ffprobe를 찾을 수 없습니다. ffmpeg를 설치해 주세요.")


# ffmpeg 오류 문구로 남길 stderr 끝 줄 수 (앞부분은 버전 배너·빌드 설정이라 원인이 아니다)
STDERR_TAIL_LINES = 3


def stderr_tail(stderr: str | None, lines: int = STDERR_TAIL_LINES) -> str:
    """ffmpeg stderr 의 마지막 몇 줄 (실패 원인은 끝에 있다).

    Args:
        stderr: ffmpeg 표준 오류 출력.
        lines: 남길 줄 수.

    Returns:
        끝 줄들을 ' / ' 로 이은 문자열.
    """
    rows = [row.strip() for row in (stderr or "").splitlines() if row.strip()]
    return " / ".join(rows[-lines:])


def _validate_input(input_path: Path) -> None:
    """입력 파일을 검증한다."""
    if not input_path.exists():
        raise AudioExtractionError(f"입력 파일이 존재하지 않습니다: {input_path}")
    if not input_path.is_file():
        raise AudioExtractionError(f"입력 경로가 파일이 아닙니다: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioExtractionError(
            f"지원하지 않는 파일 형식입니다: {input_path.suffix} "
            f"(지원: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )


def get_audio_duration(input_path: str | Path) -> float:
    """ffprobe로 오디오/영상 파일의 총 길이(초)를 반환한다.

    Args:
        input_path: 입력 파일 경로.

    Returns:
        총 길이(초).

    Raises:
        AudioExtractionError: 길이를 구할 수 없을 때.
    """
    ffprobe = _find_ffprobe()
    path = Path(input_path)
    _validate_input(path)

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
        raise AudioExtractionError(f"오디오 길이를 구할 수 없습니다: {path} ({e})") from e


def get_audio_track_count(input_path: str | Path) -> int:
    """ffprobe로 오디오 트랙 수를 반환한다.

    Args:
        input_path: 입력 파일 경로.

    Returns:
        오디오 트랙 수.
    """
    ffprobe = _find_ffprobe()
    path = Path(input_path)

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        return len(lines)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 0


# 언어 코드(ISO 639-1) → 오디오 트랙 언어 태그 (mkv 는 ISO 639-2 를 쓴다: jpn, eng)
_TRACK_LANGUAGE_TAGS: dict[str, frozenset[str]] = {
    "ja": frozenset({"ja", "jpn"}),
    "en": frozenset({"en", "eng"}),
    "ko": frozenset({"ko", "kor"}),
    "zh": frozenset({"zh", "zho", "chi", "cmn"}),
    "fr": frozenset({"fr", "fra", "fre"}),
    "de": frozenset({"de", "deu", "ger"}),
    "es": frozenset({"es", "spa"}),
    "it": frozenset({"it", "ita"}),
    "pt": frozenset({"pt", "por"}),
    "ru": frozenset({"ru", "rus"}),
}


def get_audio_track_languages(input_path: str | Path) -> list[str]:
    """오디오 트랙별 언어 태그 (소문자, 없으면 빈 문자열). ffprobe 가 실패하면 빈 목록.

    Args:
        input_path: 입력 파일 경로.

    Returns:
        오디오 트랙 순서대로의 언어 태그 목록.
    """
    try:
        ffprobe = _find_ffprobe()
    except AudioExtractionError as e:
        log.warning("오디오 트랙 언어 조회 실패", input=str(input_path), error=str(e))
        return []
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index:stream_tags=language",
        "-of",
        "csv=p=0",
        str(input_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("오디오 트랙 언어 조회 실패", input=str(input_path), error=str(e))
        return []
    languages = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split(",")
        languages.append(fields[1].strip().lower() if len(fields) > 1 else "")
    return languages


def find_audio_track_for_language(input_path: str | Path, language: str) -> int | None:
    """트랙이 여럿일 때 언어 태그가 맞는 첫 오디오 트랙 번호(0-based). 정할 수 없으면 None.

    영상에 더빙 트랙이 여럿이면 기본 트랙이 다른 언어일 수 있다 (예: 기본 eng, 두 번째 jpn).
    그 트랙을 지정한 언어로 인식하면 엉뚱한 자막이 나온다.

    Args:
        input_path: 입력 파일 경로.
        language: 음성 언어 코드 (ISO 639-1, 예: ja).

    Returns:
        트랙 번호. 트랙이 하나뿐이거나 맞는 태그가 없으면 None (기본 트랙).
    """
    tags = _TRACK_LANGUAGE_TAGS.get(language, frozenset({language}))
    languages = get_audio_track_languages(input_path)
    if len(languages) <= 1:
        return None
    for index, tag in enumerate(languages):
        if tag in tags:
            return index
    log.warning(
        "언어에 맞는 오디오 트랙이 없어 기본 트랙을 쓴다", language=language, tracks=languages
    )
    return None


@dataclass
class Chapter:
    """영상/음성 파일의 챕터 정보."""

    index: int
    title: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """챕터 길이(초)."""
        return self.end_s - self.start_s

    @property
    def start_tc(self) -> str:
        """시작 타임코드 (HH:MM:SS.mmm)."""
        return _seconds_to_timecode(self.start_s)

    @property
    def end_tc(self) -> str:
        """종료 타임코드 (HH:MM:SS.mmm)."""
        return _seconds_to_timecode(self.end_s)


def _seconds_to_timecode(seconds: float) -> str:
    """초를 HH:MM:SS.mmm 형식으로 변환한다."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def get_chapters(input_path: str | Path) -> list[Chapter]:
    """ffprobe로 MKV/MP4 챕터 메타데이터를 추출한다.

    Args:
        input_path: 입력 영상/음성 파일 경로.

    Returns:
        Chapter 목록. 챕터가 없으면 빈 리스트.
    """
    ffprobe = _find_ffprobe()
    path = Path(input_path)

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_chapters",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("챕터 메타데이터 추출 실패", path=str(path), error=str(e))
        return []

    try:
        data = _json.loads(result.stdout)
    except _json.JSONDecodeError:
        return []

    raw_chapters = data.get("chapters", [])
    chapters: list[Chapter] = []

    for i, ch in enumerate(raw_chapters):
        try:
            start = float(ch.get("start_time", 0))
            end = float(ch.get("end_time", 0))
            tags = ch.get("tags", {})
            title = tags.get("title", f"Chapter {i + 1}")

            chapters.append(
                Chapter(
                    index=i,
                    title=title,
                    start_s=start,
                    end_s=end,
                )
            )
        except (ValueError, TypeError):
            continue

    if chapters:
        log.info("챕터 메타데이터 추출 완료", count=len(chapters))

    return chapters


class AudioExtractor:
    """ffmpeg 기반 오디오 추출기.

    영상/음성 파일에서 STT에 적합한 16kHz mono WAV를 추출한다.

    Attributes:
        keep_audio: True이면 추출된 오디오를 삭제하지 않는다.
    """

    def __init__(self, keep_audio: bool = False) -> None:
        self._ffmpeg = _find_ffmpeg()
        self._temp_files: list[Path] = []
        self.keep_audio = keep_audio

    def extract(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        audio_track: int | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> Path:
        """오디오를 16kHz mono WAV로 추출한다.

        Args:
            input_path: 입력 영상/음성 파일 경로.
            output_path: 출력 WAV 경로. None이면 임시 파일 생성.
            audio_track: 오디오 트랙 인덱스 (0-based). None이면 기본 트랙.
            start: 시작 시간 (HH:MM:SS 또는 초). None이면 처음부터.
            end: 종료 시간 (HH:MM:SS 또는 초). None이면 끝까지.

        Returns:
            추출된 WAV 파일의 Path.

        Raises:
            AudioExtractionError: 추출 실패 시.
        """
        path = Path(input_path)
        _validate_input(path)

        # 출력 경로 결정
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            import os

            os.close(fd)
            out = Path(tmp)
            self._temp_files.append(out)

        cmd = self._build_command(
            input_path=path,
            output_path=out,
            audio_track=audio_track,
            start=start,
            end=end,
        )

        log.info(
            "오디오 추출 시작",
            input=str(path),
            output=str(out),
            audio_track=audio_track,
            start=start,
            end=end,
        )

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )
        except subprocess.CalledProcessError as e:
            raise AudioExtractionError(f"ffmpeg 오디오 추출 실패: {stderr_tail(e.stderr)}") from e
        except subprocess.TimeoutExpired as e:
            raise AudioExtractionError(f"ffmpeg 오디오 추출 타임아웃 (10분 초과): {path}") from e

        if not out.exists() or out.stat().st_size == 0:
            raise AudioExtractionError(f"추출된 오디오 파일이 비어 있습니다: {out}")

        log.info(
            "오디오 추출 완료", output=str(out), size_mb=round(out.stat().st_size / 1024 / 1024, 2)
        )
        return out

    def extract_by_chapters(
        self,
        input_path: str | Path,
        chapters: list[Chapter],
        audio_track: int | None = None,
    ) -> list[tuple[Chapter, Path]]:
        """챕터별로 독립적인 오디오 파일을 추출한다.

        Args:
            input_path: 입력 영상/음성 파일 경로.
            chapters: 챕터 목록.
            audio_track: 오디오 트랙 인덱스.

        Returns:
            (Chapter, 추출된 WAV 경로) 튜플 목록.
        """
        results: list[tuple[Chapter, Path]] = []

        for chapter in chapters:
            audio_path = self.extract(
                input_path=input_path,
                audio_track=audio_track,
                start=chapter.start_tc,
                end=chapter.end_tc,
            )
            results.append((chapter, audio_path))
            log.info(
                "챕터별 오디오 추출",
                chapter=chapter.title,
                start=chapter.start_tc,
                end=chapter.end_tc,
            )

        return results

    def _build_command(
        self,
        input_path: Path,
        output_path: Path,
        audio_track: int | None,
        start: str | None,
        end: str | None,
    ) -> list[str]:
        """ffmpeg 명령어를 구성한다."""
        cmd: list[str] = [self._ffmpeg]

        # 시간 범위 (입력 전에 위치해야 seek이 빠르다)
        if start is not None:
            cmd.extend(["-ss", start])
        if end is not None:
            cmd.extend(["-to", end])

        cmd.extend(["-i", str(input_path)])

        # 오디오 트랙 선택
        if audio_track is not None:
            cmd.extend(["-map", f"0:a:{audio_track}"])
        else:
            cmd.extend(["-map", "0:a:0"])

        # 채널 믹스다운 + 샘플레이트 변환
        cmd.extend(
            [
                "-ac",
                str(TARGET_CHANNELS),
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-c:a",
                TARGET_CODEC,
                "-f",
                TARGET_FORMAT,
                "-vn",  # 비디오 스트림 제거
                "-y",  # 덮어쓰기
                str(output_path),
            ]
        )

        return cmd

    def cleanup(self) -> None:
        """임시 파일을 삭제한다.

        keep_audio=True이면 삭제를 건너뛴다.
        """
        if self.keep_audio:
            log.debug("keep_audio=True, 임시 파일 보존", count=len(self._temp_files))
            return

        for f in self._temp_files:
            try:
                if f.exists():
                    f.unlink()
                    log.debug("임시 파일 삭제", path=str(f))
            except OSError as e:
                log.warning("임시 파일 삭제 실패", path=str(f), error=str(e))

        self._temp_files.clear()

    def __enter__(self) -> AudioExtractor:
        return self

    def __exit__(self, *args: object) -> None:
        self.cleanup()
