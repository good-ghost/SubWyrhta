"""SubWyrhta CLI 진입점.

click 기반 서브커맨드 구조: transcribe, diarize, info, list-models.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import structlog

from src.core.config import AppConfig, load_env_file
from src.core.logging import setup_logging
from src.engines.engine_factory import get_available_engines
from src.pipeline import PipelineOptions, TranscriptionPipeline
from src.segmentation.llm_client import LLM_PROVIDERS
from src.version import __version__

if TYPE_CHECKING:
    from src.models.result import TranscriptionResult

log = structlog.get_logger(__name__)

# --model 도움말에 보일 엔진 목록 (엔진 팩토리에 등록된 것)
_ENGINE_CHOICES_HELP = " / ".join(get_available_engines())


# ──────────────────────────────────────────────
# 메인 그룹
# ──────────────────────────────────────────────


@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="설정 파일 경로 (YAML)",
)
@click.option(
    "--env-file",
    "env_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="환경 변수 파일 (기본: 현재 폴더의 .env, 없으면 건너뛴다). 셸 환경 변수가 우선한다",
)
@click.option("--verbose", is_flag=True, default=False, help="상세 로그 (DEBUG)")
@click.option("--quiet", is_flag=True, default=False, help="최소 로그 (WARNING)")
@click.version_option(version=__version__, prog_name="subwyrhta")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: str | None,
    env_file: Path | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """SubWyrhta — 영상/음성 파일에서 고품질 자막을 자동 생성하는 STT 도구."""
    # 설정을 읽기 전에 넣는다 (HF_TOKEN·SUBWYRHTA_* 덮어쓰기, LLM API 키)
    loaded_env = load_env_file(env_file)
    # 로깅 레벨 결정
    if verbose:
        log_level = "DEBUG"
    elif quiet:
        log_level = "WARNING"
    else:
        log_level = "INFO"

    setup_logging(level=log_level)
    if loaded_env:
        # 값은 남기지 않는다 (비밀 값)
        log.debug("환경 변수 파일 읽음", keys=loaded_env)

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["log_level"] = log_level


# ──────────────────────────────────────────────
# transcribe 서브커맨드
# ──────────────────────────────────────────────


@cli.command()
@click.argument("input_path", required=False)
# 배치 입력 (REQUIREMENT 2.3·11.1)
@click.option(
    "--recursive", is_flag=True, default=False, help="디렉토리·glob 입력을 하위 폴더까지 찾는다"
)
@click.option(
    "--file-list",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="처리할 파일 목록 (한 줄에 하나, # 은 주석)",
)
@click.option(
    "--workers", type=click.IntRange(min=1), default=1, help="병렬 worker 수 (파일 여러 개일 때)"
)
@click.option(
    "--workers-device",
    type=click.Choice(["auto", "cuda", "cpu", "mps"]),
    default=None,
    help="모든 worker 의 디바이스 (기본: GPU 는 worker 하나만, 나머지 CPU)",
)
@click.option("--failed-list", default=None, help="실패한 파일 목록 저장 경로")
# 입력/출력 옵션
@click.option("--output-dir", default=None, help="출력 디렉토리 (기본: ./output)")
@click.option(
    "--output-format",
    "output_format",
    default=None,
    type=click.Choice(["srt", "ass", "vtt", "json"]),
    help="출력 포맷",
)
@click.option("--output-name", default=None, help="출력 파일명 직접 지정")
@click.option("--keep-audio", is_flag=True, default=False, help="추출 오디오 보존")
@click.option("--audio-track", type=int, default=None, help="오디오 트랙 번호 (0-based)")
@click.option("--start", default=None, help="처리 시작 시간 (HH:MM:SS)")
@click.option("--end", default=None, help="처리 종료 시간 (HH:MM:SS)")
# STT 옵션
@click.option("--model", default=None, help=f"STT 엔진 ({_ENGINE_CHOICES_HELP})")
@click.option("--model-size", default=None, help="모델 크기 (large-v3 등)")
@click.option("--model-dir", default=None, help="모델 캐시 경로")
@click.option("--source-lang", default=None, help="음성 언어 (auto / en / ko / ja / zh)")
@click.option(
    "--device", default=None, type=click.Choice(["auto", "cuda", "cpu", "mps"]), help="디바이스"
)
@click.option(
    "--compute-type",
    default=None,
    type=click.Choice(["auto", "float16", "float32", "int8"]),
    help="compute_type",
)
@click.option("--batch-size", type=int, default=None, help="배치 추론 크기")
@click.option("--beam-size", type=int, default=None, help="Beam search 크기")
@click.option("--temperature", type=float, default=None, help="샘플링 온도 (0.0=greedy)")
# LLM 옵션 (llm-assisted 세그멘테이션용)
@click.option(
    "--llm-provider",
    default=None,
    type=click.Choice(LLM_PROVIDERS),
    help="LLM 프로바이더",
)
@click.option("--llm-model", default=None, help="LLM 모델명")
@click.option("--llm-base-url", default=None, help="LLM 커스텀 엔드포인트 URL")
# 오디오 전처리·VAD 옵션 (지정하지 않으면 설정 파일 값)
@click.option("--denoise/--no-denoise", default=None, help="노이즈 억제")
@click.option("--separate-vocals/--no-separate-vocals", default=None, help="demucs 배경음 분리")
@click.option("--normalize-audio/--no-normalize-audio", default=None, help="음량 정규화")
@click.option("--vad/--no-vad", "vad", default=None, help="VAD 사용")
@click.option(
    "--pre-vad/--no-pre-vad",
    "pre_vad",
    default=None,
    help="Pre-VAD (끄면 전체 추론 후 사후 재분할)",
)
@click.option("--vad-export", is_flag=True, default=False, help="VAD 타임라인 JSON 내보내기")
# 세그멘테이션·화자 분리·품질 옵션
@click.option(
    "--segmentation-mode",
    default=None,
    type=click.Choice(["basic", "advanced", "extreme", "llm-assisted"]),
    help="세그멘테이션 모드",
)
@click.option("--max-cps", type=float, default=None, help="최대 CPS (미지정 시 언어별 자동)")
@click.option("--diarize/--no-diarize", "diarize_flag", default=None, help="화자 분리")
@click.option("--num-speakers", default=None, help="화자 수 (auto 또는 정수)")
@click.option("--hf-token", default=None, help="HuggingFace 토큰 (화자 분리)")
@click.option("--qc/--no-qc", "qc", default=None, help="품질 검사")
@click.option("--quality-report", is_flag=True, default=False, help="품질 리포트 JSON 저장")
@click.option("--export-words", is_flag=True, default=False, help="단어 타임스탬프 JSON 함께 저장")
@click.option(
    "--resume/--no-resume",
    "resume",
    default=None,
    help=(
        "체크포인트·단계 결과(<입력>.work)에서 재개 / 남은 것을 지우고 처음부터 (기본: 쓰지 않음)"
    ),
)
# 챕터 옵션
@click.option("--split-by-chapters", is_flag=True, default=False, help="챕터별 독립 자막 파일 생성")
# 기타
@click.option("--dry-run", is_flag=True, default=False, help="실제 처리 없이 계획 출력")
@click.pass_context
def transcribe(
    ctx: click.Context,
    input_path: str | None,
    recursive: bool,
    file_list: str | None,
    workers: int,
    workers_device: str | None,
    failed_list: str | None,
    output_dir: str | None,
    output_format: str | None,
    output_name: str | None,
    keep_audio: bool,
    audio_track: int | None,
    start: str | None,
    end: str | None,
    model: str | None,
    model_size: str | None,
    model_dir: str | None,
    source_lang: str | None,
    device: str | None,
    compute_type: str | None,
    batch_size: int | None,
    beam_size: int | None,
    temperature: float | None,
    llm_provider: str | None,
    llm_model: str | None,
    llm_base_url: str | None,
    denoise: bool | None,
    separate_vocals: bool | None,
    normalize_audio: bool | None,
    vad: bool | None,
    pre_vad: bool | None,
    vad_export: bool,
    segmentation_mode: str | None,
    max_cps: float | None,
    diarize_flag: bool | None,
    num_speakers: str | None,
    hf_token: str | None,
    qc: bool | None,
    quality_report: bool,
    export_words: bool,
    resume: bool | None,
    split_by_chapters: bool,
    dry_run: bool,
) -> None:
    """음성/영상 파일을 자막으로 변환한다.

    INPUT_PATH 는 파일, 디렉토리, glob 패턴("videos/*.mp4") 중 하나다. --file-list 와 함께 쓸 수 있다.
    """
    files, is_batch = _resolve_inputs(input_path, recursive, file_list)
    if is_batch and output_name:
        raise click.UsageError("--output-name 은 파일 하나를 처리할 때만 쓸 수 있습니다")
    config_path = ctx.obj["config_path"]

    # CLI 오버라이드 수집
    cli_overrides = _collect_stt_overrides(
        model=model,
        model_size=model_size,
        model_dir=model_dir,
        source_lang=source_lang,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        beam_size=beam_size,
        temperature=temperature,
    )
    if output_dir is not None:
        cli_overrides["output.dir"] = output_dir
    if output_format is not None:
        cli_overrides["output.format"] = output_format
    _collect_llm_overrides(cli_overrides, llm_provider, llm_model, llm_base_url)
    cli_overrides.update(
        _collect_pipeline_overrides(
            denoise=denoise,
            separate_vocals=separate_vocals,
            normalize_audio=normalize_audio,
            vad=vad,
            pre_vad=pre_vad,
            segmentation_mode=segmentation_mode,
            max_cps=max_cps,
            diarize=diarize_flag,
            num_speakers=num_speakers,
            hf_token=hf_token,
            qc=qc,
            quality_report=True if quality_report else None,
            export_words=True if export_words else None,
        )
    )

    config = AppConfig.load(config_path=config_path, cli_overrides=cli_overrides)

    if dry_run:
        if is_batch:
            click.echo(f"=== 배치 {len(files)}개 (worker {min(workers, len(files))}) ===")
            for path in files:
                click.echo(f"  {path}")
            click.echo()
        _print_dry_run(config, str(files[0]), output_name, keep_audio, audio_track, start, end)
        return

    options = PipelineOptions(
        output_name=output_name,
        keep_audio=keep_audio,
        audio_track=audio_track,
        start=start,
        end=end,
        resume=resume is True,
        vad_export=vad_export,
        keep_stages=True,
    )
    job = TranscribeJob(
        config=config,
        options=options,
        split_by_chapters=split_by_chapters,
        reset_checkpoint=resume is False,
        log_level=ctx.obj["log_level"],
    )
    if not is_batch:
        try:
            job.run_one(files[0])
        finally:
            job.close()
        return

    _run_batch(job, files, workers=workers, workers_device=workers_device, failed_list=failed_list)


# ──────────────────────────────────────────────
# info 서브커맨드
# ──────────────────────────────────────────────


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
def info(input_path: str) -> None:
    """입력 파일 정보를 출력한다 (duration, tracks 등)."""
    from src.audio.extractor import get_audio_duration, get_audio_track_count

    path = Path(input_path)
    click.echo(f"파일: {path.name}")
    click.echo(f"경로: {path.resolve()}")
    click.echo(f"크기: {path.stat().st_size / 1024 / 1024:.1f} MB")

    try:
        duration = get_audio_duration(input_path)
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60
        click.echo(f"길이: {hours:02d}:{minutes:02d}:{seconds:05.2f} ({duration:.1f}초)")
    except Exception as e:
        click.echo(f"길이: 확인 불가 ({e})")

    try:
        tracks = get_audio_track_count(input_path)
        click.echo(f"오디오 트랙: {tracks}개")
    except Exception:
        pass


# ──────────────────────────────────────────────
# validate 서브커맨드
# ──────────────────────────────────────────────


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--auto-fix", is_flag=True, default=False, help="자동 교정 후 결과 파일 저장")
@click.option("--output", "output_path", default=None, help="교정 결과 저장 경로")
@click.option("--max-chars", type=int, default=42, help="줄당 최대 글자 수")
@click.option("--max-cps", type=float, default=None, help="최대 CPS")
@click.option("--min-duration", type=float, default=1.0, help="자막 최소 표시 시간 (초)")
@click.pass_context
def validate(
    ctx: click.Context,
    input_path: str,
    auto_fix: bool,
    output_path: str | None,
    max_chars: int,
    max_cps: float | None,
    min_duration: float,
) -> None:
    """생성된 자막 파일의 품질을 검사한다."""
    _run_validate(input_path, auto_fix, output_path, max_chars, max_cps, min_duration)


def _run_validate(
    input_path: str,
    auto_fix: bool,
    output_path: str | None,
    max_chars: int,
    max_cps: float | None,
    min_duration: float,
) -> None:
    """validate 실행 로직."""
    from src.core.config import QualityConfig, SegmentationConfig
    from src.quality.checker import check_quality

    path = Path(input_path)
    ext = path.suffix.lower()

    # 파일 로드
    result = _load_subtitle_file(path, ext)
    if result is None:
        click.echo(f"오류: 지원하지 않는 파일 형식입니다: {ext}")
        click.echo("지원 포맷: .srt, .ass, .vtt, .json")
        return

    # 품질 검사
    seg_config = SegmentationConfig(
        max_chars_per_line=max_chars,
        min_duration_s=min_duration,
    )
    if max_cps is not None:
        seg_config.max_cps = max_cps

    qc_config = QualityConfig(enabled=True, auto_fix=auto_fix)
    report = check_quality(result, qc_config, seg_config)

    # 결과 출력
    click.echo(f"\n[검사 결과] {path.name}")
    click.echo(f"  세그먼트 수:    {report.total_segments}")
    click.echo(f"  오류:           {report.error_count}")
    click.echo(f"  경고:           {report.warning_count}")
    if report.auto_fixed_count > 0:
        click.echo(f"  자동 교정:      {report.auto_fixed_count}")
    if report.quality_summary.grade:
        click.echo(f"  품질 등급:      {report.quality_summary.grade}")
        click.echo(f"  평균 신뢰도:    {report.quality_summary.avg_confidence:.4f}")

    if report.issues:
        click.echo("\n  상세 이슈:")
        for issue in report.issues[:20]:
            fixed = " [교정됨]" if issue.auto_fixed else ""
            click.echo(
                f"    [{issue.severity.value}] seg {issue.segment_index}: {issue.message}{fixed}"
            )
        if len(report.issues) > 20:
            click.echo(f"    ... 외 {len(report.issues) - 20}건")

    # 교정 결과 저장
    if auto_fix and (report.auto_fixed_count > 0 or output_path):
        save_path = output_path or str(path.with_suffix(f".fixed{ext}"))
        from src.api import save_result

        save_result(result, save_path)
        click.echo(f"\n  교정 저장: {save_path}")


def _load_subtitle_file(path: Path, ext: str) -> TranscriptionResult | None:
    """자막 파일을 TranscriptionResult로 로드한다."""
    from src.models.result import TranscriptionResult

    if ext == ".json":
        return TranscriptionResult.load(path)

    if ext == ".srt":
        return _parse_srt(path)

    # .ass/.vtt → 기본 파싱 (줄 단위)
    if ext in (".ass", ".vtt"):
        return _parse_simple_subtitle(path, ext)

    return None


def _parse_srt(path: Path) -> TranscriptionResult:
    """SRT 파일을 파싱한다."""
    import re

    from src.models.result import Metadata, TranscriptionResult
    from src.models.segment import TranscriptionSegment

    content = path.read_text(encoding="utf-8")
    segments: list[TranscriptionSegment] = []

    # SRT 블록 파싱: 인덱스\n타임코드\n텍스트\n
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        # 인덱스
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # 타임코드
        tc_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1].strip(),
        )
        if not tc_match:
            continue

        g = tc_match.groups()
        start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000

        text = "\n".join(lines[2:])

        segments.append(
            TranscriptionSegment(
                index=index,
                text=text,
                start=start,
                end=end,
            )
        )

    return TranscriptionResult(
        metadata=Metadata(source_file=str(path)),
        segments=segments,
    )


def _parse_simple_subtitle(path: Path, ext: str) -> TranscriptionResult:
    """ASS/VTT 파일을 간이 파싱한다 (검증 목적)."""
    import re

    from src.models.result import Metadata, TranscriptionResult
    from src.models.segment import TranscriptionSegment

    content = path.read_text(encoding="utf-8")
    segments: list[TranscriptionSegment] = []
    index = 1

    if ext == ".vtt":
        # VTT: HH:MM:SS.mmm --> HH:MM:SS.mmm
        for match in re.finditer(
            r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*\n(.+?)(?=\n\n|\n\d{2}:|\Z)",
            content,
            re.DOTALL,
        ):
            g = match.groups()
            start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
            end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
            text = g[8].strip()
            segments.append(TranscriptionSegment(index=index, text=text, start=start, end=end))
            index += 1

    elif ext == ".ass":
        # ASS: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        for match in re.finditer(
            r"Dialogue:\s*\d+,(\d+):(\d{2}):(\d{2})\.(\d{2}),(\d+):(\d{2}):(\d{2})\.(\d{2}),[^,]*,[^,]*,\d+,\d+,\d+,[^,]*,(.*)",
            content,
        ):
            g = match.groups()
            start = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 100
            end = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 100
            text = g[8].replace("\\N", "\n").strip()
            segments.append(TranscriptionSegment(index=index, text=text, start=start, end=end))
            index += 1

    return TranscriptionResult(
        metadata=Metadata(source_file=str(path)),
        segments=segments,
    )


# ──────────────────────────────────────────────
# enhance 서브커맨드
# ──────────────────────────────────────────────


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output-dir", default="./output", help="출력 디렉토리")
@click.option("--denoise", is_flag=True, default=False, help="노이즈 억제")
@click.option("--separate-vocals", is_flag=True, default=False, help="demucs 배경음 분리")
@click.option("--normalize/--no-normalize", default=True, help="음량 정규화")
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "cuda", "cpu", "mps"]),
    help="demucs 디바이스",
)
def enhance(
    input_path: str,
    output_dir: str,
    denoise: bool,
    separate_vocals: bool,
    normalize: bool,
    device: str,
) -> None:
    """오디오 품질 향상 전용 (추출 + 전처리만 수행)."""
    from src.audio.enhancer import enhance_audio
    from src.audio.extractor import AudioExtractor
    from src.engines.base import resolve_device

    click.echo(f"오디오 향상 시작: {input_path}")
    out_dir = Path(output_dir)
    output_path = out_dir / f"{Path(input_path).stem}.enhanced.wav"

    with AudioExtractor() as extractor:
        audio_path = extractor.extract(input_path=input_path)
        enhance_audio(
            audio_path,
            output_path=output_path,
            normalize=normalize,
            denoise=denoise,
            separate_vocals=separate_vocals,
            device=resolve_device(device),
        )

    click.echo(f"완료: {output_path}")


# ──────────────────────────────────────────────
# list-models 서브커맨드
# ──────────────────────────────────────────────


@cli.command("list-models")
def list_models() -> None:
    """사용 가능한 STT 엔진 및 모델 목록을 출력한다."""
    from src.engines.base import get_device_info
    from src.engines.engine_factory import get_available_engines

    click.echo("=== 사용 가능한 STT 엔진 ===")
    click.echo()

    engines = get_available_engines()
    for engine in engines:
        click.echo(f"  - {engine}")

    click.echo()
    click.echo("=== 모델 크기 ===")
    click.echo()

    model_sizes = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
    for size in model_sizes:
        click.echo(f"  - {size}")

    click.echo()
    click.echo("=== distil-whisper 모델 ===")
    click.echo()

    distil_models = ["distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en"]
    for dm in distil_models:
        click.echo(f"  - {dm}")

    click.echo()
    click.echo("=== 디바이스 정보 ===")
    click.echo()

    device_info = get_device_info()
    click.echo(f"  사용 가능: {', '.join(device_info['available_devices'])}")
    if device_info.get("torch_version"):
        click.echo(f"  PyTorch: {device_info['torch_version']}")
    if device_info.get("cuda_device_name"):
        click.echo(f"  GPU: {device_info['cuda_device_name']}")
    if device_info.get("cuda_vram_gb"):
        click.echo(f"  VRAM: {device_info['cuda_vram_gb']} GB")


# ──────────────────────────────────────────────
# 내부 헬퍼
# ──────────────────────────────────────────────


def _collect_stt_overrides(**kwargs: Any) -> dict[str, Any]:
    """CLI 옵션을 dot-notation 오버라이드 딕셔너리로 변환한다."""
    mapping: dict[str, str] = {
        "model": "stt.model",
        "model_size": "stt.model_size",
        "model_dir": "stt.model_dir",
        "source_lang": "stt.source_lang",
        "device": "stt.device",
        "compute_type": "stt.compute_type",
        "batch_size": "stt.batch_size",
        "beam_size": "stt.beam_size",
        "temperature": "stt.temperature",
    }
    overrides: dict[str, Any] = {}
    for key, dot_path in mapping.items():
        value = kwargs.get(key)
        if value is not None:
            overrides[dot_path] = value
    return overrides


def _collect_llm_overrides(
    overrides: dict[str, Any],
    llm_provider: str | None,
    llm_model: str | None,
    llm_base_url: str | None,
) -> None:
    """LLM CLI 옵션을 오버라이드 딕셔너리에 추가한다."""
    if llm_provider is not None:
        overrides["llm.provider"] = llm_provider
    if llm_model is not None:
        overrides["llm.model"] = llm_model
    if llm_base_url is not None:
        overrides["llm.base_url"] = llm_base_url


def _collect_pipeline_overrides(
    *,
    denoise: bool | None,
    separate_vocals: bool | None,
    normalize_audio: bool | None,
    vad: bool | None,
    pre_vad: bool | None,
    segmentation_mode: str | None,
    max_cps: float | None,
    diarize: bool | None,
    num_speakers: str | None,
    hf_token: str | None,
    qc: bool | None,
    quality_report: bool | None,
    export_words: bool | None,
) -> dict[str, Any]:
    """전처리·VAD·세그멘테이션·화자 분리·품질 CLI 옵션을 dot-notation 오버라이드로 바꾼다.

    None인 값(지정하지 않은 옵션)은 설정 파일 값을 그대로 둔다.
    """
    mapping: dict[str, Any] = {
        "audio.denoise.enabled": denoise,
        "audio.separate_vocals": separate_vocals,
        "audio.normalize": normalize_audio,
        "audio.vad.enabled": vad,
        "audio.vad.pre_vad": pre_vad,
        "segmentation.mode": segmentation_mode,
        "segmentation.max_cps": max_cps,
        "diarization.enabled": diarize,
        "diarization.num_speakers": num_speakers,
        "diarization.hf_token": hf_token,
        "quality.enabled": qc,
        "quality.export_report": quality_report,
        "output.export_words": export_words,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _print_dry_run(
    config: AppConfig,
    input_path: str,
    output_name: str | None,
    keep_audio: bool,
    audio_track: int | None,
    start: str | None,
    end: str | None,
) -> None:
    """dry-run 모드: 실제 처리 없이 계획을 출력한다."""
    click.echo("=== Dry Run ===")
    click.echo()
    click.echo(f"입력:        {input_path}")
    click.echo(f"출력 디렉토리: {config.output.dir}")
    click.echo(f"출력 포맷:    {config.output.format}")
    if output_name:
        click.echo(f"출력 파일명:  {output_name}")
    click.echo()
    click.echo(f"STT 엔진:    {config.stt.model}")
    click.echo(f"모델 크기:    {config.stt.model_size}")
    click.echo(f"디바이스:     {config.stt.device}")
    click.echo(f"compute_type: {config.stt.compute_type}")
    click.echo(f"언어:         {config.stt.source_lang}")
    click.echo(f"배치 크기:    {config.stt.batch_size}")
    click.echo(f"beam 크기:    {config.stt.beam_size}")
    click.echo(f"temperature:  {config.stt.temperature}")
    click.echo()
    if audio_track is not None:
        click.echo(f"오디오 트랙:  {audio_track}")
    if start:
        click.echo(f"시작 시간:    {start}")
    if end:
        click.echo(f"종료 시간:    {end}")
    if keep_audio:
        click.echo("추출 오디오:  보존")
    click.echo()
    click.echo("(--dry-run 모드로 실행되었습니다. 실제 처리를 수행하려면 --dry-run을 제거하세요.)")


def _run_transcribe_by_chapters(
    pipeline: TranscriptionPipeline,
    input_path: str,
    options: PipelineOptions,
) -> None:
    """챕터마다 전체 파이프라인을 실행해 챕터별 자막 파일을 만든다 (STT 모델은 한 번 로드)."""
    from src.audio.extractor import get_chapters

    chapters = get_chapters(input_path)
    if not chapters:
        click.echo("챕터 정보가 없습니다. 일반 모드로 전환합니다.")
        _run_transcribe(pipeline=pipeline, input_path=input_path, options=options)
        return

    click.echo(f"챕터 {len(chapters)}개 감지, 챕터별 처리 시작")
    stem = Path(input_path).stem
    for chapter in chapters:
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in chapter.title).strip()
        chapter_options = dataclasses.replace(
            options,
            start=chapter.start_tc,
            end=chapter.end_tc,
            # 챕터 오프셋 보정: 세그먼트 타임스탬프에 챕터 시작 시간 추가
            time_offset_s=chapter.start_s,
            output_name=f"{stem}.ch{chapter.index + 1:02d}_{safe_title}",
        )
        outcome = pipeline.run(input_path, chapter_options)
        outcome.result.metadata.duration_s = chapter.duration_s
        for path in outcome.outputs:
            click.echo(f"  챕터 {chapter.index + 1}: {chapter.title} → {path}")

    click.echo(f"챕터별 자막 생성 완료 ({len(chapters)}개)")


_GLOB_CHARS = ("*", "?", "[")


def _resolve_inputs(
    input_path: str | None, recursive: bool, file_list: str | None
) -> tuple[list[Path], bool]:
    """입력(파일·디렉토리·glob·파일 목록)을 처리할 파일 목록으로 바꾼다.

    Returns:
        (파일 목록, 배치 여부). 파일 하나를 직접 준 경우만 배치가 아니다.

    Raises:
        click.UsageError: 입력이 없거나, 찾을 수 없거나, 처리할 파일이 없을 때.
    """
    from src.batch_processor import collect_files

    if input_path is None and file_list is None:
        raise click.UsageError("입력 파일·디렉토리·glob 패턴 또는 --file-list 가 필요합니다")
    if input_path is not None and file_list is None and Path(input_path).is_file():
        return [Path(input_path)], False
    if (
        input_path is not None
        and not Path(input_path).exists()
        and not any(c in input_path for c in _GLOB_CHARS)
    ):
        raise click.UsageError(f"입력을 찾을 수 없습니다: {input_path}")
    files = collect_files(input_path, recursive=recursive, file_list=file_list)
    if not files:
        raise click.UsageError("처리할 파일이 없습니다 (지원하는 영상·음성 확장자인지 확인하세요)")
    return files, True


def _batch_devices(config: AppConfig, workers: int, workers_device: str | None) -> list[str] | None:
    """worker 별 디바이스. None 이면 설정의 stt.device 를 그대로 쓴다.

    GPU 는 worker 하나만 쓴다 (REQUIREMENT 11.1): 설정이 cuda/mps 면 worker 0 만 그 디바이스,
    auto 면 감지해서 같은 규칙. ``--workers-device`` 를 주면 모든 worker 가 그 디바이스.
    """
    from src.batch_processor import build_device_assignments

    if workers_device is not None:
        return build_device_assignments(workers, workers_device)
    if workers <= 1:
        return None
    configured = config.stt.device
    if configured == "auto":
        return build_device_assignments(workers)
    if configured == "cpu":
        return ["cpu"] * workers
    return [configured] + ["cpu"] * (workers - 1)


@dataclasses.dataclass
class TranscribeJob:
    """transcribe 의 파일 하나 처리. 배치 worker 가 pickle 해서 받는다.

    파이프라인(STT 모델)은 처음 쓸 때 만들어 두고 같은 프로세스의 다음 파일에 다시 쓴다
    (worker 마다 모델을 한 번만 불러온다).
    """

    config: AppConfig
    options: PipelineOptions
    split_by_chapters: bool = False
    # --no-resume: 남은 체크포인트·단계 결과를 지우고 처음부터
    reset_checkpoint: bool = False
    # spawn worker 는 부모의 로그 설정을 물려받지 않아 다시 설정한다 (--verbose/--quiet)
    log_level: str = "INFO"
    _pipeline: TranscriptionPipeline | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False
    )

    def __call__(self, file: Path, device: str | None) -> bool:
        """배치 처리 함수 (``process_batch``). 실패하면 예외를 그대로 올린다."""
        self.run_one(file, device)
        return True

    def run_one(self, file: Path, device: str | None = None) -> None:
        """파일 하나를 처리한다 (챕터 분할이면 챕터별 파일)."""
        pipeline = self._pipeline_for(device)
        if self.reset_checkpoint:
            _remove_checkpoint(file)
        if self.split_by_chapters:
            _run_transcribe_by_chapters(
                pipeline=pipeline, input_path=str(file), options=self.options
            )
        else:
            _run_transcribe(pipeline=pipeline, input_path=str(file), options=self.options)

    def close(self) -> None:
        """붙잡아 둔 STT 모델을 내린다."""
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None

    def _pipeline_for(self, device: str | None) -> TranscriptionPipeline:
        if self._pipeline is None:
            setup_logging(level=self.log_level)
            config = self.config
            if device is not None and device != config.stt.device:
                config = dataclasses.replace(
                    config, stt=dataclasses.replace(config.stt, device=device)
                )
            self._pipeline = TranscriptionPipeline(config, keep_engine=True)
        return self._pipeline


def _remove_checkpoint(file: Path) -> None:
    from src.checkpoint import get_checkpoint_path
    from src.stage_cache import StageCache, get_work_dir

    checkpoint = get_checkpoint_path(file)
    if checkpoint.exists():
        checkpoint.unlink()
        log.info("체크포인트 삭제 (--no-resume)", path=str(checkpoint))
    StageCache(get_work_dir(file)).clear()


def _run_batch(
    job: TranscribeJob,
    files: list[Path],
    workers: int,
    workers_device: str | None,
    failed_list: str | None,
) -> None:
    """여러 파일을 처리한다. 실패한 파일은 건너뛰고, 하나라도 실패하면 종료 코드 1."""
    from src.batch_processor import process_batch

    devices = _batch_devices(job.config, workers, workers_device)
    click.echo(f"배치 처리: 파일 {len(files)}개, worker {min(workers, len(files))}")
    try:
        succeeded, failed_count, failed = process_batch(
            files, job, workers=workers, failed_list_path=failed_list, devices=devices
        )
    finally:
        job.close()
    click.echo(f"배치 완료: 성공 {succeeded}, 실패 {failed_count}")
    for path in failed:
        click.echo(f"  실패: {path}")
    if failed_list and failed:
        click.echo(f"  실패 목록: {failed_list}")
    if failed_count:
        raise click.exceptions.Exit(1)


def _run_transcribe(
    pipeline: TranscriptionPipeline,
    input_path: str,
    options: PipelineOptions,
) -> None:
    """전체 파이프라인을 실행한다."""
    outcome = pipeline.run(input_path, options)
    for path in outcome.outputs:
        click.echo(f"완료: {path}")


# ──────────────────────────────────────────────
# diarize 서브커맨드
# ──────────────────────────────────────────────


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
# 입력/출력 옵션
@click.option("--output-dir", default=None, help="출력 디렉토리 (기본: ./output)")
@click.option(
    "--output-format",
    "output_format",
    default=None,
    type=click.Choice(["srt", "ass", "vtt", "json"]),
    help="출력 포맷 (기본: ass)",
)
@click.option("--output-name", default=None, help="출력 파일명 직접 지정")
@click.option("--keep-audio", is_flag=True, default=False, help="추출 오디오 보존")
@click.option("--audio-track", type=int, default=None, help="오디오 트랙 번호 (0-based)")
@click.option("--start", default=None, help="처리 시작 시간 (HH:MM:SS)")
@click.option("--end", default=None, help="처리 종료 시간 (HH:MM:SS)")
# STT 옵션
@click.option("--model", default=None, help=f"STT 엔진 ({_ENGINE_CHOICES_HELP})")
@click.option("--model-size", default=None, help="모델 크기 (large-v3 등)")
@click.option("--source-lang", default=None, help="음성 언어 (auto / en / ko / ja / zh)")
@click.option(
    "--device", default=None, type=click.Choice(["auto", "cuda", "cpu", "mps"]), help="디바이스"
)
# 화자 분리 옵션
@click.option("--num-speakers", default=None, help="화자 수 (auto 또는 정수)")
@click.option("--hf-token", default=None, help="HuggingFace 토큰")
@click.option(
    "--speaker-map",
    "speaker_map_str",
    default=None,
    help="화자 이름 매핑 (SPEAKER_00:John,SPEAKER_01:Jane)",
)
@click.option(
    "--speaker-profile", default=None, type=click.Path(exists=True), help="화자 프로파일 JSON 파일"
)
@click.option(
    "--speaker-embeddings",
    default=None,
    type=click.Path(exists=True),
    help="재사용할 화자 임베딩 파일",
)
@click.option("--export-embeddings", is_flag=True, default=False, help="화자 임베딩 내보내기")
@click.option("--no-speaker-split", is_flag=True, default=False, help="화자 전환 분할 비활성화")
# LLM 옵션 (llm-assisted 세그멘테이션용)
@click.option(
    "--llm-provider",
    default=None,
    type=click.Choice(LLM_PROVIDERS),
    help="LLM 프로바이더",
)
@click.option("--llm-model", default=None, help="LLM 모델명")
@click.option("--llm-base-url", default=None, help="LLM 커스텀 엔드포인트 URL")
# 기타
@click.option("--dry-run", is_flag=True, default=False, help="실제 처리 없이 계획 출력")
@click.pass_context
def diarize(
    ctx: click.Context,
    input_path: str,
    output_dir: str | None,
    output_format: str | None,
    output_name: str | None,
    keep_audio: bool,
    audio_track: int | None,
    start: str | None,
    end: str | None,
    model: str | None,
    model_size: str | None,
    source_lang: str | None,
    device: str | None,
    num_speakers: str | None,
    hf_token: str | None,
    speaker_map_str: str | None,
    speaker_profile: str | None,
    speaker_embeddings: str | None,
    export_embeddings: bool,
    no_speaker_split: bool,
    llm_provider: str | None,
    llm_model: str | None,
    llm_base_url: str | None,
    dry_run: bool,
) -> None:
    """화자 분리 포함 자막 생성."""
    config_path = ctx.obj["config_path"]

    # CLI 오버라이드 수집
    cli_overrides = _collect_stt_overrides(
        model=model,
        model_size=model_size,
        source_lang=source_lang,
        device=device,
    )
    cli_overrides["diarization.enabled"] = True
    if num_speakers is not None:
        cli_overrides["diarization.num_speakers"] = num_speakers
    if hf_token is not None:
        cli_overrides["diarization.hf_token"] = hf_token
    if no_speaker_split:
        cli_overrides["segmentation.speaker_split"] = False
    if output_dir is not None:
        cli_overrides["output.dir"] = output_dir
    if output_format is not None:
        cli_overrides["output.format"] = output_format
    else:
        cli_overrides["output.format"] = "ass"  # diarize 기본 포맷
    _collect_llm_overrides(cli_overrides, llm_provider, llm_model, llm_base_url)

    config = AppConfig.load(config_path=config_path, cli_overrides=cli_overrides)

    if dry_run:
        _print_diarize_dry_run(
            config,
            input_path,
            output_name,
            speaker_map_str,
            speaker_profile,
            export_embeddings,
        )
        return

    _run_diarize(
        config=config,
        input_path=input_path,
        output_name=output_name,
        keep_audio=keep_audio,
        audio_track=audio_track,
        start=start,
        end=end,
        speaker_map_str=speaker_map_str,
        speaker_profile=speaker_profile,
        speaker_embeddings_path=speaker_embeddings,
        do_export_embeddings=export_embeddings,
    )


def _print_diarize_dry_run(
    config: AppConfig,
    input_path: str,
    output_name: str | None,
    speaker_map_str: str | None,
    speaker_profile: str | None,
    export_embeddings: bool,
) -> None:
    """diarize dry-run 출력."""
    click.echo("=== Dry Run (diarize) ===")
    click.echo()
    click.echo(f"입력:        {input_path}")
    click.echo(f"출력 디렉토리: {config.output.dir}")
    click.echo(f"출력 포맷:    {config.output.format}")
    if output_name:
        click.echo(f"출력 파일명:  {output_name}")
    click.echo()
    click.echo(f"STT 엔진:    {config.stt.model}")
    click.echo(f"모델 크기:    {config.stt.model_size}")
    click.echo(f"디바이스:     {config.stt.device}")
    click.echo(f"언어:         {config.stt.source_lang}")
    click.echo()
    click.echo("화자 분리:    활성화")
    click.echo(f"화자 수:      {config.diarization.num_speakers}")
    click.echo(f"화자 분할:    {'비활성화' if not config.segmentation.speaker_split else '활성화'}")
    if speaker_map_str:
        click.echo(f"화자 매핑:    {speaker_map_str}")
    if speaker_profile:
        click.echo(f"화자 프로파일: {speaker_profile}")
    if export_embeddings:
        click.echo("임베딩 내보내기: 활성화")
    click.echo()
    click.echo("(--dry-run 모드로 실행되었습니다.)")


def _run_diarize(
    config: AppConfig,
    input_path: str,
    output_name: str | None,
    keep_audio: bool,
    audio_track: int | None,
    start: str | None,
    end: str | None,
    speaker_map_str: str | None,
    speaker_profile: str | None,
    speaker_embeddings_path: str | None,
    do_export_embeddings: bool,
) -> None:
    """화자 분리를 켠 전체 파이프라인을 실행한다."""
    options = PipelineOptions(
        output_name=output_name,
        keep_audio=keep_audio,
        audio_track=audio_track,
        start=start,
        end=end,
        speaker_map_str=speaker_map_str,
        speaker_profile=speaker_profile,
        speaker_embeddings=speaker_embeddings_path,
        export_embeddings=do_export_embeddings,
    )
    with TranscriptionPipeline(config) as pipeline:
        outcome = pipeline.run(input_path, options)
    for path in outcome.outputs:
        click.echo(f"완료: {path} ({outcome.num_speakers}명 화자)")


# ──────────────────────────────────────────────
# serve 서브커맨드 (웹 화면)
# ──────────────────────────────────────────────


@cli.command()
@click.option(
    "--data",
    "data_dir",
    default="./data",
    type=click.Path(file_okay=False),
    help="데이터 폴더 (작업·업로드·로그·계정)",
)
@click.option("--host", default=None, help="바인딩 주소 (기본: 설정 파일 web.host)")
@click.option("--port", type=int, default=None, help="포트 (기본: 설정 파일 web.port)")
@click.option(
    "--static-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="화면 빌드 폴더 (기본: 설정 파일 web.static_dir)",
)
@click.pass_context
def serve(
    ctx: click.Context,
    data_dir: str,
    host: str | None,
    port: int | None,
    static_dir: str | None,
) -> None:
    """웹 화면 서버를 실행한다 (작업은 한 번에 하나씩 순서대로 처리한다).

    설정 파일은 --config 로 준다. 없으면 데이터 폴더의 config.yaml 을 쓰고, 그 파일도 없으면 만든다.
    """
    from src.web.server import serve as run_server

    config_path = ctx.obj["config_path"]
    path = Path(config_path) if config_path else Path(data_dir) / "config.yaml"
    run_server(
        config_path=path,
        data_dir=Path(data_dir),
        host=host,
        port=port,
        static_dir=Path(static_dir) if static_dir else None,
    )


if __name__ == "__main__":
    cli()
