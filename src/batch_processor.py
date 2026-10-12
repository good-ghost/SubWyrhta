"""다중 파일 배치 처리 (CLI ``transcribe`` 가 쓴다, REQUIREMENT 2.3·11.1).

디렉토리 재귀 탐색, glob 패턴, 파일 목록 입력을 지원하며,
multiprocessing(spawn) 기반 병렬 처리를 제공한다. GPU 는 worker 하나만 쓴다.
"""

from __future__ import annotations

import glob as glob_module
from collections.abc import Callable
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import structlog

from src.audio.extractor import SUPPORTED_EXTENSIONS
from src.progress import ProgressTracker

log = structlog.get_logger(__name__)


def collect_files(
    input_path: str | None,
    recursive: bool = False,
    file_list: str | None = None,
) -> list[Path]:
    """처리할 파일 목록을 수집한다.

    Args:
        input_path: 파일 경로, 디렉토리 경로, 또는 glob 패턴. None 이면 file_list 만 쓴다.
        recursive: 디렉토리 재귀 탐색 여부.
        file_list: 파일 목록 텍스트 파일 경로.

    Returns:
        처리할 파일 Path 목록 (중복 제거, 정렬).
    """
    files: set[Path] = set()

    # 1. 파일 목록 텍스트
    if file_list is not None:
        files.update(_read_file_list(file_list))

    if input_path is None:
        result = sorted(files)
        log.info("파일 수집 완료", count=len(result))
        return result

    path = Path(input_path)

    if path.is_file():
        # 2. 단일 파일
        if _is_supported(path):
            files.add(path.resolve())
    elif path.is_dir():
        # 3. 디렉토리
        files.update(_scan_directory(path, recursive))
    elif "*" in input_path or "?" in input_path:
        # 4. glob 패턴
        files.update(_expand_glob(input_path, recursive))

    result = sorted(files)
    log.info("파일 수집 완료", count=len(result))
    return result


def _read_file_list(file_list_path: str) -> list[Path]:
    """텍스트 파일에서 파일 목록을 읽는다."""
    path = Path(file_list_path)
    if not path.exists():
        log.warning("파일 목록 없음", path=str(path))
        return []

    files: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        if p.exists() and p.is_file() and _is_supported(p):
            files.append(p.resolve())

    return files


def _scan_directory(directory: Path, recursive: bool) -> list[Path]:
    """디렉토리에서 지원되는 파일을 탐색한다."""
    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for p in directory.glob(pattern):
        if p.is_file() and _is_supported(p):
            files.append(p.resolve())
    return files


def _expand_glob(pattern: str, recursive: bool) -> list[Path]:
    """glob 패턴을 확장한다."""
    files: list[Path] = []
    for match in glob_module.glob(pattern, recursive=recursive):
        p = Path(match)
        if p.is_file() and _is_supported(p):
            files.append(p.resolve())
    return files


def _is_supported(path: Path) -> bool:
    """지원되는 파일 확장자인지 확인한다."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


# ──────────────────────────────────────────────
# 배치 실행
# ──────────────────────────────────────────────


# 파일 하나를 처리한다: (파일, 이 worker 의 디바이스 또는 None=설정 값) → 성공 여부
ProcessFn = Callable[[Path, str | None], bool]

# 병렬 worker 프로세스마다 initializer 가 채운다: (처리 함수, 디바이스)
_WORKER_STATE: tuple[ProcessFn, str | None] | None = None


def process_batch(
    files: list[Path],
    process_fn: ProcessFn,
    workers: int = 1,
    failed_list_path: str | None = None,
    devices: list[str] | None = None,
) -> tuple[int, int, list[Path]]:
    """파일 목록을 배치 처리한다. 실패한 파일은 건너뛰고 나머지를 계속한다.

    병렬일 때 worker 프로세스(spawn)마다 처리 함수를 한 번 받아 두고 파일만 넘긴다. 처리 함수가
    모델 등을 붙잡아 두면 worker 마다 한 번만 불러온다.

    Args:
        files: 처리할 파일 목록.
        process_fn: 파일 하나 처리 함수 ``(파일, 디바이스) -> 성공``. 병렬이면 pickle 할 수 있어야 한다.
        workers: 병렬 worker 수. 1이면 이 프로세스에서 순차 처리.
        failed_list_path: 실패 파일 목록 저장 경로.
        devices: worker 별 디바이스 (``build_device_assignments`` 결과). None 이면 설정 값.

    Returns:
        (성공 수, 실패 수, 실패 파일 목록).
    """
    if not files:
        return 0, 0, []

    failed_files: list[Path] = []
    workers = min(workers, len(files))

    with ProgressTracker(total=len(files), desc="배치 처리") as progress:
        if workers <= 1:
            device = devices[0] if devices else None
            for f in files:
                success = _safe_process(f, process_fn, device)
                progress.update(f.name, success=success)
                if not success:
                    failed_files.append(f)
        else:
            ctx = get_context("spawn")
            device_queue = ctx.Queue()
            for index in range(workers):
                device_queue.put(get_device_for_worker(index, devices) if devices else None)
            with ctx.Pool(
                processes=workers, initializer=_init_worker, initargs=(process_fn, device_queue)
            ) as pool:
                for f, success in pool.imap_unordered(_run_in_worker, files):
                    progress.update(f.name, success=success)
                    if not success:
                        failed_files.append(f)

    # 실패 목록 저장
    if failed_list_path and failed_files:
        _save_failed_list(failed_files, failed_list_path)

    succeeded = len(files) - len(failed_files)

    log.info(
        "배치 처리 완료",
        total=len(files),
        succeeded=succeeded,
        failed=len(failed_files),
    )

    return succeeded, len(failed_files), failed_files


def _safe_process(file: Path, process_fn: ProcessFn, device: str | None) -> bool:
    """파일 하나를 처리한다. 예외는 기록하고 실패(False)로 돌린다 (배치 전체를 멈추지 않는다)."""
    try:
        return bool(process_fn(file, device))
    except Exception as e:
        log.error("파일 처리 실패", file=str(file), error=str(e))
        return False


def _init_worker(process_fn: ProcessFn, device_queue: Any) -> None:
    """worker 프로세스 시작 때 처리 함수와 이 worker 의 디바이스를 받아 둔다."""
    global _WORKER_STATE
    _WORKER_STATE = (process_fn, device_queue.get())


def _run_in_worker(file: Path) -> tuple[Path, bool]:
    """worker 에서 파일 하나를 처리한다."""
    if _WORKER_STATE is None:
        raise RuntimeError("배치 worker 가 초기화되지 않았습니다")
    process_fn, device = _WORKER_STATE
    return file, _safe_process(file, process_fn, device)


def _save_failed_list(failed: list[Path], output_path: str) -> None:
    """실패 파일 목록을 저장한다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(str(f) for f in failed)
    path.write_text(content, encoding="utf-8")
    log.info("실패 파일 목록 저장", path=str(path), count=len(failed))


# ──────────────────────────────────────────────
# GPU 자원 관리
# ──────────────────────────────────────────────


def build_device_assignments(
    workers: int,
    workers_device: str | None = None,
) -> list[str]:
    """각 worker에 할당할 디바이스 목록을 생성한다.

    규칙:
        - workers_device가 지정되면 모든 worker에 해당 디바이스 할당
        - 미지정 시: worker 0 → GPU(auto), worker 1+ → CPU

    Args:
        workers: worker 수.
        workers_device: 명시적 디바이스 ('cuda', 'cpu', 'auto' 등).
            None이면 자동 할당.

    Returns:
        worker 인덱스별 디바이스 문자열 리스트.
    """
    if workers <= 0:
        return []

    if workers_device is not None:
        # 명시적 할당: 모든 worker에 동일 디바이스
        return [workers_device] * workers

    # 자동 할당: GPU worker 1개 + 나머지 CPU
    from src.engines.base import resolve_device

    gpu_device = resolve_device("auto")

    devices: list[str] = []
    for i in range(workers):
        if i == 0 and gpu_device != "cpu":
            devices.append(gpu_device)
        else:
            devices.append("cpu")

    if gpu_device != "cpu":
        log.info(
            "GPU 자원 할당",
            gpu_worker=0,
            gpu_device=gpu_device,
            cpu_workers=workers - 1,
        )
    else:
        log.info("GPU 미감지, 모든 worker CPU 사용", workers=workers)

    return devices


def get_device_for_worker(
    worker_id: int,
    device_assignments: list[str],
) -> str:
    """worker ID에 할당된 디바이스를 반환한다.

    Args:
        worker_id: worker 인덱스 (0-based).
        device_assignments: build_device_assignments() 결과.

    Returns:
        디바이스 문자열 ('cuda', 'cpu' 등).
    """
    if worker_id < len(device_assignments):
        return device_assignments[worker_id]
    return "cpu"
