"""시스템 점검: 디바이스·설치 패키지·ffmpeg 를 확인해 JSON 으로 출력한다.

웹 서버는 이 모듈을 별도 프로세스(``python -m src.web.system_probe``)로 한 번 실행한다. torch 를
서버 프로세스에 import 하면 서버가 상주하는 동안 수백 MB 를 더 쓰기 때문이다.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# 엔진·기능별로 필요한 패키지 (화면의 설치 상태 표)
PACKAGES: dict[str, str] = {
    "whisperx": "whisperx",
    "faster_whisper": "faster_whisper",
    "nemo": "nemo",
    "silero_vad": "silero_vad",
    "pyannote": "pyannote.audio",
    "demucs": "demucs",
    "noisereduce": "noisereduce",
    "pysbd": "pysbd",
}
PROBE_TIMEOUT_S = 120.0
# `python -m src.web.system_probe` 가 src 패키지를 찾는 위치
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def collect() -> dict[str, Any]:
    """이 프로세스에서 시스템 정보를 모은다 (torch 를 import 한다)."""
    from src.engines.base import get_device_info

    return {
        "device": get_device_info(),
        "packages": {name: _installed(module) for name, module in PACKAGES.items()},
        "ffmpeg": shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None,
    }


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class SystemInfo:
    """백그라운드에서 한 번 채우는 시스템 정보. 채우기 전에는 state 가 checking."""

    state: str = "checking"
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self) -> None:
        """점검 프로세스를 백그라운드 스레드에서 실행한다."""
        threading.Thread(target=self._probe, name="system-probe", daemon=True).start()

    def snapshot(self) -> tuple[str, dict[str, Any], str | None]:
        """(상태, 정보, 오류)."""
        with self._lock:
            return self.state, dict(self.data), self.error

    def _probe(self) -> None:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "src.web.system_probe"],
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_S,
                check=True,
                cwd=_PROJECT_ROOT,
            )
            data = json.loads(completed.stdout)
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as e:
            log.warning("시스템 점검 실패", error=str(e))
            with self._lock:
                self.state, self.error = "failed", str(e)
            return
        with self._lock:
            self.state, self.data = "ready", data
        log.info("시스템 점검 완료", devices=data.get("device", {}).get("available_devices"))


if __name__ == "__main__":
    json.dump(collect(), sys.stdout)
