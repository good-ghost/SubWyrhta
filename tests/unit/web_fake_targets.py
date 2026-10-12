"""작업 워커 테스트용 자식 프로세스 진입 함수 (spawn 이 모듈 이름으로 다시 import 한다)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.web.runner import EVENT_DONE, EVENT_FAILED, EVENT_PROGRESS, JobPayload

# 취소 테스트가 기다리는 시간 상한 (그 전에 부모가 프로세스를 끝낸다)
SLEEP_S = 30.0


def fake_done(payload: JobPayload, events: Any) -> None:
    """진행 두 번 + 출력 파일 하나를 만들고 끝난다."""
    events.put((EVENT_PROGRESS, "transcribe", 1, 2))
    events.put((EVENT_PROGRESS, "transcribe", 2, 2))
    out = Path(payload.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(payload.input_path).stem}.en.srt"
    (out / name).write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    summary = {
        "language": "en",
        "segments": 1,
        "resume": payload.resume,
        "formats": payload.formats,
        "work_dir": payload.work_dir,
    }
    events.put((EVENT_DONE, [name], summary))


def fake_fail(payload: JobPayload, events: Any) -> None:
    """실패를 알린다."""
    events.put((EVENT_FAILED, "boom"))


def fake_crash(payload: JobPayload, events: Any) -> None:
    """결과를 보내지 않고 비정상 종료한다 (OOM kill 흉내)."""
    raise SystemExit(3)


def fake_sleep(payload: JobPayload, events: Any) -> None:
    """취소될 때까지 오래 돈다."""
    events.put((EVENT_PROGRESS, "load_model", 0, 0))
    time.sleep(SLEEP_S)
