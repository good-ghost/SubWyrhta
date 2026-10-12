"""진행 상황 표시.

tqdm 기반 배치 처리 진행률을 표시한다.
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger(__name__)


class ProgressTracker:
    """배치 처리 진행 상황 추적기.

    tqdm이 설치되어 있으면 프로그레스 바를 표시하고,
    없으면 로그로 진행 상황을 출력한다.

    Attributes:
        total: 전체 항목 수.
        completed: 완료된 항목 수.
        failed: 실패한 항목 수.
    """

    def __init__(self, total: int, desc: str = "처리 중") -> None:
        self.total = total
        self.completed: int = 0
        self.failed: int = 0
        self._pbar: Any = None

        try:
            from tqdm import tqdm

            self._pbar = tqdm(total=total, desc=desc, unit="file")
        except ImportError:
            log.info("tqdm 미설치, 로그 기반 진행 표시", total=total)

    def update(self, filename: str, success: bool = True) -> None:
        """항목 완료를 기록한다."""
        if success:
            self.completed += 1
        else:
            self.failed += 1

        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(ok=self.completed, fail=self.failed)
        else:
            status = "완료" if success else "실패"
            log.info(
                f"[{self.completed + self.failed}/{self.total}] {status}: {filename}",
            )

    def close(self) -> None:
        """프로그레스 바를 닫는다."""
        if self._pbar is not None:
            self._pbar.close()

    def __enter__(self) -> ProgressTracker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
