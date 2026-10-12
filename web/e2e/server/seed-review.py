"""E2E 용: 완료한 작업에 LLM 검수 리포트(``*.review.json``)를 만들어 붙인다.

E2E 서버는 LLM 없이 돌아 검수 리포트가 나오지 않는다. 검수 화면의 메모·종류 필터를 보이려고
작업의 SRT 자막 시각에 맞춘 리포트를 만들고 출력 목록에 더한다 (서버와 같은 JobStore 를 쓴다).

사용 (저장소 루트에서): python web/e2e/server/seed-review.py <데이터 폴더> <작업 번호>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.web.jobs import JobStore
from src.web.review import REPORT_SUFFIX, SRT_SUFFIX, is_subtitle, parse_srt

# (리포트 키, 종류, 메모 또는 교정) — 자막 앞에서부터 하나씩 붙인다. 자막이 모자라면 돌아가며 붙인다
_ITEMS = (
    ("issues", "misrecognition", "'sub title' may be a misrecognition of 'subtitle'"),
    ("issues", "punctuation", "A full stop seems to be missing at the end"),
    ("issues", "other", "Check the speaker of this line"),
    ("suggested", "", ("demo", "Demo", "proper noun")),
    ("corrected", "", ("watch", "watching", "glossary")),
)


def main(data_dir: Path, job_id: int) -> None:
    """작업의 첫 SRT 에 맞춘 검수 리포트를 쓰고 출력 목록에 더한다."""
    store = JobStore(data_dir)
    try:
        job = store.get(job_id)
        name = next(output for output in job.outputs if is_subtitle(output))
        out = store.output_dir(job_id)
        cues = parse_srt((out / name).read_text(encoding="utf-8"))
        report: dict[str, object] = {"source_file": job.name, "provider": "e2e", "model": "seed"}
        lists: dict[str, list[dict[str, object]]] = {"issues": [], "suggested": [], "corrected": []}
        for at, (key, kind, body) in enumerate(_ITEMS):
            cue = cues[at % len(cues)]
            item: dict[str, object] = {"start": cue.start, "end": cue.end, "text": cue.text}
            if isinstance(body, str):
                item.update(type=kind, note=body)
            else:
                item.update({"from": body[0], "to": body[1], "reason": body[2]})
            lists[key].append(item)
        report.update(lists)
        report["llm_error"] = ""
        # 출력 이름의 언어 꼬리표(.en.srt)를 떼어 파이프라인의 리포트 이름(<이름>.review.json)과 맞춘다
        stem = name[: -len(SRT_SUFFIX)].rsplit(".", 1)[0]
        report_name = f"{stem}{REPORT_SUFFIX}"
        (out / report_name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        store.add_output(job_id, report_name)
    finally:
        store.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
