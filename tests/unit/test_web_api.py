"""웹 API·작업 워커 테스트 (FastAPI TestClient, 가짜 작업 프로세스)."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from src.web.jobs import STATUS_DONE
from src.web.server import build_context, create_app
from src.web.settings_io import ensure_config_file
from tests.unit import web_fake_targets

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from src.web.context import WebContext

PASSWORD = "password123"
WAIT_S = 30.0


def _context(tmp_path: Path, **kwargs: Any) -> WebContext:
    config = tmp_path / "config.yaml"
    ensure_config_file(config)
    context = build_context(
        config_path=config,
        data_dir=tmp_path / "data",
        session_hours=1,
        start_probe=False,
        **kwargs,
    )
    return context


@pytest.fixture()
def context(tmp_path: Path) -> Iterator[WebContext]:
    ctx = _context(tmp_path, start_worker=False)
    yield ctx
    ctx.store.close()


@pytest.fixture()
def client(context: WebContext) -> TestClient:
    app = create_app(context, static_dir=None)
    test_client = TestClient(app)
    response = test_client.post(
        "/api/session/setup", json={"username": "admin", "password": PASSWORD}
    )
    assert response.status_code == 200
    return test_client


def _upload(
    client: TestClient, name: str = "talk.wav", data: bytes = b"RIFF0000"
) -> dict[str, Any]:
    response = client.post(
        "/api/uploads",
        params={"name": name},
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def _create_job(client: TestClient, **option_updates: Any) -> dict[str, Any]:
    upload = _upload(client)
    options = client.get("/api/options").json()["defaults"]
    options.update(option_updates)
    response = client.post("/api/jobs", json={"upload_id": upload["id"], "options": options})
    assert response.status_code == 200, response.text
    return dict(response.json())


def _wait(predicate: Callable[[], bool], timeout: float = WAIT_S) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("시간 안에 조건을 만족하지 못함")


class TestSession:
    """처음 설정·로그인·로그아웃."""

    def test_setup_login_logout(self, context: WebContext) -> None:
        app = create_app(context, static_dir=None)
        client = TestClient(app)
        assert client.get("/api/session").json()["setup_required"] is True
        assert client.get("/api/status").status_code == 401
        assert (
            client.post("/api/session", json={"username": "a", "password": "b"}).status_code == 409
        )

        assert (
            client.post(
                "/api/session/setup", json={"username": "admin", "password": PASSWORD}
            ).status_code
            == 200
        )
        assert client.get("/api/status").status_code == 200
        assert (
            client.post(
                "/api/session/setup", json={"username": "x", "password": PASSWORD}
            ).status_code
            == 409
        )

        client.delete("/api/session", headers={"Content-Type": "application/json"})
        assert client.get("/api/status").status_code == 401
        assert (
            client.post(
                "/api/session", json={"username": "admin", "password": PASSWORD}
            ).status_code
            == 200
        )
        assert client.get("/api/session").json()["authenticated"] is True

    def test_wrong_password(self, client: TestClient, context: WebContext) -> None:
        object.__setattr__(context, "failed_login_delay_s", 0.0)
        client.delete("/api/session", headers={"Content-Type": "application/json"})
        response = client.post("/api/session", json={"username": "admin", "password": "nope-nope"})
        assert response.status_code == 401
        assert response.json()["detail"] == "invalid_credentials"

    def test_write_requests_need_json(self, client: TestClient) -> None:
        response = client.post(
            "/api/jobs/1/cancel", content=b"x", headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 415
        response = client.post("/api/uploads", params={"name": "a.wav"}, json={})
        assert response.status_code == 415
        assert response.headers["X-Frame-Options"] == "DENY"


class TestUploadsAndJobs:
    """업로드 검증과 작업 조작 (워커 없이)."""

    def test_upload_validation(self, client: TestClient, context: WebContext) -> None:
        headers = {"Content-Type": "application/octet-stream"}
        assert (
            client.post(
                "/api/uploads", params={"name": "a.txt"}, content=b"x", headers=headers
            ).json()["detail"]
            == "unsupported_format"
        )
        assert (
            client.post(
                "/api/uploads", params={"name": ".wav"}, content=b"x", headers=headers
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/uploads", params={"name": "a.wav"}, content=b"", headers=headers
            ).json()["detail"]
            == "empty_file"
        )
        uploaded = _upload(client, name="../../etc/passwd.wav")
        assert uploaded["name"] == "passwd.wav"

        config = context.config_path.read_text(encoding="utf-8").replace(
            "max_upload_mb: 4096", "max_upload_mb: 1"
        )
        context.config_path.write_text(config, encoding="utf-8")
        big = b"0" * (1024 * 1024 + 1)
        response = client.post(
            "/api/uploads", params={"name": "big.wav"}, content=big, headers=headers
        )
        assert response.status_code == 413
        leftovers = [
            p.name for p in (context.data_dir / "uploads").iterdir() if p.name.endswith(".part")
        ]
        assert leftovers == []

    def test_job_lifecycle_without_worker(self, client: TestClient, context: WebContext) -> None:
        job = _create_job(client, formats=["srt", "vtt"], engine="faster-whisper")
        assert job["status"] == "queued" and job["options"]["formats"] == ["srt", "vtt"]
        assert client.get("/api/jobs", params={"status": "queued"}).json()[0]["id"] == job["id"]
        assert client.get("/api/jobs", params={"status": "nope"}).status_code == 400

        cancelled = client.post(f"/api/jobs/{job['id']}/cancel", json={}).json()
        assert cancelled["status"] == "cancelled" and cancelled["retryable"] is True
        assert client.post(f"/api/jobs/{job['id']}/cancel", json={}).status_code == 409
        assert client.post(f"/api/jobs/{job['id']}/retry", json={}).json()["status"] == "queued"
        assert client.post("/api/jobs/999/retry", json={}).status_code == 404

        assert client.request("DELETE", f"/api/jobs/{job['id']}", json={}).json() == {"count": 1}
        assert client.get(f"/api/jobs/{job['id']}").status_code == 404

    def test_invalid_job_requests(self, client: TestClient) -> None:
        options = client.get("/api/options").json()["defaults"]
        assert (
            client.post("/api/jobs", json={"upload_id": "x" * 32, "options": options}).status_code
            == 400
        )
        assert (
            client.post("/api/jobs", json={"upload_id": "a" * 32, "options": options}).status_code
            == 404
        )
        upload = _upload(client)
        bad = {**options, "engine": "nope"}
        assert (
            client.post("/api/jobs", json={"upload_id": upload["id"], "options": bad}).status_code
            == 422
        )

    def test_download_and_preview(self, client: TestClient, context: WebContext) -> None:
        job = _create_job(client)
        out = context.store.output_dir(job["id"])
        out.mkdir(parents=True)
        (out / "talk.en.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n안녕\n", encoding="utf-8"
        )
        context.store.claim_next()
        context.store.finish(
            job["id"], STATUS_DONE, outputs=["talk.en.srt"], result={"segments": 1}
        )

        download = client.get(f"/api/jobs/{job['id']}/files/talk.en.srt")
        assert download.status_code == 200
        assert "attachment" in download.headers["content-disposition"]
        assert "안녕" in download.text
        preview = client.get(f"/api/jobs/{job['id']}/preview/talk.en.srt").json()
        assert preview["truncated"] is False and "안녕" in preview["text"]
        assert client.get(f"/api/jobs/{job['id']}/files/..%2Finput%2Ftalk.wav").status_code == 404

        assert client.post("/api/jobs/clear-finished", json={}).json() == {"count": 1}
        assert client.get("/api/jobs").json() == []

    def test_review_lists_and_edits_subtitles(
        self, client: TestClient, context: WebContext
    ) -> None:
        job = _create_job(client)
        out = context.store.output_dir(job["id"])
        out.mkdir(parents=True)
        srt = (
            "1\n00:00:00,000 --> 00:00:01,000\n안녕\n\n"
            "2\n00:00:02,000 --> 00:00:03,500\n기회가 고장났다\n"
        )
        (out / "talk.ko.srt").write_text(srt, encoding="utf-8")
        report = {
            "llm_error": "",
            "issues": [{"start": 2.1, "end": 3.4, "type": "misrecognition", "note": "기회 → 기계"}],
        }
        (out / "talk.review.json").write_text(json.dumps(report), encoding="utf-8")
        context.store.claim_next()
        context.store.finish(
            job["id"],
            STATUS_DONE,
            outputs=["talk.ko.srt", "talk.review.json"],
            result={"segments": 2},
        )
        url = f"/api/jobs/{job['id']}/review"

        review = client.get(url).json()
        assert review["name"] == "talk.ko.srt" and review["subtitles"] == ["talk.ko.srt"]
        assert review["report"] == "talk.review.json" and review["edited"] is False
        assert [row["notes"] for row in review["rows"]] == [
            [],
            [{"type": "misrecognition", "note": "기회 → 기계"}],
        ]
        assert review["rows"][1]["start"] == pytest.approx(2.0)
        assert [row["reviewed"] for row in review["rows"]] == [False, False]

        saved = client.put(
            url, json={"name": "talk.ko.srt", "index": 2, "text": "기계가 고장났다 "}
        )
        assert saved.status_code == 200 and saved.json()["text"] == "기계가 고장났다"
        assert saved.json()["reviewed"] is True
        assert (out / "talk.ko.srt").read_text(encoding="utf-8") == srt.replace("기회", "기계")
        assert (out / "talk.ko.orig.srt").read_text(encoding="utf-8") == srt
        # 남긴 원본은 내려받을 수 있고 검수 대상에는 들어가지 않는다
        assert "talk.ko.orig.srt" in client.get(f"/api/jobs/{job['id']}").json()["outputs"]
        review = client.get(url).json()
        assert review["subtitles"] == ["talk.ko.srt"] and review["edited"] is True
        assert review["rows"][1]["text"] == "기계가 고장났다"
        assert [row["reviewed"] for row in review["rows"]] == [False, True]

        # 글자를 바꾸지 않고 저장해도 검수 완료다 (파일은 그대로)
        before = (out / "talk.ko.srt").read_text(encoding="utf-8")
        same = client.put(url, json={"name": "talk.ko.srt", "index": 1, "text": "안녕"})
        assert same.status_code == 200 and same.json()["reviewed"] is True
        assert (out / "talk.ko.srt").read_text(encoding="utf-8") == before
        assert [row["reviewed"] for row in client.get(url).json()["rows"]] == [True, True]
        # 검수 상태 파일은 출력이 아니다
        assert "review-state.json" not in client.get(f"/api/jobs/{job['id']}").json()["outputs"]

        def put(**changes: Any) -> Any:
            body = {"name": "talk.ko.srt", "index": 1, "text": "안녕하세요", **changes}
            return client.put(url, json=body)

        assert (put(index=9).status_code, put(index=9).json()["detail"]) == (404, "cue_not_found")
        assert (put(text="  ").status_code, put(text="  ").json()["detail"]) == (422, "text_empty")
        assert put(name="talk.review.json").status_code == 404
        assert put(name="talk.ko.orig.srt").status_code == 404
        assert put(name="../input/talk.wav").status_code == 404
        assert client.get(url, params={"name": "nope.srt"}).status_code == 404
        assert client.get("/api/jobs/999/review").status_code == 404

    def test_review_without_srt_or_report(self, client: TestClient, context: WebContext) -> None:
        job = _create_job(client)
        out = context.store.output_dir(job["id"])
        out.mkdir(parents=True)
        (out / "talk.ko.vtt").write_text("WEBVTT\n", encoding="utf-8")
        (out / "talk.ko.srt").write_text("1\nbroken\n안녕\n", encoding="utf-8")
        context.store.claim_next()
        context.store.finish(job["id"], STATUS_DONE, outputs=["talk.ko.vtt"], result={})
        url = f"/api/jobs/{job['id']}/review"

        response = client.get(url)
        assert (response.status_code, response.json()["detail"]) == (404, "subtitle_not_found")

        context.store.add_output(job["id"], "talk.ko.srt")
        response = client.get(url)
        assert (response.status_code, response.json()["detail"]) == (422, "srt_invalid")

        (out / "talk.ko.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n안녕\n", encoding="utf-8"
        )
        review = client.get(url).json()
        assert review["report"] is None and review["rows"][0]["notes"] == []
        assert review["rows"][0]["reviewed"] is False

    def test_status_and_options(self, client: TestClient) -> None:
        status = client.get("/api/status").json()
        assert status["worker_alive"] is False and status["current"] is None
        assert status["system_state"] == "checking"
        assert set(status["queue"]) == {"queued", "running", "done", "failed", "cancelled"}
        options = client.get("/api/options").json()
        assert "whisperx" in options["choices"]["engines"]
        assert ".mp4" in options["extensions"]
        assert options["choices"]["stages"][0] == "extract"


class TestSettingsAndLogs:
    """설정·로그·계정 API."""

    def test_settings_roundtrip(self, client: TestClient) -> None:
        state = client.get("/api/settings").json()
        assert state["writable"] is True and "hf_token" not in state["values"]["diarization"]
        values = state["values"]
        values["stt"]["compute_type"] = "int8"
        values["web"]["max_upload_mb"] = 10
        saved = client.put("/api/settings", json={"values": values})
        assert saved.status_code == 200
        assert saved.json()["values"]["stt"]["compute_type"] == "int8"
        assert client.get("/api/options").json()["max_upload_mb"] == 10

        values["stt"]["batch_size"] = 0
        rejected = client.put("/api/settings", json={"values": values})
        assert rejected.status_code == 400
        detail = rejected.json()["detail"]
        assert detail["code"] == "invalid_settings"
        assert detail["errors"][0]["loc"] == "stt.batch_size"

    def test_logs_and_account(self, client: TestClient, context: WebContext) -> None:
        context.log_file.parent.mkdir(parents=True, exist_ok=True)
        context.log_file.write_text(
            '{"event": "hello", "level": "info", "logger": "src.web"}\n', encoding="utf-8"
        )
        chunk = client.get("/api/logs", params={"level": "INFO"}).json()
        assert [entry["message"] for entry in chunk["entries"]] == ["hello"]
        assert client.get("/api/logs", params={"level": "TRACE"}).status_code == 422

        assert client.get("/api/account").json() == {"username": "admin"}
        wrong = client.put("/api/account", json={"current_password": "x", "username": "admin"})
        assert wrong.status_code == 403
        changed = client.put(
            "/api/account",
            json={"current_password": PASSWORD, "username": "root", "new_password": "password456"},
        )
        assert changed.json() == {"username": "root"}
        # 계정을 바꾸면 지금 세션도 끝난다
        assert client.get("/api/status").status_code == 401


class TestWorker:
    """작업 워커: 한 번에 하나씩, 자식 프로세스로 실행한다."""

    def _client(self, tmp_path: Path, target: Any) -> tuple[TestClient, WebContext]:
        context = _context(tmp_path, worker_target=target)
        client = TestClient(create_app(context, static_dir=None))
        credentials = {"username": "admin", "password": PASSWORD}
        if client.post("/api/session/setup", json=credentials).status_code == 409:
            # 같은 데이터 폴더로 다시 띄운 서버: 계정이 이미 있다
            assert client.post("/api/session", json=credentials).status_code == 200
        return client, context

    def test_runs_jobs_in_order_and_deletes_input(self, tmp_path: Path) -> None:
        client, context = self._client(tmp_path, web_fake_targets.fake_done)
        try:
            first = _create_job(client, formats=["srt", "vtt"])
            second = _create_job(client)
            _wait(
                lambda: all(
                    client.get(f"/api/jobs/{j['id']}").json()["status"] == "done"
                    for j in (first, second)
                )
            )
            done = client.get(f"/api/jobs/{first['id']}").json()
            assert done["outputs"] == ["talk.en.srt"]
            assert done["result"]["formats"] == ["srt", "vtt"]
            assert done["result"]["resume"] is False
            # 단계 결과는 작업 폴더 안에 둔다 (작업을 지우면 함께 지워진다)
            assert done["result"]["work_dir"] == str(context.store.job_dir(first["id"]) / "work")
            # 완료한 작업의 입력은 지운다 (설정 web.delete_input_after_done 기본값)
            assert done["input_deleted"] is True and done["retryable"] is False
            second_done = client.get(f"/api/jobs/{second['id']}").json()
            assert done["finished_at"] <= second_done["started_at"]
        finally:
            context.worker.stop()
            context.store.close()

    def test_failure_crash_and_retry_resume(self, tmp_path: Path) -> None:
        client, context = self._client(tmp_path, web_fake_targets.fake_fail)
        try:
            job = _create_job(client)
            _wait(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] == "failed")
            failed = client.get(f"/api/jobs/{job['id']}").json()
            assert failed["error"] == "boom" and failed["retryable"] is True
        finally:
            context.worker.stop()
            context.store.close()

        client, context = self._client(tmp_path, web_fake_targets.fake_crash)
        try:
            assert client.post(f"/api/jobs/{job['id']}/retry", json={}).status_code == 200
            _wait(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] == "failed")
            crashed = client.get(f"/api/jobs/{job['id']}").json()
            assert crashed["error"] == "process_crashed" and crashed["attempts"] == 2
        finally:
            context.worker.stop()
            context.store.close()

        client, context = self._client(tmp_path, web_fake_targets.fake_done)
        try:
            client.post(f"/api/jobs/{job['id']}/retry", json={})
            _wait(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] == "done")
            # 다시 시도한 작업은 체크포인트에서 재개한다
            assert client.get(f"/api/jobs/{job['id']}").json()["result"]["resume"] is True
        finally:
            context.worker.stop()
            context.store.close()

    def test_cancel_running_job(self, tmp_path: Path) -> None:
        client, context = self._client(tmp_path, web_fake_targets.fake_sleep)
        try:
            job = _create_job(client)
            _wait(lambda: client.get(f"/api/jobs/{job['id']}").json()["stage"] == "load_model")
            status = client.get("/api/status").json()
            assert status["current"]["id"] == job["id"]
            started = time.monotonic()
            cancelled = client.post(f"/api/jobs/{job['id']}/cancel", json={}).json()
            assert cancelled["status"] == "cancelled"
            assert time.monotonic() - started < web_fake_targets.SLEEP_S
            assert client.get("/api/status").json()["current"] is None
        finally:
            context.worker.stop()
            context.store.close()
