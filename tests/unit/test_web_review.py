"""자막 검수 테스트: SRT 읽기·쓰기, 검수 리포트 항목을 자막에 붙이기, 자막 글자 수정."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.web.review import (
    MAX_TEXT_LENGTH,
    Note,
    ReviewError,
    attach_notes,
    backup_name,
    clean_text,
    format_srt,
    is_subtitle,
    mark_reviewed,
    parse_srt,
    read_report,
    read_reviewed,
    report_notes,
    save_text,
)

if TYPE_CHECKING:
    from pathlib import Path

_SRT = (
    "1\n00:00:00,885 --> 00:00:03,139\nというようなわけで、\n\n"
    "2\n00:00:03,179 --> 00:00:04,669\n駅の近くにある\n店はもう閉まっていた。\n\n"
    "3\n01:02:03,004 --> 01:02:05,500\nわかった。\n"
)


class TestSrt:
    def test_parse_reads_index_time_and_lines(self) -> None:
        cues = parse_srt(_SRT)
        assert [cue.index for cue in cues] == [1, 2, 3]
        assert cues[0].start == pytest.approx(0.885) and cues[0].end == pytest.approx(3.139)
        assert cues[1].text == "駅の近くにある\n店はもう閉まっていた。"
        assert cues[2].start == pytest.approx(3723.004)

    def test_format_is_the_file_as_read(self) -> None:
        assert format_srt(parse_srt(_SRT)) == _SRT

    def test_crlf_bom_and_missing_numbers(self) -> None:
        text = (
            "\ufeff00:00:01,000 --> 00:00:02,000\r\nHello\r\n\r\n"
            "00:00:03.000 --> 00:00:04.000\r\nBye\r\n"
        )
        cues = parse_srt(text)
        assert [(cue.index, cue.text) for cue in cues] == [(1, "Hello"), (2, "Bye")]
        assert cues[1].start == pytest.approx(3.0)

    def test_empty_file_has_no_cues(self) -> None:
        assert parse_srt("\n\n") == []

    @pytest.mark.parametrize(
        "text",
        [
            "1\nnot a timing line\nHello\n",
            "x\n00:00:01,000 --> 00:00:02,000\nHello\n",
            "1\n00:00:01,000 --> 00:00:02,000\nA\n\n1\n00:00:03,000 --> 00:00:04,000\nB\n",
        ],
    )
    def test_invalid_blocks_are_rejected(self, text: str) -> None:
        with pytest.raises(ReviewError, match="srt_invalid"):
            parse_srt(text)

    def test_names(self) -> None:
        assert is_subtitle("talk.ja.srt") and not is_subtitle("talk.ja.orig.srt")
        assert not is_subtitle("talk.ja.vtt")
        assert backup_name("talk.ja.srt") == "talk.ja.orig.srt"


class TestCleanText:
    def test_trims_lines_and_drops_blank_lines(self) -> None:
        assert clean_text("  一行目 \r\n\r\n 二行目\n") == "一行目\n二行目"

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            (" \n ", "text_empty"),
            ("あ" * (MAX_TEXT_LENGTH + 1), "text_too_long"),
            ("00:00:01,000 --> 00:00:02,000", "text_invalid"),
        ],
    )
    def test_rejects(self, text: str, code: str) -> None:
        with pytest.raises(ReviewError, match=code):
            clean_text(text)


class TestNotes:
    def test_report_items_become_notes(self) -> None:
        report = {
            "issues": [
                {"start": 1.0, "end": 2.0, "type": "misrecognition", "note": "機会 → 機械 로 보임"},
                {"start": 3.0, "end": 4.0, "type": "weird", "note": "확인"},
                {"start": 5.0, "end": 6.0, "type": "other", "note": ""},
                "not an item",
            ],
            "suggested": [
                {"start": 7.0, "end": 8.0, "from": "テレビ", "to": "ラジオ", "reason": "문맥"}
            ],
            "corrected": [
                {"start": 9.0, "end": 9.5, "from": "以外な", "to": "意外な", "reason": ""}
            ],
        }
        assert [(note.type, note.note) for note in report_notes(report)] == [
            ("misrecognition", "機会 → 機械 로 보임"),
            ("other", "확인"),
            ("suggestion", "テレビ → ラジオ (문맥)"),
            ("corrected", "以外な → 意外な"),
        ]

    def test_old_report_without_corrections(self) -> None:
        assert report_notes({"issues": []}) == [] and report_notes({}) == []

    def test_attach_by_overlap_then_nearest_start(self) -> None:
        cues = parse_srt(_SRT)
        notes = [
            Note(3.0, 4.6, "punctuation", "겹침: 2번이 더 많이 겹친다"),
            Note(10.0, 11.0, "other", "겹치지 않음: 시작이 가까운 2번"),
            Note(3700.0, 3701.0, "other", "겹치지 않음: 시작이 가까운 3번"),
        ]
        attached = attach_notes(cues, notes)
        assert [note.note[:2] for note in attached[2]] == ["겹침", "겹치"]
        assert [note.type for note in attached[3]] == ["other"] and 1 not in attached
        assert attach_notes([], notes) == {}

    def test_read_report_tolerates_missing_and_broken_files(self, tmp_path: Path) -> None:
        assert read_report(None) == {}
        broken = tmp_path / "a.review.json"
        broken.write_text("{not json", encoding="utf-8")
        assert read_report(broken) == {}
        broken.write_text("[1]", encoding="utf-8")
        assert read_report(broken) == {}
        broken.write_text(json.dumps({"llm_error": "timed out"}), encoding="utf-8")
        assert read_report(broken) == {"llm_error": "timed out"}


class TestSaveText:
    def test_first_save_keeps_original_and_changes_only_text(self, tmp_path: Path) -> None:
        path = tmp_path / "talk.ja.srt"
        path.write_text(_SRT, encoding="utf-8")

        cue, backed_up = save_text(path, 2, "駅の近くにある\n\n店はもう開いていた。 ")

        assert backed_up and cue.text == "駅の近くにある\n店はもう開いていた。"
        assert (tmp_path / "talk.ja.orig.srt").read_text(encoding="utf-8") == _SRT
        assert path.read_text(encoding="utf-8") == _SRT.replace("閉まって", "開いて")
        assert not (tmp_path / "talk.ja.srt.tmp").exists()

    def test_second_save_does_not_overwrite_original(self, tmp_path: Path) -> None:
        path = tmp_path / "talk.ja.srt"
        path.write_text(_SRT, encoding="utf-8")
        save_text(path, 1, "一回目")
        _, backed_up = save_text(path, 1, "二回目")
        assert not backed_up
        assert (tmp_path / "talk.ja.orig.srt").read_text(encoding="utf-8") == _SRT
        assert parse_srt(path.read_text(encoding="utf-8"))[0].text == "二回目"

    def test_invalid_request_leaves_files_untouched(self, tmp_path: Path) -> None:
        path = tmp_path / "talk.ja.srt"
        path.write_text(_SRT, encoding="utf-8")
        with pytest.raises(ReviewError, match="cue_not_found"):
            save_text(path, 9, "없는 번호")
        with pytest.raises(ReviewError, match="text_empty"):
            save_text(path, 1, "  ")
        assert path.read_text(encoding="utf-8") == _SRT
        assert not (tmp_path / "talk.ja.orig.srt").exists()

    def test_unchanged_text_does_not_rewrite_or_back_up(self, tmp_path: Path) -> None:
        """확인만 하고 저장하면 파일을 그대로 둔다 (원본도 남기지 않는다)."""
        path = tmp_path / "talk.ja.srt"
        path.write_text(_SRT, encoding="utf-8")
        cue, backed_up = save_text(path, 3, " わかった。\n")
        assert (cue.text, backed_up) == ("わかった。", False)
        assert path.read_text(encoding="utf-8") == _SRT
        assert not (tmp_path / "talk.ja.orig.srt").exists()


class TestReviewedState:
    def test_marks_are_kept_per_subtitle_file(self, tmp_path: Path) -> None:
        state = tmp_path / "review-state.json"
        assert read_reviewed(state, "talk.ja.srt") == set()
        mark_reviewed(state, "talk.ja.srt", 3)
        mark_reviewed(state, "talk.ja.srt", 1)
        mark_reviewed(state, "talk.ja.srt", 3)
        mark_reviewed(state, "talk.en.srt", 2)
        assert read_reviewed(state, "talk.ja.srt") == {1, 3}
        assert read_reviewed(state, "talk.en.srt") == {2}
        assert not (tmp_path / "review-state.json.tmp").exists()

    @pytest.mark.parametrize("content", ["{broken", "[1, 2]", '{"reviewed": [1]}'])
    def test_broken_state_is_empty_and_can_be_rewritten(self, tmp_path: Path, content: str) -> None:
        state = tmp_path / "review-state.json"
        state.write_text(content, encoding="utf-8")
        assert read_reviewed(state, "talk.ja.srt") == set()
        mark_reviewed(state, "talk.ja.srt", 2)
        assert read_reviewed(state, "talk.ja.srt") == {2}
