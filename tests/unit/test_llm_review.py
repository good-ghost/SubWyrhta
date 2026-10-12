"""LLM 줄바꿈 수정 검증 테스트 (마지막 검토의 줄바꿈 항목이 쓴다, 2026-10-10)."""

from __future__ import annotations

import pytest

from src.postprocess.llm_review import is_acceptable_revision, tidy_revision

_TWO_LINES = "東京都立中央図書館\n地域資料閲覧室利用案内"
_FIXED = "東京都立中央図書館地域\n資料閲覧室利用案内"


class TestAcceptableRevision:
    def test_accepts_moved_line_break(self) -> None:
        assert is_acceptable_revision(_TWO_LINES, _FIXED, "ja", 16, 2)

    def test_utterance_separator_may_become_line_break(self) -> None:
        assert is_acceptable_revision(
            "初めてです　あ、はい\nそうです", "初めてです\nあ、はいそうです", "ja", 16, 2
        )

    @pytest.mark.parametrize(
        "proposed",
        [
            "東京都立中央図書館地域。\n資料閲覧室利用案内",  # 부호 더함
            "東京都立中央図書館地域\n資料閲覧室利用",  # 글자 지움
            "東京都立中央圖書館地域\n資料閲覧室利用案内",  # 글자 바꿈
            "東京都立中央図書館地域資料閲覧室利\n用案内",  # 17자 줄
            "東京都立\n中央図書館地域\n資料閲覧室利用案内",  # 세 줄
        ],
    )
    def test_rejects(self, proposed: str) -> None:
        assert not is_acceptable_revision(_TWO_LINES, proposed, "ja", 16, 2)

    def test_english_keeps_spaces(self) -> None:
        original = "If so school is out\nfor the summer"
        assert is_acceptable_revision(original, "If so school\nis out for the summer", "en", 42, 2)
        assert not is_acceptable_revision(
            original, "If so schoolis out\nfor the summer", "en", 42, 2
        )


class TestTidyRevision:
    @pytest.mark.parametrize(
        ("original", "proposed", "expected"),
        [
            ("きっとあれは", "きっとあれは。", "きっとあれは"),
            ("以上である!気をつけ!", "以上である! 気をつけ!", "以上である!気をつけ!"),
            ("できました。", "できました。", "できました。"),
        ],
    )
    def test_tidy(self, original: str, proposed: str, expected: str) -> None:
        assert tidy_revision(original, proposed, "ja") == expected
