"""Tests for i18n module."""

import pytest
from src.i18n import (
    get_ui_text,
    get_welcome_message,
    get_language_instruction,
    UI_TEXT,
    WELCOME_MESSAGES,
    LANGUAGE_INSTRUCTIONS,
)


class TestGetUiText:
    """Tests for get_ui_text function."""

    def test_english_returns_all_keys(self):
        ui = get_ui_text("en")
        expected_keys = {"title", "input_placeholder", "send_button", "clear_button", "language_label", "chat_label"}
        assert set(ui.keys()) == expected_keys

    def test_chinese_returns_all_keys(self):
        ui = get_ui_text("zh")
        expected_keys = {"title", "input_placeholder", "send_button", "clear_button", "language_label", "chat_label"}
        assert set(ui.keys()) == expected_keys

    def test_unknown_lang_falls_back_to_english(self):
        ui = get_ui_text("fr")
        assert ui == UI_TEXT["en"]

    def test_english_and_chinese_have_different_content(self):
        en = get_ui_text("en")
        zh = get_ui_text("zh")
        assert en["title"] != zh["title"]
        assert en["send_button"] != zh["send_button"]


class TestGetWelcomeMessage:
    """Tests for get_welcome_message function."""

    def test_english_welcome(self):
        msg = get_welcome_message("en")
        assert "Singapore" in msg
        assert "Housing" in msg or "housing" in msg

    def test_chinese_welcome(self):
        msg = get_welcome_message("zh")
        assert "新加坡" in msg

    def test_unknown_lang_falls_back_to_english(self):
        msg = get_welcome_message("ja")
        assert msg == WELCOME_MESSAGES["en"]


class TestGetLanguageInstruction:
    """Tests for get_language_instruction function."""

    def test_english_instruction(self):
        inst = get_language_instruction("en")
        assert "English" in inst

    def test_chinese_instruction(self):
        inst = get_language_instruction("zh")
        assert "中文" in inst

    def test_unknown_lang_falls_back_to_english(self):
        inst = get_language_instruction("de")
        assert inst == LANGUAGE_INSTRUCTIONS["en"]
