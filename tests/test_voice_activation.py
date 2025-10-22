import pytest

from src.bot import UnifiedMessageHandler


@pytest.fixture
def handler(monkeypatch):
    # Default activation phrase for tests
    monkeypatch.setattr("src.bot.VOICE_ACTIVATION_PHRASE", "hey assistant")
    return UnifiedMessageHandler()


def test_activation_phrase_detects_command(handler):
    should_chat, chat_text, cleaned = handler._parse_voice_transcription(" Hey Assistant, what time is it? ")
    assert should_chat is True
    assert chat_text == "what time is it?"
    assert cleaned == "Hey Assistant, what time is it?"


def test_activation_phrase_allows_empty_command(handler):
    should_chat, chat_text, cleaned = handler._parse_voice_transcription("hey assistant")
    assert should_chat is True
    assert chat_text == ""
    assert cleaned == "hey assistant"


def test_regular_transcription_without_activation(handler, monkeypatch):
    monkeypatch.setattr("src.bot.VOICE_ACTIVATION_PHRASE", "jarvis")
    should_chat, chat_text, cleaned = handler._parse_voice_transcription("Just leaving a memo for later.")
    assert should_chat is False
    assert chat_text == "Just leaving a memo for later."
    assert cleaned == "Just leaving a memo for later."
