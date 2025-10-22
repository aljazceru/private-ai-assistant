"""
Client modules for Signal Bot.

This package contains client implementations for:
- Speech-to-Text (STT) processing
- Text-to-Speech (TTS) synthesis
- External API integrations (fallbacks)
"""

from .stt import STTClient
from .tts import TTSClient

__all__ = ["STTClient", "TTSClient"]