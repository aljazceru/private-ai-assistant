"""
Signal Bot - Signal messenger bot with AI chat capabilities and local speech processing.

This package provides a complete Signal bot solution with:
- Local STT/TTS processing using Sherpa-ONNX
- PrivateMode.ai integration for AI responses
- Role-based permission system
- Voice message transcription and synthesis
"""

__version__ = "2.0.0"
__author__ = "Signal Bot Team"
__email__ = "team@signalbot.dev"

from .bot import main

__all__ = ["main"]