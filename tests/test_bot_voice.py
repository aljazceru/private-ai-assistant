import asyncio
import sys
import types

import numpy as np
import pytest

from src.bot import UnifiedMessageHandler
import src.clients.stt as stt_module
from src.clients.stt import STTClient


class DummyContext:
    def __init__(self, message):
        self.message = message


@pytest.fixture
def handler():
    # Minimal handler; dependencies not needed for extension detection
    return UnifiedMessageHandler()


def test_determine_extension_from_manual_attachment(handler):
    message = types.SimpleNamespace(
        attachments_local_filenames=[],
        _manual_attachments=[
            {
                "id": "123",
                "contentType": "audio/aac",
                "filename": None,
            }
        ],
    )
    context = DummyContext(message)

    assert handler._determine_audio_extension(context) == ".aac"


def test_determine_extension_from_local_filename(handler, tmp_path):
    path = tmp_path / "VOICE.M4A"
    path.write_bytes(b"fake audio")

    message = types.SimpleNamespace(
        attachments_local_filenames=[str(path)],
        _manual_attachments=[],
    )
    context = DummyContext(message)

    assert handler._determine_audio_extension(context) == ".m4a"


def test_determine_extension_default(handler):
    message = types.SimpleNamespace(
        attachments_local_filenames=[],
        _manual_attachments=[],
        raw_message={},
    )
    context = DummyContext(message)

    assert handler._determine_audio_extension(context) == ".ogg"


class DummySherpaSTT(STTClient):
    def __init__(self):
        # Bypass parent initialization
        self.use_sherpa = True
        self.whisper_client = None
        self.last_extension = None

    async def _transcribe_with_sherpa(self, audio_data: bytes, file_extension: str = ".ogg"):
        self.last_extension = file_extension
        return "ok"


def test_stt_receives_extension():
    client = DummySherpaSTT()
    result = asyncio.run(client.transcribe_audio(b"123", ".aac"))
    assert result == "ok"
    assert client.last_extension == ".aac"


class DummyFallbackSherpa(STTClient):
    def __init__(self):
        # Minimal configuration to exercise Sherpa path without loading models
        self.sample_rate = 16000
        self.use_sherpa = True
        self.whisper_client = None
        self.asr_model = "dummy"
        self.asr_language = "en"
        self.provider = "cpu"
        self.ffmpeg_calls = 0
        self._cache_key = f"{self.asr_model}_{self.asr_language}_{self.provider}"
        stt_module._asr_engines[self._cache_key] = object()

    async def _transcribe_offline(self, recognizer, audio: np.ndarray) -> str:
        return f"processed-{len(audio)}"

    async def _transcribe_streaming(self, recognizer, audio: np.ndarray) -> str:
        return f"stream-{len(audio)}"


def test_stt_ffmpeg_fallback(monkeypatch):
    client = DummyFallbackSherpa()
    client._log_audio_debug = lambda path: None

    load_calls = {"count": 0}

    def fake_read(path, dtype=None, always_2d=True):
        load_calls["count"] += 1
        if str(path).endswith(".aac"):
            raise RuntimeError("unsupported format")
        # Return dummy mono audio at a higher sample rate to trigger resample
        audio = np.zeros((48000, 1), dtype=np.float32) if always_2d else np.zeros(48000, dtype=np.float32)
        return audio, 48000

    fake_soundfile = types.SimpleNamespace(read=fake_read)
    fake_resample = lambda audio, num_samples: np.zeros(num_samples, dtype=np.float32)
    fake_signal_module = types.SimpleNamespace(resample=fake_resample)
    fake_scipy = types.SimpleNamespace(signal=fake_signal_module)
    fake_sherpa = types.SimpleNamespace()

    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa)
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.signal", fake_signal_module)

    def fake_decode(self, input_path: str):
        self.ffmpeg_calls += 1
        return np.zeros(16000, dtype=np.float32), 16000

    client._decode_audio_with_ffmpeg = types.MethodType(fake_decode, client)

    try:
        result = asyncio.run(client.transcribe_audio(b"\x00\x01", ".aac"))
        assert result == "processed-16000"
        assert client.ffmpeg_calls == 1
        assert load_calls["count"] == 1
    finally:
        stt_module._asr_engines.pop(client._cache_key, None)
