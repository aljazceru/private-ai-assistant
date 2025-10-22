"""
Speech-to-Text client for Signal Bot.

Provides both local Sherpa-ONNX processing and external API fallbacks.
"""

import os
import asyncio
import logging
import time
import tempfile
import numpy as np
import subprocess
from typing import Optional, Dict, List, Union, Tuple
import shutil
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)

# Global cache for models to avoid reloading
_asr_engines = {}
_vad_engine = None


class STTClient:
    """Speech-to-Text client with support for local Sherpa-ONNX and external APIs"""

    def __init__(self,
                 models_root: str = "./models",
                 asr_model: str = "sensevoice",
                 asr_language: str = "zh",
                 provider: str = "cpu",
                 num_threads: int = 2,
                 whisper_urls: Optional[str] = None,
                 whisper_model: Optional[str] = None,
                 whisper_timeout_seconds: Optional[int] = 60,
                 whisper_transcribe_paths: Optional[List[str]] = None,
                 output_format: str = "text",
                 vad_filter: bool = True,
                 language: Optional[str] = None):
        """
        Initialize STT client with automatic fallback support

        Args:
            models_root: Root directory for sherpa-onnx models
            asr_model: ASR model name (sensevoice, zipformer-bilingual, etc.)
            asr_language: Language for ASR (zh, en, ja, ko, yue)
            provider: ONNX Runtime provider (cpu, cuda)
            num_threads: Number of threads for processing
            whisper_urls: Comma-separated list of Whisper API URLs (fallback)
            output_format: Output format for transcription
            vad_filter: Enable voice activity detection filter
            language: Language for external API transcription
        """
        self.models_root = models_root
        self.asr_model = asr_model
        self.asr_language = asr_language
        self.provider = provider
        self.num_threads = num_threads
        self.sample_rate = 16000
        self.whisper_urls = [url.strip() for url in whisper_urls.split(',')] if whisper_urls else []
        self.whisper_model = whisper_model
        self.whisper_timeout_seconds = whisper_timeout_seconds
        self.whisper_transcribe_paths = whisper_transcribe_paths
        self.output_format = output_format
        self.vad_filter = vad_filter
        self.language = language

        self.use_sherpa = False
        self.whisper_client = None

        # Try to initialize Sherpa-ONNX
        try:
            self._load_sherpa_engine()
            self.use_sherpa = True
            logger.info(f"Initialized STT client with Sherpa-ONNX: {asr_model}")
        except Exception as e:
            logger.warning(f"Sherpa-ONNX initialization failed: {e}")
            if self.whisper_urls:
                self._load_whisper_client()
                logger.info(f"Falling back to Whisper API with {len(self.whisper_urls)} instances")
            else:
                logger.error("No STT client available - both Sherpa-ONNX and Whisper failed")

    def _load_sherpa_engine(self):
        """Load and cache the Sherpa-ONNX ASR engine"""
        cache_key = f"{self.asr_model}_{self.asr_language}_{self.provider}"

        if cache_key in _asr_engines:
            logger.info(f"Using cached ASR engine for {self.asr_model}")
            return _asr_engines[cache_key]

        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError("sherpa-onnx package not available")

        start_time = time.time()
        model_path = self._get_model_path()

        if self.asr_model == "sensevoice":
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=os.path.join(model_path, 'model.onnx'),
                tokens=os.path.join(model_path, 'tokens.txt'),
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                use_itn=True,
                debug=0,
                language=self.asr_language,
            )
            self._load_vad_engine()

        elif self.asr_model == "zipformer-bilingual":
            encoder = os.path.join(model_path, "encoder-epoch-99-avg-1.onnx")
            decoder = os.path.join(model_path, "decoder-epoch-99-avg-1.onnx")
            joiner = os.path.join(model_path, "joiner-epoch-99-avg-1.onnx")
            tokens = os.path.join(model_path, "tokens.txt")

            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                provider=self.provider,
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=20,
            )

        elif self.asr_model in ["paraformer-trilingual", "paraformer-en", "fireredasr"]:
            if self.asr_model == "paraformer-trilingual":
                recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    paraformer=os.path.join(model_path, 'model.onnx'),
                    tokens=os.path.join(model_path, 'tokens.txt'),
                    num_threads=self.num_threads,
                    sample_rate=self.sample_rate,
                    use_itn=True,
                    debug=0,
                    provider=self.provider,
                )
            elif self.asr_model == "paraformer-en":
                recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    paraformer=os.path.join(model_path, 'model.onnx'),
                    tokens=os.path.join(model_path, 'tokens.txt'),
                    num_threads=self.num_threads,
                    sample_rate=self.sample_rate,
                    use_itn=True,
                    debug=0,
                    provider=self.provider,
                )
            elif self.asr_model == "fireredasr":
                encoder = os.path.join(model_path, "encoder.int8.onnx")
                decoder = os.path.join(model_path, "decoder.int8.onnx")
                tokens = os.path.join(model_path, "tokens.txt")

                recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
                    encoder=encoder,
                    decoder=decoder,
                    tokens=tokens,
                    debug=0,
                    provider=self.provider,
                )

            if self.asr_model in ["paraformer-trilingual", "paraformer-en", "fireredasr"]:
                self._load_vad_engine()

        else:
            raise ValueError(f"Unsupported ASR model: {self.asr_model}")

        elapsed_time = time.time() - start_time
        logger.info(f"Loaded ASR model {self.asr_model} in {elapsed_time:.2f}s")

        _asr_engines[cache_key] = recognizer
        return recognizer

    def _get_model_path(self) -> str:
        """Get the model directory path based on model name"""
        model_paths = {
            "sensevoice": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            "zipformer-bilingual": "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
            "paraformer-trilingual": "sherpa-onnx-paraformer-trilingual-zh-cantonese-en",
            "paraformer-en": "sherpa-onnx-paraformer-en-2024-03-09",
            "fireredasr": "sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16"
        }

        if self.asr_model not in model_paths:
            raise ValueError(f"Unsupported ASR model: {self.asr_model}")

        return os.path.join(self.models_root, model_paths[self.asr_model])

    def _load_vad_engine(self):
        """Load VAD engine for offline models"""
        global _vad_engine

        if _vad_engine is not None:
            return

        try:
            import sherpa_onnx
        except ImportError:
            logger.warning("sherpa-onnx not available for VAD")
            return

        try:
            vad_path = os.path.join(self.models_root, 'silero_vad')
            if not os.path.exists(vad_path):
                logger.warning(f"VAD model not found at {vad_path}")
                return

            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = os.path.join(vad_path, 'silero_vad.onnx')
            config.silero_vad.min_silence_duration = 0.25
            config.sample_rate = self.sample_rate
            config.provider = self.provider
            config.num_threads = self.num_threads

            _vad_engine = sherpa_onnx.VoiceActivityDetector(
                config,
                buffer_size_in_seconds=100
            )

            logger.info("Loaded VAD engine for offline ASR")

        except Exception as e:
            logger.warning(f"Failed to load VAD engine: {str(e)}")

    def _load_whisper_client(self):
        """Load Whisper API client as fallback"""
        try:
            from .api import WhisperClient
            self.whisper_client = WhisperClient(
                whisper_urls=','.join(self.whisper_urls),
                output_format=self.output_format,
                vad_filter=self.vad_filter,
                language=self.language,
                model=self.whisper_model,
                timeout_seconds=self.whisper_timeout_seconds,
                transcribe_paths=self.whisper_transcribe_paths,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Whisper client: {str(e)}")
            raise

    async def transcribe_audio(self, audio_data: bytes, file_extension: str = ".ogg") -> Optional[str]:
        """
        Transcribe audio data using available STT method

        Args:
            audio_data: Raw audio data in bytes
            file_extension: File extension for temporary audio file (default: .ogg)

        Returns:
            Transcribed text or None if transcription fails
        """
        if not audio_data:
            logger.warning("No audio data provided for transcription")
            return None

        if self.use_sherpa:
            result = await self._transcribe_with_sherpa(audio_data, file_extension)
            if result:
                return result

            logger.warning("Sherpa-ONNX returned no transcription, attempting Whisper fallback if available")

        if self.whisper_urls and not self.whisper_client:
            try:
                self._load_whisper_client()
            except Exception:
                logger.error("Failed to initialize Whisper fallback after Sherpa failure")

        if self.whisper_client:
            return await self.whisper_client.transcribe_audio(audio_data, file_extension)

        logger.error("No STT client available")
        return None
 
    async def _transcribe_with_sherpa(self, audio_data: bytes, file_extension: str = ".ogg") -> Optional[str]:
        """Transcribe audio using Sherpa-ONNX"""
        try:
            import sherpa_onnx
            import soundfile
            from scipy.signal import resample
        except ImportError as e:
            logger.error(f"Required packages not available: {e}")
            return None

        # Save to temp file and decode
        effective_extension = file_extension or ".ogg"
        if not effective_extension.startswith("."):
            effective_extension = f".{effective_extension}"
        with tempfile.NamedTemporaryFile(suffix=effective_extension, delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        cleanup_paths: List[str] = [temp_file_path]

        try:
            load_result = self._load_audio_with_fallback(soundfile, temp_file_path, cleanup_paths)
            if not load_result:
                return None

            audio, sample_rate = load_result

            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample to 16kHz if needed
            if sample_rate != self.sample_rate:
                num_samples = int(len(audio) * self.sample_rate / sample_rate)
                audio = resample(audio, num_samples)

            # Convert to float32
            audio = audio.astype(np.float32)

            # Get the ASR engine
            recognizer = _asr_engines[f"{self.asr_model}_{self.asr_language}_{self.provider}"]

            # Perform transcription
            if self.asr_model == "zipformer-bilingual":
                # Streaming model
                result = await self._transcribe_streaming(recognizer, audio)
            else:
                # Offline model with VAD
                result = await self._transcribe_offline(recognizer, audio)

            return result

        except Exception as e:
            logger.error(f"Failed to transcribe audio with Sherpa-ONNX: {e}")
            return None
        finally:
            self._cleanup_temp_files(cleanup_paths)

    def _load_audio_with_fallback(
        self,
        soundfile_module,
        temp_file_path: str,
        cleanup_paths: List[str],
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Load audio, retrying with ffmpeg conversion when libsoundfile cannot decode."""
        logger.debug(f"Attempting to read audio file: {temp_file_path}")
        try:
            audio, sample_rate = soundfile_module.read(temp_file_path, dtype="float32", always_2d=True)
            logger.debug(f"Successfully read audio: shape={audio.shape}, sample_rate={sample_rate}")
            return audio, sample_rate
        except Exception as primary_err:
            logger.warning(
                f"libsoundfile could not decode {temp_file_path}: {primary_err}. Trying ffmpeg fallback."
            )
            ffmpeg_result = self._decode_audio_with_ffmpeg(temp_file_path)
            if not ffmpeg_result:
                self._log_audio_debug(temp_file_path)
                return None
            return ffmpeg_result

    def _decode_audio_with_ffmpeg(self, input_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """Decode audio to mono float32 numpy array using ffmpeg."""
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg binary not found in PATH; cannot convert audio attachments.")
            return None

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception as exc:
            logger.error(f"ffmpeg invocation failed: {exc}")
            return None

        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="ignore").strip()
            logger.error(f"ffmpeg conversion failed (exit {result.returncode}): {stderr_text}")
            return None

        raw_audio = result.stdout
        if not raw_audio:
            logger.error("ffmpeg produced no audio data.")
            return None

        samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            logger.error("Decoded audio buffer is empty after ffmpeg conversion.")
            return None

        return samples, self.sample_rate

    def _cleanup_temp_files(self, paths: List[str]) -> None:
        """Remove temporary files produced during transcription."""
        for path in paths:
            if not path:
                continue
            self._safe_unlink(path)

    def _safe_unlink(self, path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError as exc:
            logger.debug(f"Failed to remove temp file {path}: {exc}")

    def _log_audio_debug(self, file_path: str) -> None:
        """Log file diagnostics to help debug decoding failures."""
        try:
            size = os.path.getsize(file_path)
            logger.error(f"Audio file '{file_path}' has size {size} bytes")
            with open(file_path, "rb") as f:
                header = f.read(16)
            logger.debug(f"First 16 bytes of '{file_path}': {header.hex()}")
        except Exception as exc:
            logger.debug(f"Unable to inspect audio file {file_path}: {exc}")

    async def _transcribe_streaming(self, recognizer, audio: np.ndarray) -> str:
        """Transcribe using streaming ASR model"""
        stream = recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio)

        # Process the entire audio at once
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        # Get final result
        result = recognizer.get_result(stream)

        if result and result.strip():
            logger.info(f"Streaming ASR result: {result}")
            return result.strip()
        else:
            logger.warning("No transcription result from streaming ASR")
            return None

    async def _transcribe_offline(self, recognizer, audio: np.ndarray) -> str:
        """Transcribe using offline ASR model with VAD"""
        global _vad_engine

        if _vad_engine is None:
            logger.warning("VAD engine not available, processing as single chunk")
            return await self._transcribe_single_chunk(recognizer, audio)

        # Use VAD to segment the audio
        _vad_engine.accept_waveform(audio)

        transcription_segments = []

        while not _vad_engine.empty():
            # Create stream for each VAD segment
            stream = recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, _vad_engine.front.samples)

            _vad_engine.pop()

            # Recognize the segment
            recognizer.decode_stream(stream)
            result = stream.result.text.strip()

            if result:
                transcription_segments.append(result)

        # Combine all segments
        if transcription_segments:
            full_transcription = " ".join(transcription_segments)
            logger.info(f"Offline ASR result ({len(transcription_segments)} segments): {full_transcription}")
            return full_transcription
        else:
            # Fallback to single chunk processing
            return await self._transcribe_single_chunk(recognizer, audio)

    async def _transcribe_single_chunk(self, recognizer, audio: np.ndarray) -> str:
        """Transcribe entire audio as single chunk"""
        stream = recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio)

        recognizer.decode_stream(stream)
        result = stream.result.text.strip()

        if result:
            logger.info(f"Single chunk ASR result: {result}")
            return result
        else:
            logger.warning("No transcription result from single chunk ASR")
            return None

    def get_supported_models(self) -> List[str]:
        """Get list of supported ASR models"""
        return [
            "sensevoice",
            "zipformer-bilingual",
            "paraformer-trilingual",
            "paraformer-en",
            "fireredasr"
        ]

    def get_model_info(self) -> Dict:
        """Get information about current model configuration"""
        return {
            "client_type": "sherpa-onnx" if self.use_sherpa else "whisper_api",
            "model": self.asr_model if self.use_sherpa else "whisper",
            "language": self.asr_language if self.use_sherpa else self.language,
            "provider": self.provider if self.use_sherpa else None,
            "sample_rate": self.sample_rate,
            "threads": self.num_threads,
            "whisper_urls": self.whisper_urls if not self.use_sherpa else []
        }


# Legacy class names for backward compatibility
SherpaSTTClient = STTClient
