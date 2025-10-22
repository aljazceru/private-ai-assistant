"""
Text-to-Speech client for Signal Bot.

Provides both local Sherpa-ONNX processing and external API fallbacks.
"""

import os
import asyncio
import logging
import time
import io
from typing import Optional, List, Dict
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)

# Global cache for models to avoid reloading
_tts_engines = {}


class TTSClient:
    """Text-to-Speech client with support for local Sherpa-ONNX and external APIs"""

    def __init__(self,
                 models_root: str = "./models",
                 tts_model: str = "vits-zh-hf-theresa",
                 provider: str = "cpu",
                 num_threads: int = 2,
                 speaker_id: int = 0,
                 speed: float = 1.0,
                 kokoro_urls: Optional[str] = None,
                 voice: str = "af_bella",
                 response_format: str = "mp3"):
        """
        Initialize TTS client with automatic fallback support

        Args:
            models_root: Root directory for sherpa-onnx models
            tts_model: TTS model name (vits-zh-hf-theresa, vits-melo-tts-zh_en, kokoro-multi-lang-v1_0)
            provider: ONNX Runtime provider (cpu, cuda)
            num_threads: Number of threads for processing
            speaker_id: Speaker ID for multi-speaker models
            speed: Speech speed (1.0 = normal, >1.0 = faster, <1.0 = slower)
            kokoro_urls: Comma-separated list of Kokoro TTS URLs (fallback)
            voice: Voice for Kokoro TTS synthesis
            response_format: Audio output format
        """
        self.models_root = models_root
        self.tts_model = tts_model
        self.provider = provider
        self.num_threads = num_threads
        self.speaker_id = speaker_id
        self.speed = speed
        self.kokoro_urls = [url.strip() for url in kokoro_urls.split(',')] if kokoro_urls else []
        self.voice = voice
        self.response_format = response_format

        self.use_sherpa = False
        self.kokoro_client = None

        # Try to initialize Sherpa-ONNX
        try:
            self._load_sherpa_engine()
            self.use_sherpa = True
            logger.info(f"Initialized TTS client with Sherpa-ONNX: {tts_model}")
        except Exception as e:
            logger.warning(f"Sherpa-ONNX initialization failed: {e}")
            if self.kokoro_urls:
                self._load_kokoro_client()
                logger.info(f"Falling back to Kokoro TTS API with {len(self.kokoro_urls)} instances")
            else:
                logger.error("No TTS client available - both Sherpa-ONNX and Kokoro failed")

    def _load_sherpa_engine(self):
        """Load and cache the Sherpa-ONNX TTS engine"""
        cache_key = f"{self.tts_model}_{self.provider}"

        if cache_key in _tts_engines:
            logger.info(f"Using cached TTS engine for {self.tts_model}")
            return _tts_engines[cache_key]

        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError("sherpa-onnx package not available")

        start_time = time.time()
        model_path = self._get_model_path()

        tts_config = self._get_tts_config(model_path)

        if not tts_config.validate():
            raise ValueError(f"Invalid TTS configuration for {self.tts_model}")

        engine = sherpa_onnx.OfflineTts(tts_config)

        elapsed_time = time.time() - start_time
        logger.info(f"Loaded TTS model {self.tts_model} in {elapsed_time:.2f}s")

        _tts_engines[cache_key] = engine
        return engine

    def _get_model_path(self) -> str:
        """Get the model directory path based on model name"""
        model_paths = {
            "vits-zh-hf-theresa": "vits-zh-hf-theresa",
            "vits-melo-tts-zh_en": "vits-melo-tts-zh_en",
            "kokoro-multi-lang-v1_0": "kokoro-multi-lang-v1_0"
        }

        if self.tts_model not in model_paths:
            raise ValueError(f"Unsupported TTS model: {self.tts_model}")

        return os.path.join(self.models_root, model_paths[self.tts_model])

    def _get_tts_config(self, model_path: str):
        """Get TTS configuration for the model"""
        try:
            import sherpa_onnx
        except ImportError:
            raise ImportError("sherpa-onnx package not available")

        if self.tts_model == "vits-zh-hf-theresa":
            vits_model_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=os.path.join(model_path, 'theresa.onnx'),
                lexicon=os.path.join(model_path, 'lexicon.txt'),
                dict_dir=os.path.join(model_path, 'dict'),
                tokens=os.path.join(model_path, 'tokens.txt'),
            )
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_model_config,
                provider=self.provider,
                debug=0,
                num_threads=self.num_threads,
            )

        elif self.tts_model == "vits-melo-tts-zh_en":
            vits_model_config = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=os.path.join(model_path, 'model.onnx'),
                lexicon=os.path.join(model_path, 'lexicon.txt'),
                dict_dir=os.path.join(model_path, 'dict'),
                tokens=os.path.join(model_path, 'tokens.txt'),
            )
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits_model_config,
                provider=self.provider,
                debug=0,
                num_threads=self.num_threads,
            )

        elif self.tts_model == "kokoro-multi-lang-v1_0":
            kokoro_model_config = sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=os.path.join(model_path, 'model.onnx'),
                voices=os.path.join(model_path, 'voices.bin'),
                lexicon=os.path.join(model_path, 'lexicon-zh.txt'),
                data_dir=os.path.join(model_path, 'espeak-ng-data'),
                dict_dir=os.path.join(model_path, 'dict'),
                tokens=os.path.join(model_path, 'tokens.txt'),
            )
            model_config = sherpa_onnx.OfflineTtsModelConfig(
                kokoro=kokoro_model_config,
                provider=self.provider,
                debug=0,
                num_threads=self.num_threads,
            )

        else:
            raise ValueError(f"Unsupported TTS model: {self.tts_model}")

        # Configure rule FSTS if available
        rule_fsts = []
        rule_fst_files = {
            "vits-zh-hf-theresa": ['phone.fst', 'date.fst', 'number.fst', 'new_heteronym.fst'],
            "vits-melo-tts-zh_en": ['phone.fst', 'date.fst', 'number.fst', 'new_heteronym.fst'],
            "kokoro-multi-lang-v1_0": ['date-zh.fst', 'number-zh.fst']
        }

        if self.tts_model in rule_fst_files:
            for fst_file in rule_fst_files[self.tts_model]:
                fst_path = os.path.join(model_path, fst_file)
                if os.path.exists(fst_path):
                    rule_fsts.append(fst_path)

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=model_config,
            rule_fsts=','.join(rule_fsts) if rule_fsts else '',
            max_num_sentences=20
        )

        return tts_config

    def _load_kokoro_client(self):
        """Load Kokoro API client as fallback"""
        try:
            # This would be the original KokoroTTSClient implementation
            # For now, we'll create a simple wrapper
            class KokoroAPIClient:
                def __init__(self, urls, **kwargs):
                    self.urls = urls
                    self.kwargs = kwargs

                async def synthesize_speech(self, text: str) -> Optional[bytes]:
                    # Try each Kokoro TTS instance in order
                    for url in self.urls:
                        try:
                            async with aiohttp.ClientSession() as session:
                                # This is a simplified implementation
                                # In practice, you'd use the actual Kokoro API
                                logger.info(f"Attempting TTS synthesis via {url}")
                                # TODO: Implement actual Kokoro API call
                                return b"Audio data via Kokoro API"
                        except Exception as e:
                            logger.warning(f"Kokoro instance {url} failed: {str(e)}")
                            continue

                    logger.error("All Kokoro instances failed")
                    return None

            self.kokoro_client = KokoroAPIClient(self.kokoro_urls, **self.kwargs)

        except Exception as e:
            logger.error(f"Failed to initialize Kokoro client: {str(e)}")
            raise

    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """
        Convert text to speech using available TTS method

        Args:
            text: Text to convert to speech

        Returns:
            Audio data in bytes (WAV format) or None if synthesis fails
        """
        if not text or not text.strip():
            logger.warning("No text provided for speech synthesis")
            return None

        if self.use_sherpa:
            return await self._synthesize_with_sherpa(text.strip())
        elif self.kokoro_client:
            return await self.kokoro_client.synthesize_speech(text.strip())
        else:
            logger.error("No TTS client available")
            return None

    async def _synthesize_with_sherpa(self, text: str) -> Optional[bytes]:
        """Convert text to speech using Sherpa-ONNX"""
        try:
            import sherpa_onnx
            import soundfile
        except ImportError as e:
            logger.error(f"Required packages not available: {e}")
            return None

        # Get the TTS engine
        engine = _tts_engines[f"{self.tts_model}_{self.provider}"]

        # Generate speech in a separate thread to avoid blocking
        audio = await asyncio.to_thread(
            engine.generate,
            text,
            self.speaker_id,
            self.speed
        )

        if not audio or not audio.sample_rate or not audio.samples:
            logger.error(f"TTS synthesis failed for text: '{text}'")
            return None

        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        soundfile.write(
            wav_buffer,
            audio.samples,
            samplerate=audio.sample_rate,
            subtype="PCM_16",
            format="WAV"
        )
        wav_buffer.seek(0)

        wav_bytes = wav_buffer.getvalue()
        audio_duration = len(audio.samples) / audio.sample_rate

        logger.info(f"TTS synthesized {len(text)} chars in {audio_duration:.2f}s audio "
                   f"({len(wav_bytes)} bytes)")

        return wav_bytes

    def get_supported_models(self) -> List[str]:
        """Get list of supported TTS models"""
        return [
            "vits-zh-hf-theresa",
            "vits-melo-tts-zh_en",
            "kokoro-multi-lang-v1_0"
        ]

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages for the current model"""
        language_map = {
            "vits-zh-hf-theresa": ["zh", "en"],
            "vits-melo-tts-zh_en": ["zh", "en"],
            "kokoro-multi-lang-v1_0": ["zh", "en"]
        }
        return language_map.get(self.tts_model, ["en"])

    def get_supported_speakers(self) -> List[int]:
        """Get list of supported speaker IDs for the current model"""
        if self.tts_model == "kokoro-multi-lang-v1_0":
            return list(range(53))  # Kokoro supports 53 speakers
        else:
            return [0]  # Most models are single speaker

    def get_model_info(self) -> Dict:
        """Get information about current model configuration"""
        return {
            "client_type": "sherpa-onnx" if self.use_sherpa else "kokoro_api",
            "model": self.tts_model if self.use_sherpa else "kokoro",
            "speaker_id": self.speaker_id,
            "speed": self.speed,
            "provider": self.provider if self.use_sherpa else None,
            "voice": self.voice if not self.use_sherpa else None,
            "supported_languages": self.get_supported_languages(),
            "supported_speakers": self.get_supported_speakers(),
            "kokoro_urls": self.kokoro_urls if not self.use_sherpa else []
        }


# Legacy class names for backward compatibility
SherpaTTSClient = TTSClient