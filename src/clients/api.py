"""
API clients for Signal Bot.

Provides clients for PrivateMode.ai and external services.
"""

import json
import logging
import mimetypes
from typing import Optional, Union, List, Tuple, Dict
from io import BytesIO

import aiohttp
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not available - audio conversion disabled")

logger = logging.getLogger(__name__)


class PrivateModeClient:
    """Client for PrivateMode.ai API"""

    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Content-Type": "application/json"
        }

        # Add Authorization header if API key is provided
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def list_models(self) -> list:
        """List available models from PrivateMode.ai"""
        url = f"{self.base_url}/v1/models"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model['id'] for model in data.get('data', [])]
                    else:
                        logger.error(f"Failed to list models: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"Model listing failed: {str(e)}")
                return []

    async def verify_models_at_startup(self, preferred_model: str = None) -> Tuple[bool, List[str], Optional[str]]:
        """
        Verify that the AI endpoint is working and list available models.
        Returns a tuple of (success, available_models, error_message)
        """
        logger.info("Verifying AI endpoint connectivity and available models...")

        try:
            # Test basic connectivity with a timeout
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # First try standard OpenAI /v1/models endpoint
                models_url = f"{self.base_url}/v1/models"
                logger.info(f"Trying models endpoint: {models_url}")

                async with session.get(models_url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['id'] for model in data.get('data', [])]

                        if not models:
                            return False, [], "No models available from API"

                        logger.info(f"✅ AI endpoint verified - {len(models)} models available")
                        for model in models[:5]:  # Log first 5 models
                            logger.info(f"  - {model}")
                        if len(models) > 5:
                            logger.info(f"  ... and {len(models) - 5} more models")

                        # Check if preferred model is available
                        if preferred_model and preferred_model in models:
                            logger.info(f"✅ Preferred model '{preferred_model}' is available")
                        elif preferred_model:
                            logger.warning(f"⚠️ Preferred model '{preferred_model}' not found, will use first available model")

                        return True, models, None
                    elif response.status == 404:
                        # If /v1/models returns 404, try to use the configured model directly
                        logger.info(f"Models endpoint not found (404), trying to verify with configured model")
                        return await self._verify_with_chat_completion(session, preferred_model)
                    else:
                        error_text = await response.text()
                        error_msg = f"AI endpoint returned HTTP {response.status}: {error_text}"
                        logger.error(f"❌ {error_msg}")
                        return False, [], error_msg

        except aiohttp.ClientTimeout:
            error_msg = f"AI endpoint timeout after 10 seconds: {self.base_url}"
            logger.error(f"❌ {error_msg}")
            return False, [], error_msg
        except aiohttp.ClientConnectorError:
            error_msg = f"Cannot connect to AI endpoint: {self.base_url}"
            logger.error(f"❌ {error_msg}")
            return False, [], error_msg
        except Exception as e:
            error_msg = f"Failed to verify AI endpoint: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, [], error_msg

    async def _verify_with_chat_completion(self, session, preferred_model: str) -> Tuple[bool, List[str], Optional[str]]:
        """
        Verify endpoint works by trying a simple chat completion when /v1/models is not available.
        This is common for custom API wrappers that don't expose the models endpoint.
        """
        try:
            # Use the preferred model or a common default
            test_model = preferred_model or "gpt-3.5-turbo"

            # Create a minimal test request
            chat_url = f"{self.base_url}/v1/chat/completions"
            logger.info(f"Testing chat endpoint: {chat_url} with model: {test_model}")

            payload = {
                "model": test_model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }

            async with session.post(chat_url, json=payload, headers=self.headers) as response:
                if response.status == 200:
                    logger.info(f"✅ Chat endpoint verified with model: {test_model}")
                    # Return the model as if it came from a models list
                    return True, [test_model], None
                else:
                    error_text = await response.text()
                    error_msg = f"Chat endpoint test failed with HTTP {response.status}: {error_text}"
                    logger.error(f"❌ {error_msg}")
                    return False, [], error_msg

        except Exception as e:
            error_msg = f"Chat endpoint verification failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, [], error_msg

    async def chat_completion(self, messages: list, model: str = None) -> str:
        """Generate chat completion using PrivateMode.ai"""
        url = f"{self.base_url}/v1/chat/completions"

        # Use provided model or get the first available one
        if not model:
            available_models = await self.list_models()
            if available_models:
                model = available_models[0]
                logger.info(f"Using model: {model}")
            else:
                return "Sorry, no models are available at the moment."

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status} - {error_text}")
                        return f"Sorry, I encountered an error: {response.status}"
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                return f"Sorry, I couldn't process your request: {str(e)}"


class WhisperClient:
    """Client for Whisper ASR service with failover support"""

    def __init__(
        self,
        whisper_urls: str,
        output_format: str = "text",
        vad_filter: bool = True,
        language: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = 60,
        transcribe_paths: Optional[List[str]] = None,
    ):
        """
        Initialize Whisper client with multiple instance support

        Args:
            whisper_urls: Comma-separated list of Whisper ASR URLs
            output_format: Output format (text, json, vtt, srt, tsv)
            vad_filter: Enable voice activity detection filter
            language: Language for transcription (auto-detect if None)
            model: Optional Whisper model identifier when required by the API
            timeout_seconds: Request timeout for each Whisper call
        """
        self.urls = [url.strip() for url in whisper_urls.split(',') if url.strip()]
        self.output_format = output_format or "text"
        self.vad_filter = vad_filter
        self.language = language if language and language.lower() != "auto" else None
        self.model = model.strip() if model else None
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds) if timeout_seconds else None
        self.transcribe_paths = self._normalize_paths(transcribe_paths)
        logger.info(f"Initialized WhisperClient with {len(self.urls)} instances")

    def _convert_to_wav(self, audio_data: bytes, file_extension: str) -> bytes:
        """
        Convert audio data to WAV format for better Whisper compatibility

        Args:
            audio_data: Raw audio data in bytes
            file_extension: Original file extension (e.g., .aac, .m4a, .ogg)

        Returns:
            WAV audio data in bytes
        """
        if not PYDUB_AVAILABLE:
            logger.warning("pydub not available, using original audio format")
            return audio_data

        # Skip conversion for WAV files
        if file_extension.lower() in ['.wav', '.wave']:
            return audio_data

        try:
            # Create AudioSegment from bytes
            audio = AudioSegment.from_file(BytesIO(audio_data))

            # Export to WAV format
            wav_buffer = BytesIO()
            audio.export(wav_buffer, format='wav', parameters=['-ar', '16000'])  # 16kHz sample rate for better Whisper performance
            wav_data = wav_buffer.getvalue()
            wav_buffer.close()

            logger.info(f"Converted {file_extension} ({len(audio_data)} bytes) to WAV ({len(wav_data)} bytes)")
            return wav_data

        except Exception as e:
            logger.warning(f"Failed to convert {file_extension} to WAV: {e}")
            return audio_data

    async def transcribe_audio(self, audio_data: bytes, file_extension: str = ".ogg") -> Optional[str]:
        """
        Transcribe audio data with failover support

        Args:
            audio_data: Audio file data in bytes
            file_extension: File extension (e.g., .ogg, .m4a) to inform the API

        Returns:
            Transcribed text or None if all instances fail
        """
        if not audio_data:
            logger.warning("No audio data provided to WhisperClient.transcribe_audio")
            return None

        normalized_extension = file_extension if file_extension else ".ogg"
        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"

        # Convert audio to WAV format for better Whisper compatibility
        converted_audio = self._convert_to_wav(audio_data, normalized_extension)
        if converted_audio != audio_data:
            # Audio was converted, update extension
            normalized_extension = ".wav"
            logger.info("Using converted WAV audio for transcription")

        for url in self.urls:
            try:
                result = await self._transcribe_with_instance(url, converted_audio, normalized_extension)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Whisper instance {url} failed: {str(e)}")
                continue

        logger.error("All Whisper instances failed")
        return None

    async def _transcribe_with_instance(self, url: str, audio_data: bytes, file_extension: str) -> Optional[str]:
        """Transcribe audio using specific Whisper instance"""
        filename = f"audio{file_extension}"
        guessed_content_type, _ = mimetypes.guess_type(filename)
        content_type = guessed_content_type or "audio/ogg"

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            last_error: Optional[str] = None
            for path in self.transcribe_paths:
                transcribe_url = self._build_endpoint(url, path)
                use_asr_endpoint = path.strip("/").startswith("asr")
                data = self._build_form_data(audio_data, filename, content_type, use_asr_endpoint)
                params = self._build_query_params(use_asr_endpoint)

                try:
                    async with session.post(transcribe_url, data=data, params=params) as response:
                        if response.status == 404 and self._has_alternative_paths(path):
                            logger.debug(f"Endpoint {transcribe_url} returned 404. Trying next path.")
                            last_error = f"404 {transcribe_url}"
                            continue

                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Transcription failed ({response.status}) from {transcribe_url}: {error_text}")
                            last_error = f"{response.status} {error_text}"
                            continue

                        payload = await self._extract_payload(response)
                        transcript = self._extract_transcription_text(payload)
                        if transcript:
                            return transcript

                        logger.warning(f"Whisper response from {transcribe_url} did not contain transcription text")
                        last_error = "empty transcription"
                except Exception as e:
                    logger.error(f"Request to {transcribe_url} failed: {str(e)}")
                    last_error = str(e)

            if last_error:
                logger.error(f"All configured endpoints for {url} failed: {last_error}")
            return None

    async def _extract_payload(self, response: aiohttp.ClientResponse) -> Union[str, dict, list, None]:
        """Decode Whisper response into python data structures."""
        content_type_header = response.headers.get("Content-Type", "")
        if "application/json" in content_type_header or self.output_format == "json":
            try:
                return await response.json()
            except aiohttp.ContentTypeError:
                text_body = await response.text()
                return self._maybe_parse_json(text_body)

        text_body = await response.text()
        return self._maybe_parse_json(text_body)

    def _build_form_data(
        self,
        audio_data: bytes,
        filename: str,
        content_type: str,
        use_asr_endpoint: bool,
    ) -> aiohttp.FormData:
        """Create a multipart form payload for Whisper."""
        data = aiohttp.FormData()
        field_name = 'audio_file' if use_asr_endpoint else 'file'
        data.add_field(field_name, audio_data, filename=filename, content_type=content_type)

        data.add_field('response_format', self.output_format)
        if self.vad_filter is not None:
            data.add_field('vad_filter', 'true' if self.vad_filter else 'false')
        if self.language:
            data.add_field('language', self.language)
        if self.model:
            data.add_field('model', self.model)

        return data

    def _build_query_params(self, use_asr_endpoint: bool) -> Dict[str, str]:
        """Build query parameters tailored to the active endpoint style."""
        params: Dict[str, str] = {}

        if use_asr_endpoint:
            if self.language:
                params["language"] = self.language
            params["task"] = "transcribe"
            if self.vad_filter is not None:
                params["encode"] = "true" if self.vad_filter else "false"
            params["output"] = self._map_output_format()
        else:
            if self.language:
                params["language"] = self.language

        return params

    def _map_output_format(self) -> str:
        """Translate internal response format into Whisper-ASR output identifiers."""
        mapping = {
            "text": "txt",
            "txt": "txt",
            "json": "json",
            "srt": "srt",
            "vtt": "vtt",
            "tsv": "tsv",
        }
        return mapping.get((self.output_format or "txt").lower(), "txt")

    @staticmethod
    def _build_endpoint(base_url: str, path: str) -> str:
        """Combine base URL with relative path."""
        if not path:
            return base_url.rstrip('/')

        if path.startswith("http://") or path.startswith("https://"):
            return path.rstrip('/')

        base = base_url.rstrip('/')
        path_component = path.lstrip('/')
        return f"{base}/{path_component}"

    def _has_alternative_paths(self, current_path: str) -> bool:
        """Determine if there are other paths to try beyond the current one."""
        try:
            index = self.transcribe_paths.index(current_path)
        except ValueError:
            return False
        return index < len(self.transcribe_paths) - 1

    @staticmethod
    def _normalize_paths(paths: Optional[List[str]]) -> List[str]:
        """Prepare list of transcription endpoints to probe."""
        if paths:
            normalized = [p.strip() for p in paths if p and p.strip()]
            if normalized:
                return normalized

        # Default endpoints to try, ordered by most common APIs first.
        return [
            "asr",
            "v1/audio/transcriptions",
            "audio/transcriptions",
            "v1/transcriptions",
            "transcriptions",
            "",
        ]

    @staticmethod
    async def probe_transcribe_paths(
        urls: List[str],
        candidate_paths: List[str],
        timeout_seconds: int = 10,
    ) -> Tuple[List[str], Dict[str, Dict[str, Optional[Union[int, str]]]]]:
        """
        Probe Whisper endpoints to determine which paths are reachable.

        Args:
            urls: Base URLs for Whisper services.
            candidate_paths: Paths to test in order.
            timeout_seconds: Timeout applied to each HTTP request.

        Returns:
            Tuple containing reordered paths (working paths first) and probe results per URL.
        """
        if not urls or not candidate_paths:
            return candidate_paths, {}

        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        successful_paths: List[str] = []
        results: Dict[str, Dict[str, Optional[Union[int, str]]]] = {}

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for url in urls:
                url_result: Dict[str, Optional[Union[int, str]]] = {
                    "status": "unreachable",
                    "path": None,
                    "http_status": None,
                    "error": None,
                }

                for path in candidate_paths:
                    endpoint = WhisperClient._build_endpoint(url, path)
                    try:
                        async with session.head(endpoint, allow_redirects=True) as response:
                            status = response.status

                        if status == 405:
                            async with session.get(endpoint, allow_redirects=True) as response:
                                status = response.status

                        if status == 404:
                            continue

                        if status >= 500:
                            url_result = {
                                "status": "error",
                                "path": path,
                                "http_status": status,
                                "error": f"HTTP {status}",
                            }
                            continue

                        url_result = {
                            "status": "reachable",
                            "path": path,
                            "http_status": status,
                            "error": None,
                        }

                        if path not in successful_paths:
                            successful_paths.append(path)
                        break
                    except Exception as exc:
                        url_result = {
                            "status": "error",
                            "path": path,
                            "http_status": None,
                            "error": str(exc),
                        }

                results[url] = url_result

        reordered_paths = successful_paths + [p for p in candidate_paths if p not in successful_paths]
        return reordered_paths, results

    @staticmethod
    def _maybe_parse_json(body: str) -> Union[str, dict, list]:
        """Attempt to parse a text body as JSON; return raw string on failure."""
        if not body:
            return body
        stripped = body.strip()
        if stripped and stripped[0] in ("{", "["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
        return stripped

    @staticmethod
    def _extract_transcription_text(payload: Union[str, dict, list, None]) -> Optional[str]:
        """Normalize Whisper API responses into a plain transcription string."""
        if payload is None:
            return None

        if isinstance(payload, str):
            return payload.strip() or None

        if isinstance(payload, dict):
            candidate = WhisperClient._extract_from_dict(payload)
            if candidate:
                return candidate

        if isinstance(payload, list):
            parts: List[str] = []
            for item in payload:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item.strip())
                elif isinstance(item, dict):
                    text_value = WhisperClient._extract_from_dict(item)
                    if text_value:
                        parts.append(text_value)
            if parts:
                return " ".join(parts).strip()

        return None

    @staticmethod
    def _extract_from_dict(payload: dict) -> Optional[str]:
        """Helper to extract text from a Whisper response dictionary."""
        text_keys = ("text", "transcription", "result", "data")
        for key in text_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                segments = [segment.strip() for segment in value if isinstance(segment, str) and segment.strip()]
                if segments:
                    return " ".join(segments).strip()

        segments_value = payload.get("segments")
        if isinstance(segments_value, list):
            collected: List[str] = []
            for segment in segments_value:
                if isinstance(segment, dict):
                    segment_text = segment.get("text") or segment.get("transcription")
                    if isinstance(segment_text, str) and segment_text.strip():
                        collected.append(segment_text.strip())
            if collected:
                return " ".join(collected).strip()

        return None


class KokoroTTSClient:
    """Client for Kokoro TTS service with failover support"""

    def __init__(self, kokoro_urls: str, voice: str = "af_bella",
                 response_format: str = "mp3", **kwargs):
        """
        Initialize Kokoro TTS client with multiple instance support

        Args:
            kokoro_urls: Comma-separated list of Kokoro TTS URLs
            voice: Voice for TTS synthesis
            response_format: Audio output format
            **kwargs: Additional TTS parameters
        """
        self.urls = [url.strip() for url in kokoro_urls.split(',')]
        self.voice = voice
        self.response_format = response_format
        self.kwargs = kwargs
        logger.info(f"Initialized KokoroTTSClient with {len(self.urls)} instances")

    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech from text with failover support

        Args:
            text: Text to convert to speech

        Returns:
            Audio data in bytes or None if all instances fail
        """
        # Try each Kokoro instance in order
        for url in self.urls:
            try:
                result = await self._synthesize_with_instance(url, text)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Kokoro instance {url} failed: {str(e)}")
                continue

        logger.error("All Kokoro instances failed")
        return None

    async def _synthesize_with_instance(self, url: str, text: str) -> Optional[bytes]:
        """Synthesize speech using specific Kokoro instance"""
        payload = {
            "input": text,
            "voice": self.voice,
            "response_format": self.response_format,
            **self.kwargs
        }

        synthesize_url = f"{url.rstrip('/')}/v1/audio/speech"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(synthesize_url, json=payload) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        error_text = await response.text()
                        logger.error(f"Speech synthesis failed: {response.status} - {error_text}")
                        return None
            except Exception as e:
                logger.error(f"Request to {url} failed: {str(e)}")
                return None
