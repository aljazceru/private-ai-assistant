"""
Signal Bot - Main application entry point.

Provides AI chat capabilities with local speech processing and role-based permissions.
"""

import os
import asyncio
import logging
import mimetypes
import re
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, List, Dict, Optional
from signalbot import SignalBot
from dotenv import load_dotenv

if TYPE_CHECKING:
    from signalbot import Context

# Import our modular components
from .permissions.manager import PermissionManager
from .clients.stt import STTClient
from .clients.tts import TTSClient
from .clients.api import PrivateModeClient, WhisperClient, KokoroTTSClient
from .commands.chat import ChatCommand, ClearCommand
from .commands.admin import ModelsCommand, HelpCommand, AddUserCommand, RemoveUserCommand, ListUsersCommand, ListRolesCommand
from .commands.voice import VoiceMessageHandler

load_dotenv()

# Configuration
VOICE_ACTIVATION_PHRASE = os.getenv('VOICE_ACTIVATION_PHRASE', 'hey assistant').lower().strip()

# Constants
MAX_CONVERSATION_HISTORY = 10
TTS_TIMEOUT_SECONDS = 30
WHISPER_TIMEOUT_SECONDS = 60

AUDIO_EXTENSION_MAP = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}

# Valid bot commands
VALID_COMMANDS = {
    "!clear",
    "!models",
    "!help",
    "!adduser",
    "!removeuser",
    "!listusers",
    "!listroles"
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedMessageHandler:
    """Unified message handler for both text and voice messages"""

    def __init__(self, stt_client=None, tts_client=None, permission_manager=None,
                 privatemode_client=None, model=None, storage=None):
        self.conversations = {}  # Store conversations here (cache)
        self.stt_client = stt_client  # Speech-to-Text client
        self.tts_client = tts_client  # Text-to-Speech client
        self.permission_manager = permission_manager
        self.privatemode_client = privatemode_client
        self.model = model
        self.storage = storage  # Persistent storage

        # Command instances
        self.clear_command = ClearCommand(self, permission_manager)
        self.models_command = ModelsCommand(privatemode_client, permission_manager)
        self.help_command = HelpCommand(permission_manager)
        self.add_user_command = AddUserCommand(permission_manager)
        self.remove_user_command = RemoveUserCommand(permission_manager)
        self.list_users_command = ListUsersCommand(permission_manager)
        self.list_roles_command = ListRolesCommand(permission_manager)

    def describe(self) -> str:
        return "Main message handler"

    def setup(self):
        """Setup method called by SignalBot during registration"""
        logger.info("UnifiedMessageHandler setup complete")

    def _determine_audio_extension(self, c) -> str:
        """Infer the best audio file extension for an attachment"""
        default_extension = ".ogg"

        try:
            # Prefer explicit local filenames from signalbot
            if hasattr(c.message, "attachments_local_filenames") and c.message.attachments_local_filenames:
                suffix = Path(c.message.attachments_local_filenames[0]).suffix
                if suffix:
                    return suffix.lower()

            attachment_candidates = []

            if hasattr(c.message, "_manual_attachments") and c.message._manual_attachments:
                attachment_candidates.extend(c.message._manual_attachments)
            elif hasattr(c.message, "raw_message") and isinstance(c.message.raw_message, dict):
                envelope = c.message.raw_message.get("envelope", {})
                data_message = envelope.get("dataMessage", {})
                attachments = data_message.get("attachments") or []
                if isinstance(attachments, list):
                    attachment_candidates.extend(attachments)

            for attachment in attachment_candidates:
                content_type = (attachment.get("contentType") or attachment.get("content_type") or "").lower()
                filename = attachment.get("filename") or attachment.get("name")

                if content_type:
                    guessed = AUDIO_EXTENSION_MAP.get(content_type)
                    if not guessed:
                        guessed = mimetypes.guess_extension(content_type)
                    if guessed:
                        normalized = guessed if guessed.startswith(".") else f".{guessed}"
                        return normalized.lower()

                if filename:
                    suffix = Path(filename).suffix
                    if suffix:
                        return suffix.lower()

        except Exception as exc:
            logger.debug(f"Unable to infer audio extension: {exc}")

        return default_extension

    def _parse_voice_transcription(self, transcription: str) -> Tuple[bool, str, str]:
        """
        Determine whether a transcription should trigger chat mode.

        Returns:
            should_chat: True when the transcription starts with the activation phrase
            chat_text: Transcription content with activation phrase removed (if present)
            cleaned_transcription: Whitespace-trimmed transcription for display
        """
        if not transcription:
            return False, "", ""

        cleaned = transcription.strip()
        if not cleaned:
            return False, "", ""

        if VOICE_ACTIVATION_PHRASE and cleaned.lower().startswith(VOICE_ACTIVATION_PHRASE):
            activation_pattern = f'^{re.escape(VOICE_ACTIVATION_PHRASE)}[,\\s]*'
            chat_text = re.sub(activation_pattern, '', cleaned, flags=re.IGNORECASE).strip()
            return True, chat_text, cleaned

        return False, cleaned, cleaned

    async def send_ai_response(self, c, response_text: str):
        """Send AI response as both text and voice (if TTS available)"""
        # Always send text response first
        await c.send(f"🤖 {response_text}")

        # Generate and send voice response if TTS is available
        if self.tts_client:
            try:
                audio_data = await self.tts_client.synthesize_speech(response_text)
                if audio_data:
                    # Write audio data to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=f".{self.tts_client.response_format}", delete=False) as temp_file:
                        temp_file.write(audio_data)
                        temp_file_path = temp_file.name

                    try:
                        # Try to send the audio file using various possible signalbot methods
                        sent = False

                        # Try different possible methods for sending attachments
                        if hasattr(c, 'send_attachment'):
                            await c.send_attachment(temp_file_path)
                            sent = True
                            logger.info(f"Sent TTS audio via send_attachment: {response_text[:50]}...")
                        elif hasattr(c, 'send_file'):
                            await c.send_file(temp_file_path)
                            sent = True
                            logger.info(f"Sent TTS audio via send_file: {response_text[:50]}...")
                        elif hasattr(c, 'send_media'):
                            await c.send_media(temp_file_path)
                            sent = True
                            logger.info(f"Sent TTS audio via send_media: {response_text[:50]}...")

                        if not sent:
                            # Log that TTS was generated but couldn't be sent
                            logger.info(f"Generated TTS audio ({len(audio_data)} bytes) but no file sending method found")
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                else:
                    logger.warning("TTS synthesis failed for response")
            except Exception as e:
                logger.error(f"Error generating TTS response: {str(e)}")

    def _get_conversation_key(self, sender: str) -> str:
        """Get storage key for conversation"""
        return f"conversation:{sender}"

    def _load_conversation(self, sender: str):
        """Load conversation from persistent storage"""
        if sender in self.conversations:
            return self.conversations[sender]

        if self.storage:
            try:
                key = self._get_conversation_key(sender)
                conversation = self.storage.read(key)
                if conversation:
                    self.conversations[sender] = conversation
                    logger.debug(f"Loaded conversation for {sender} from storage")
                    return conversation
            except Exception as e:
                logger.debug(f"Failed to load conversation for {sender}: {e}")

        # Default to empty conversation
        self.conversations[sender] = []
        return self.conversations[sender]

    def _save_conversation(self, sender: str):
        """Save conversation to persistent storage"""
        if self.storage and sender in self.conversations:
            try:
                key = self._get_conversation_key(sender)
                self.storage.write(key, self.conversations[sender])
                logger.debug(f"Saved conversation for {sender} to storage")
            except Exception as e:
                logger.debug(f"Failed to save conversation for {sender}: {e}")

    def add_user_message(self, sender: str, content: str):
        """Add user message to conversation history with automatic cleanup"""
        conversation = self._load_conversation(sender)

        conversation.append({
            "role": "user",
            "content": content
        })

        # Keep only last MAX_CONVERSATION_HISTORY messages for context
        if len(conversation) > MAX_CONVERSATION_HISTORY:
            conversation = conversation[-MAX_CONVERSATION_HISTORY:]

        self.conversations[sender] = conversation
        self._save_conversation(sender)

    def add_assistant_message(self, sender: str, content: str):
        """Add assistant message to conversation history"""
        conversation = self._load_conversation(sender)

        conversation.append({
            "role": "assistant",
            "content": content
        })

        self.conversations[sender] = conversation
        self._save_conversation(sender)

    async def handle(self, c):
        """Main message handling logic"""
        from signalbot import Context
        sender = c.message.source
        message_text = c.message.text

        # Check for voice messages (attachments)
        has_base64_attachments = hasattr(c.message, 'base64_attachments') and c.message.base64_attachments
        has_local_attachments = hasattr(c.message, 'attachments_local_filenames') and c.message.attachments_local_filenames

        # Extract attachment data from raw message if signalbot library fails
        has_manual_attachments = False
        if hasattr(c.message, 'raw_message') and isinstance(c.message.raw_message, dict):
            if 'envelope' in c.message.raw_message and 'dataMessage' in c.message.raw_message['envelope']:
                data_msg = c.message.raw_message['envelope']['dataMessage']
                if 'attachments' in data_msg:
                    raw_attachments = data_msg['attachments']
                    for attachment in raw_attachments:
                        att_id = attachment.get('id')
                        if att_id:
                            if not hasattr(c.message, '_manual_attachments'):
                                c.message._manual_attachments = []
                            c.message._manual_attachments.append({
                                'id': att_id,
                                'contentType': attachment.get('contentType'),
                                'size': attachment.get('size'),
                                'filename': attachment.get('filename')
                            })
                            has_manual_attachments = True

        # Process voice message if attachments found
        if has_base64_attachments or has_local_attachments or has_manual_attachments:
            await self.handle_voice_message(c)
            return

        # Skip messages with no text content
        if not message_text:
            return

        logger.info(f"Received message from {sender}: {message_text}")

        # Check if message starts with ! (command) and is a valid command
        if message_text.startswith('!') and message_text.split()[0].lower() in VALID_COMMANDS:
            await self.handle_command(c, message_text)
        else:
            # Regular chat message - check chat permission
            if not self.permission_manager.user_has_permission(sender, "chat"):
                self.permission_manager.log_unauthorized_access(sender, "chat", message_text)
                return

            await self.handle_chat(c, message_text)

    async def handle_command(self, c, message_text: str):
        """Handle command messages that start with !"""
        sender = c.message.source
        command_parts = message_text.strip().split()
        command = command_parts[0].lower()

        # Route to appropriate command handler
        if command == "!clear":
            if not self.permission_manager.user_has_permission(sender, "chat"):
                self.permission_manager.log_unauthorized_access(sender, "chat", message_text)
                return
            await self.clear_command.handle(c)

        elif command == "!models":
            if not self.permission_manager.user_has_permission(sender, "models"):
                self.permission_manager.log_unauthorized_access(sender, "models", message_text)
                return
            await self.models_command.handle(c)

        elif command == "!help":
            if not self.permission_manager.user_has_permission(sender, "help"):
                self.permission_manager.log_unauthorized_access(sender, "help", message_text)
                return
            await self.help_command.handle(c)

        elif command == "!adduser":
            if not self.permission_manager.user_has_permission(sender, "admin"):
                self.permission_manager.log_unauthorized_access(sender, "admin", message_text)
                return
            await self.add_user_command.handle(c)

        elif command == "!removeuser":
            if not self.permission_manager.user_has_permission(sender, "admin"):
                self.permission_manager.log_unauthorized_access(sender, "admin", message_text)
                return
            await self.remove_user_command.handle(c)

        elif command == "!listusers":
            if not self.permission_manager.user_has_permission(sender, "admin"):
                self.permission_manager.log_unauthorized_access(sender, "admin", message_text)
                return
            await self.list_users_command.handle(c)

        elif command == "!listroles":
            if not self.permission_manager.user_has_permission(sender, "admin"):
                self.permission_manager.log_unauthorized_access(sender, "admin", message_text)
                return
            await self.list_roles_command.handle(c)

        else:
            # Unknown command - silently ignore
            return

    async def handle_chat(self, c, message_text: str):
        """Handle regular chat messages (non-commands)"""
        sender = c.message.source

        # Add user message to conversation history
        self.add_user_message(sender, message_text)

        # Get AI response
        conversation = self._load_conversation(sender)
        response = await self.privatemode_client.chat_completion(conversation, self.model)

        # Add assistant response to history
        self.add_assistant_message(sender, response)

        # Send response
        await self.send_ai_response(c, response)

    async def handle_voice_message(self, c):
        """Handle voice messages by transcribing them"""
        import base64
        import aiohttp
        from signalbot import Context

        sender = c.message.source

        # Check if user has chat permission
        if not self.permission_manager.user_has_permission(sender, "chat"):
            self.permission_manager.log_unauthorized_access(sender, "voice_message", "Voice message")
            return

        # Check if STT client is configured
        if not self.stt_client:
            await c.send("Voice transcription is not configured. Please contact the administrator.")
            return

        logger.info(f"Processing voice message from {sender}")

        try:
            audio_data = None
            file_extension = self._determine_audio_extension(c)

            # Try base64 attachments first (preferred method)
            if hasattr(c.message, 'base64_attachments') and c.message.base64_attachments:
                audio_data = base64.b64decode(c.message.base64_attachments[0])

            # Try local attachment files
            elif hasattr(c.message, 'attachments_local_filenames') and c.message.attachments_local_filenames:
                local_filename = c.message.attachments_local_filenames[0]
                try:
                    with open(local_filename, 'rb') as f:
                        audio_data = f.read()
                    suffix = Path(local_filename).suffix
                    if suffix:
                        file_extension = suffix.lower()
                except Exception as e:
                    logger.error(f"Failed to read local attachment file {local_filename}: {e}")
                    await c.send("Unable to read voice message attachment.")
                    return

            # Download from signal-cli API directly (fallback method)
            elif hasattr(c.message, '_manual_attachments') and c.message._manual_attachments:
                attachment = c.message._manual_attachments[0]
                att_id = attachment['id']
                content_type = (attachment.get('contentType') or '').lower()
                filename = attachment.get('filename') or ''

                guessed_extension = AUDIO_EXTENSION_MAP.get(content_type) if content_type else None
                if not guessed_extension and content_type:
                    guessed_extension = mimetypes.guess_extension(content_type)
                if guessed_extension:
                    normalized = guessed_extension if guessed_extension.startswith('.') else f".{guessed_extension}"
                    file_extension = normalized.lower()
                elif filename:
                    suffix = Path(filename).suffix
                    if suffix:
                        file_extension = suffix.lower()
                logger.debug(
                    f"Inferred extension {file_extension} from content_type={content_type or 'unknown'} "
                    f"filename={filename or 'unknown'}"
                )

                # Download attachment from signal-cli-rest-api
                signal_service = os.getenv("SIGNAL_SERVICE", "127.0.0.1:18380")
                phone_number = os.getenv("SIGNAL_PHONE_NUMBER")
                attachment_url = f"http://{signal_service}/v1/attachments/{att_id}"

                async with aiohttp.ClientSession() as session:
                    try:
                        params = {"number": phone_number}
                        async with session.get(attachment_url, params=params) as response:
                            if response.status == 200:
                                audio_data = await response.read()
                                logger.info(f"Downloaded {len(audio_data)} bytes for voice transcription")
                            else:
                                error_text = await response.text()
                                logger.error(f"Failed to download attachment: {response.status} - {error_text}")
                                await c.send("Unable to download voice message attachment.")
                                return
                    except Exception as e:
                        logger.error(f"Error downloading attachment: {str(e)}")
                        await c.send("Error downloading voice message attachment.")
                        return

            if not audio_data:
                logger.warning("No attachment data available")
                await c.send("Voice message received but no audio data available.")
                return

            # Transcribe the audio
            transcription = await self.stt_client.transcribe_audio(audio_data, file_extension)

            if transcription:
                should_chat, chat_text, cleaned_transcription = self._parse_voice_transcription(transcription)

                if not should_chat:
                    # Just transcription request - send status and result
                    await c.send("Transcribing your message...")
                    await c.send(f"Transcription:\n{cleaned_transcription}")

                if should_chat:
                    if chat_text:  # Only proceed if there's text after activation phrase
                        # Add user message to conversation history
                        self.add_user_message(sender, chat_text)

                        # Get AI response
                        conversation = self._load_conversation(sender)
                        ai_response = await self.privatemode_client.chat_completion(conversation, self.model)

                        # Add assistant response to history
                        self.add_assistant_message(sender, ai_response)

                        # Send AI response (text + voice if TTS available)
                        await self.send_ai_response(c, ai_response)
                    else:
                        await self.send_ai_response(c, "Yes? How can I help you?")
            else:
                await c.send("Sorry, I couldn't transcribe the voice message. Please try again or check if the audio is clear.")

        except Exception as e:
            logger.error(f"Error processing voice message: {str(e)}")
            await c.send("An error occurred while processing the voice message. Please try again.")


def main():
    """Main application entry point"""
    logger.info("=== Starting Signal Bot Initialization ===")

    # Load configuration
    signal_service = os.getenv("SIGNAL_SERVICE", "localhost:8080")
    phone_number = os.getenv("SIGNAL_PHONE_NUMBER")

    logger.info(f"Signal service: {signal_service}")
    logger.info(f"Phone number: {phone_number}")

    if not phone_number:
        logger.error("SIGNAL_PHONE_NUMBER environment variable is required")
        return

    # PrivateMode API configuration
    privatemode_base_url = os.getenv("PRIVATEMODE_BASE_URL", "http://localhost:8080")
    model = os.getenv("PRIVATEMODE_MODEL", None)

    # Initialize PrivateMode client
    privatemode_client = PrivateModeClient(privatemode_base_url)

    # Initialize permission manager
    permission_manager = PermissionManager()

    # STT (Speech-to-Text) configuration
    stt_client = None
    stt_preferred_system = os.getenv("STT_PREFERRED_SYSTEM", "auto").lower()
    whisper_urls = os.getenv("WHISPER_ASR_URLS", "")
    use_sherpa_stt = os.getenv("USE_SHERPA_STT", "true").lower() == "true"
    whisper_model = os.getenv("WHISPER_MODEL", "").strip() or None
    whisper_timeout_env = os.getenv("WHISPER_TIMEOUT_SECONDS", "").strip()
    try:
        whisper_timeout_seconds = int(whisper_timeout_env) if whisper_timeout_env else 60
    except ValueError:
        logger.warning(f"Invalid WHISPER_TIMEOUT_SECONDS value '{whisper_timeout_env}', defaulting to 60 seconds")
        whisper_timeout_seconds = 60
    whisper_language = os.getenv("WHISPER_LANGUAGE", "auto")
    whisper_path_env = os.getenv("WHISPER_TRANSCRIBE_PATHS", "")
    user_defined_whisper_paths = [p.strip() for p in whisper_path_env.split(',') if p.strip()]
    whisper_transcribe_paths = WhisperClient._normalize_paths(user_defined_whisper_paths or None)
    whisper_url_list = [url.strip() for url in whisper_urls.split(',') if url.strip()]
    whisper_probe_timeout = min(whisper_timeout_seconds, 10) if whisper_timeout_seconds else 10

    if whisper_url_list:
        probe_loop = None
        try:
            probe_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(probe_loop)
            probe_paths, probe_results = probe_loop.run_until_complete(
                WhisperClient.probe_transcribe_paths(
                    whisper_url_list,
                    whisper_transcribe_paths,
                    timeout_seconds=whisper_probe_timeout,
                )
            )

            whisper_transcribe_paths = probe_paths or whisper_transcribe_paths

            for base_url, result in probe_results.items():
                status = result.get("status")
                path = result.get("path")
                http_status = result.get("http_status")
                error_detail = result.get("error")

                if status == "reachable":
                    if path:
                        display_path = f"/{path.lstrip('/')}"
                    else:
                        display_path = "(root)"
                    logger.info(f"Verified Whisper endpoint {base_url} -> {display_path} (HTTP {http_status})")
                else:
                    logger.error(
                        f"Failed to verify Whisper endpoint {base_url}: "
                        f"{error_detail or f'HTTP {http_status}' if http_status else 'no response'}"
                    )
        except Exception as e:
            logger.error(f"Error probing Whisper endpoints: {e}")
        finally:
            if probe_loop is not None:
                probe_loop.close()
            asyncio.set_event_loop(None)

    def initialize_sherpa_stt():
        """Initialize Sherpa-ONNX STT client"""
        sherpa_models_root = os.getenv("SHERPA_MODELS_ROOT", "./models")
        sherpa_asr_model = os.getenv("SHERPA_ASR_MODEL", "sensevoice")
        sherpa_asr_language = os.getenv("SHERPA_ASR_LANGUAGE", "zh")
        sherpa_provider = os.getenv("SHERPA_PROVIDER", "cpu")
        sherpa_threads = int(os.getenv("SHERPA_THREADS", "2"))

        try:
            return STTClient(
                models_root=sherpa_models_root,
                asr_model=sherpa_asr_model,
                asr_language=sherpa_asr_language,
                provider=sherpa_provider,
                num_threads=sherpa_threads,
                whisper_urls=whisper_urls,
                whisper_model=whisper_model,
                whisper_timeout_seconds=whisper_timeout_seconds,
                whisper_transcribe_paths=whisper_transcribe_paths,
                output_format="text",
                vad_filter=True,
                language=whisper_language
            )
        except Exception as e:
            logger.error(f"Failed to initialize Sherpa-ONNX STT: {e}")
            return None

    def initialize_whisper_stt():
        """Initialize Whisper STT client"""
        if not whisper_urls:
            logger.warning("Whisper URLs not configured")
            return None

        try:
            return WhisperClient(
                whisper_urls=whisper_urls,
                output_format="text",
                vad_filter=True,
                language=whisper_language,
                model=whisper_model,
                timeout_seconds=whisper_timeout_seconds,
                transcribe_paths=whisper_transcribe_paths,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Whisper client: {e}")
            return None

    # Debug logging
    logger.info(f"STT Configuration:")
    logger.info(f"  Preferred system: {stt_preferred_system}")
    logger.info(f"  Sherpa enabled: {use_sherpa_stt}")
    logger.info(f"  Whisper URLs: {whisper_urls}")
    if whisper_model:
        logger.info(f"  Whisper model: {whisper_model}")
    logger.info(f"  Whisper timeout: {whisper_timeout_seconds}s")
    if whisper_transcribe_paths:
        logger.info(f"  Whisper transcribe paths: {whisper_transcribe_paths}")

    # Initialize based on preferred system
    if stt_preferred_system == "sherpa":
        if use_sherpa_stt:
            stt_client = initialize_sherpa_stt()
            if stt_client:
                logger.info("STT: Using Sherpa-ONNX (preferred)")
            else:
                logger.error("Failed to initialize preferred Sherpa-ONNX STT")
        else:
            logger.warning("Sherpa-ONNX STT disabled by USE_SHERPA_STT=false")

    elif stt_preferred_system == "whisper":
        stt_client = initialize_whisper_stt()
        if stt_client:
            logger.info("STT: Using Whisper API (preferred)")
        else:
            logger.error("Failed to initialize preferred Whisper STT")

    else:  # auto mode (default)
        if use_sherpa_stt:
            stt_client = initialize_sherpa_stt()
            if stt_client:
                logger.info("STT: Using Sherpa-ONNX (auto mode)")
            else:
                logger.info("Sherpa-ONNX failed, trying Whisper API")
                stt_client = initialize_whisper_stt()
                if stt_client:
                    logger.info("STT: Using Whisper API (fallback)")
        else:
            # Sherpa disabled, try Whisper directly
            stt_client = initialize_whisper_stt()
            if stt_client:
                logger.info("STT: Using Whisper API (Sherpa disabled)")

    if not stt_client:
        logger.error("Failed to initialize any STT client - voice features will be unavailable")
    else:
        logger.info(f"STT client initialized successfully: {type(stt_client).__name__}")

    # TTS (Text-to-Speech) configuration
    tts_client = None
    use_sherpa_tts = os.getenv("USE_SHERPA_TTS", "true").lower() == "true"

    if use_sherpa_tts:
        # Sherpa-ONNX TTS configuration
        sherpa_models_root = os.getenv("SHERPA_MODELS_ROOT", "./models")
        sherpa_tts_model = os.getenv("SHERPA_TTS_MODEL", "vits-zh-hf-theresa")
        sherpa_provider = os.getenv("SHERPA_PROVIDER", "cpu")
        sherpa_threads = int(os.getenv("SHERPA_THREADS", "2"))

        try:
            tts_client = TTSClient(
                models_root=sherpa_models_root,
                tts_model=sherpa_tts_model,
                provider=sherpa_provider,
                num_threads=sherpa_threads,
                voice=os.getenv("KOKORO_VOICE", "af_bella"),
                response_format=os.getenv("TTS_RESPONSE_FORMAT", "mp3")
            )
            logger.info(f"Sherpa-ONNX TTS enabled with model: {sherpa_tts_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Sherpa-ONNX TTS: {e}")
            tts_client = None

    # Fallback to Kokoro API if Sherpa-ONNX fails
    if not tts_client:
        kokoro_urls = os.getenv("KOKORO_URLS", "")
        if kokoro_urls:
            try:
                tts_client = KokoroTTSClient(
                    kokoro_urls=kokoro_urls,
                    voice=os.getenv("KOKORO_VOICE", "af_bella"),
                    response_format=os.getenv("TTS_RESPONSE_FORMAT", "mp3")
                )
                logger.info("Fallback to Kokoro API for TTS")
            except Exception as e:
                logger.error(f"Failed to initialize Kokoro client: {e}")

    # Create Signal bot instance
    bot_config = {
        "signal_service": signal_service,
        "phone_number": phone_number
    }

    # Add storage configuration for persistent storage
    storage_type = os.getenv("STORAGE_TYPE", "sqlite")
    if storage_type == "sqlite":
        # Use /tmp/data directory which is mounted for persistence
        db_path = os.getenv("STORAGE_DB_PATH", "/tmp/data/bot.db")
        bot_config["storage"] = {
            "type": "sqlite",
            "sqlite_db": db_path,
            "check_same_thread": False
        }
        logger.info(f"Using SQLite persistent storage: {db_path}")
    else:
        bot_config["storage"] = {"type": "in-memory"}
        logger.info("Using in-memory storage")

    try:
        loop = asyncio.get_running_loop()
        if loop.is_closed():
            raise RuntimeError("Loop closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    bot = SignalBot(bot_config)

    # Create unified message handler with storage
    message_handler = UnifiedMessageHandler(
        stt_client=stt_client,
        tts_client=tts_client,
        permission_manager=permission_manager,
        privatemode_client=privatemode_client,
        model=model,
        storage=bot.storage
    )

    # Register the message handler
    bot.register(message_handler)

    logger.info(f"Starting Signal bot on {signal_service} with number {phone_number}")
    logger.info(f"Using PrivateMode API at {privatemode_base_url}")
    if model:
        logger.info(f"Using model: {model}")
    logger.info("Permission system enabled")
    if stt_client:
        logger.info("Voice transcription enabled")
    if tts_client:
        logger.info("Voice synthesis enabled")

    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        # Clean up file watcher
        permission_manager.stop_file_watcher()
