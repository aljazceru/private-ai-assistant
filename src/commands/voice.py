"""
Voice message handling for Signal Bot.

Provides speech-to-text transcription and text-to-speech synthesis for voice messages.
"""

import os
import base64
import tempfile
import logging
import re
import aiohttp
from signalbot import Command, Context

logger = logging.getLogger(__name__)

MAX_CONVERSATION_HISTORY = 10
VOICE_ACTIVATION_PHRASE = os.getenv('VOICE_ACTIVATION_PHRASE', 'hey assistant').lower().strip()


class VoiceMessageHandler(Command):
    def __init__(self, stt_client=None, tts_client=None):
        self.conversations = {}  # Store conversations here instead
        self.stt_client = stt_client  # Speech-to-Text client (Whisper or Sherpa)
        self.tts_client = tts_client  # Text-to-Speech client (Kokoro or Sherpa)

    def describe(self) -> str:
        return "Voice message handler"

    async def send_ai_response(self, c: Context, response_text: str):
        """Send AI response as both text and voice (if TTS available)"""
        # Always send text response first
        await c.send(f"🤖 {response_text}")

        # Generate and send voice response if TTS is available
        if self.tts_client:
            try:
                audio_data = await self.tts_client.synthesize_speech(response_text)
                if audio_data:
                    # Write audio data to temporary file
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
                            logger.info(f"Generated TTS audio ({len(audio_data)} bytes) but no file sending method found for: {response_text[:50]}...")
                            logger.info("TTS audio generation working, but signalbot file sending API needs to be researched")

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

    def add_user_message(self, sender: str, content: str):
        """Add user message to conversation history with automatic cleanup"""
        if sender not in self.conversations:
            self.conversations[sender] = []

        self.conversations[sender].append({
            "role": "user",
            "content": content
        })

        # Keep only last MAX_CONVERSATION_HISTORY messages for context
        if len(self.conversations[sender]) > MAX_CONVERSATION_HISTORY:
            self.conversations[sender] = self.conversations[sender][-MAX_CONVERSATION_HISTORY:]

    def add_assistant_message(self, sender: str, content: str):
        """Add assistant message to conversation history"""
        if sender not in self.conversations:
            self.conversations[sender] = []

        self.conversations[sender].append({
            "role": "assistant",
            "content": content
        })

    async def handle(self, c: Context):
        """Handle voice messages by transcribing them"""
        sender = c.message.source

        logger.info(f"Processing voice message from {sender}")

        try:
            audio_data = None
            file_extension = ".ogg"  # Default extension

            # Try base64 attachments first (preferred method)
            if hasattr(c.message, 'base64_attachments') and c.message.base64_attachments:
                audio_data = base64.b64decode(c.message.base64_attachments[0])

            # Try local attachment files
            elif hasattr(c.message, 'attachments_local_filenames') and c.message.attachments_local_filenames:
                local_filename = c.message.attachments_local_filenames[0]
                try:
                    with open(local_filename, 'rb') as f:
                        audio_data = f.read()
                except Exception as e:
                    logger.error(f"Failed to read local attachment file {local_filename}: {e}")
                    await c.send("Unable to read voice message attachment.")
                    return

            # Download from signal-cli API directly (fallback method)
            elif hasattr(c.message, '_manual_attachments') and c.message._manual_attachments:
                attachment = c.message._manual_attachments[0]
                att_id = attachment['id']

                # Extract file extension from content type or filename
                content_type = attachment.get('contentType', 'audio/ogg')
                filename = attachment.get('filename', '')

                # Map content types to file extensions
                extension_map = {
                    'audio/aac': '.aac',
                    'audio/ogg': '.ogg',
                    'audio/mpeg': '.mp3',
                    'audio/mp4': '.m4a',
                    'audio/wav': '.wav',
                    'audio/x-wav': '.wav'
                }

                # Try to get extension from content type first, then from filename
                file_extension = extension_map.get(content_type, '.ogg')
                if filename and '.' in filename:
                    file_extension = '.' + filename.split('.')[-1]

                logger.debug(f"Audio content type: {content_type}, filename: {filename}, using extension: {file_extension}")

                # Download attachment from signal-cli-rest-api
                signal_service = os.getenv("SIGNAL_SERVICE", "127.0.0.1:18380")
                download_url = f"http://{signal_service}/v1/attachments/{att_id}"

                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(download_url) as response:
                            if response.status == 200:
                                audio_data = await response.read()
                            else:
                                logger.error(f"Failed to download attachment {att_id}: {response.status}")
                                await c.send("Unable to download voice message.")
                                return
                    except Exception as e:
                        logger.error(f"Error downloading attachment {att_id}: {e}")
                        await c.send("Unable to download voice message.")
                        return
            else:
                await c.send("No voice attachment found.")
                return

            if not audio_data:
                await c.send("Voice message data is empty.")
                return

            # Transcribe audio
            logger.info(f"Transcribing {len(audio_data)} bytes of audio data")
            transcribed_text = await self.stt_client.transcribe_audio(audio_data, file_extension)

            if not transcribed_text:
                await c.send("Unable to transcribe voice message. Please try again.")
                return

            logger.info(f"Transcribed text: {transcribed_text}")

            # Check for voice activation phrase if set
            should_chat, chat_text, cleaned_transcription = self._parse_transcription(transcribed_text)

            # Send transcription back to user
            await c.send(f"🎤 Transcribed: {cleaned_transcription}")

            if should_chat:
                if chat_text:
                    self.add_user_message(sender, chat_text)
                    await c.send("Activation phrase detected. Forwarding to AI chat is not configured in this handler.")
                else:
                    await c.send("Activation phrase detected. What can I help you with?")
                return

            # Store memo transcription in conversation history for context
            if cleaned_transcription:
                self.add_user_message(sender, cleaned_transcription)
                await c.send("Voice message transcribed. You can send a text message for AI response.")

        except Exception as e:
            logger.error(f"Error processing voice message: {str(e)}")
            await c.send("Error processing voice message. Please try again.")

    @staticmethod
    def _parse_transcription(transcription: str):
        """Parse transcription and determine if activation phrase is present."""
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
