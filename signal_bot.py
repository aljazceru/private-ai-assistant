#!/usr/bin/env python3
import os
import asyncio
import logging
import aiohttp
import json
import time
import base64
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from signalbot import SignalBot, Command, Context
from dotenv import load_dotenv

load_dotenv()

# Configuration
VOICE_ACTIVATION_PHRASE = os.getenv('VOICE_ACTIVATION_PHRASE', 'hey jarvis').lower().strip()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
class PermissionFileHandler(FileSystemEventHandler):
    """File system event handler for permissions.json changes"""
    
    def __init__(self, permission_manager):
        self.permission_manager = permission_manager
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('permissions.json'):
            logger.info("Permissions file changed, reloading...")
            self.permission_manager.reload_config()


class PermissionManager:
    """Manages user permissions and authorization"""
    
    def __init__(self, config_path: str = "permissions.json", enable_file_watcher: bool = True):
        self.config_path = config_path
        self.config: Dict = {}
        self.observer: Optional[Observer] = None
        self.load_config()
        if enable_file_watcher:
            self.start_file_watcher()
        else:
            logger.info("File watcher disabled for debugging")
    
    def load_config(self) -> None:
        """Load permissions configuration from JSON file"""
        import os.path
        
        try:
            # Read config file with retry mechanism to avoid race conditions
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with open(self.config_path, 'r') as f:
                        file_content = f.read()
                    break
                except IOError as io_e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Permissions file read attempt {attempt + 1} failed: {io_e}, retrying...")
                        time.sleep(0.1)
                        continue
                    raise
            
            # Parse JSON configuration
            self.config = json.loads(file_content)
            logger.info(f"Loaded permissions config: {len(self.config.get('users', {}))} users, {len(self.config.get('roles', {}))} roles")
            
        except FileNotFoundError:
            logger.error(f"Permissions file {self.config_path} not found")
            self.config = {"roles": {}, "users": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in permissions file: {e}")
            logger.error(f"JSON error at line {e.lineno}, column {e.colno}, position {e.pos}")
            
            # Try to show the problematic part of the file
            try:
                with open(self.config_path, 'r') as f:
                    file_content = f.read()
                
                # Show context around the error
                error_pos = e.pos if e.pos is not None else 0
                start = max(0, error_pos - 50)
                end = min(len(file_content), error_pos + 50)
                context = file_content[start:end]
                
                logger.error(f"File content around error (pos {error_pos}): {repr(context)}")
                logger.error(f"Error character: {repr(file_content[error_pos:error_pos+1]) if error_pos < len(file_content) else 'EOF'}")
                
            except Exception as debug_e:
                logger.error(f"Could not read file for debugging: {debug_e}")
            
            self.config = {"roles": {}, "users": {}}
    
    def reload_config(self) -> None:
        """Reload configuration (called by file watcher)"""
        self.load_config()
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Permissions configuration saved")
        except Exception as e:
            logger.error(f"Failed to save permissions config: {e}")
    
    def start_file_watcher(self) -> None:
        """Start watching the permissions file for changes"""
        try:
            self.observer = Observer()
            event_handler = PermissionFileHandler(self)
            watch_dir = os.path.dirname(os.path.abspath(self.config_path)) or "."
            self.observer.schedule(event_handler, watch_dir, recursive=False)
            self.observer.start()
            logger.info(f"Started file watcher for {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
    
    def stop_file_watcher(self) -> None:
        """Stop the file watcher"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
    
    def is_user_authorized(self, phone_number: str) -> bool:
        """Check if user is authorized to use the bot"""
        users = self.config.get("users", {})
        return phone_number in users
    
    def get_user_role(self, phone_number: str) -> Optional[str]:
        """Get user's role"""
        user = self.config.get("users", {}).get(phone_number)
        return user.get("role") if user else None
    
    def get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a role"""
        role_data = self.config.get("roles", {}).get(role, {})
        return role_data.get("permissions", [])
    
    def user_has_permission(self, phone_number: str, permission: str) -> bool:
        """Check if user has specific permission"""
        if not self.is_user_authorized(phone_number):
            return False
        
        role = self.get_user_role(phone_number)
        if not role:
            return False
        
        permissions = self.get_role_permissions(role)
        
        # Check for wildcard permission
        if "*" in permissions:
            return True
        
        # Check for specific permission
        return permission in permissions
    
    def add_user(self, phone_number: str, role: str, name: str = None, added_by: str = "admin") -> bool:
        """Add a new user"""
        if role not in self.config.get("roles", {}):
            logger.error(f"Role '{role}' does not exist")
            return False
        
        user_data = {
            "role": role,
            "name": name or f"User {phone_number}",
            "added_by": added_by,
            "added_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.config.setdefault("users", {})[phone_number] = user_data
        self.save_config()
        logger.info(f"Added user {phone_number} with role {role}")
        return True
    
    def remove_user(self, phone_number: str) -> bool:
        """Remove a user"""
        if phone_number in self.config.get("users", {}):
            del self.config["users"][phone_number]
            self.save_config()
            logger.info(f"Removed user {phone_number}")
            return True
        return False
    
    def list_users(self, role_filter: str = None) -> Dict[str, Dict]:
        """List all users, optionally filtered by role"""
        users = self.config.get("users", {})
        if role_filter:
            return {phone: data for phone, data in users.items()
                   if data.get("role") == role_filter}
        return users
    
    def list_roles(self) -> Dict[str, Dict]:
        """List all available roles"""
        return self.config.get("roles", {})
    
    def log_unauthorized_access(self, phone_number: str, command: str = None, message: str = None) -> None:
        """Log unauthorized access attempt"""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phone_number": phone_number,
            "command": command,
            "message_preview": message[:50] + "..." if message and len(message) > 50 else message,
            "action": "unauthorized_access"
        }
        logger.warning(f"Unauthorized access attempt: {json.dumps(log_data)}")


def requires_permission(permission: str):
    """Decorator to check permissions before executing command"""
    def decorator(command_class):
        original_handle = command_class.handle
        
        async def wrapped_handle(self, c: Context):
            sender = c.message.source
            permission_manager = getattr(self, 'permission_manager', None)
            
            if not permission_manager:
                logger.error("No permission manager found in command")
                return
            
            if not permission_manager.user_has_permission(sender, permission):
                # Log unauthorized access
                permission_manager.log_unauthorized_access(
                    sender,
                    permission,
                    c.message.text
                )
                return  # Silently ignore unauthorized access
            
            # User is authorized, proceed with command
            await original_handle(self, c)
        
        command_class.handle = wrapped_handle
        return command_class
    
    return decorator


class PrivateModeClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Content-Type": "application/json"
        }
    
    async def list_models(self) -> list:
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
    
    async def chat_completion(self, messages: list, model: str = None) -> str:
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
    
    def __init__(self, whisper_urls: str, output_format: str = "text", 
                 vad_filter: bool = True, language: Optional[str] = None):
        """
        Initialize Whisper client with multiple instance support
        
        Args:
            whisper_urls: Comma-separated list of Whisper ASR URLs
            output_format: Output format (text, json, vtt, srt, tsv)
            vad_filter: Enable voice activity detection filter
            language: Language for transcription (auto-detect if None)
        """
        self.urls = [url.strip() for url in whisper_urls.split(',')]
        self.output_format = output_format
        self.vad_filter = vad_filter
        self.language = language if language else None
        logger.info(f"Initialized WhisperClient with {len(self.urls)} instances")
    
    async def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio data with failover support
        
        Args:
            audio_data: Audio file data in bytes
            
        Returns:
            Transcribed text or None if all instances fail
        """
        # Try each Whisper instance in order
        for url in self.urls:
            try:
                result = await self._transcribe_with_instance(url, audio_data)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Whisper instance {url} failed: {str(e)}")
                continue
        
        logger.error("All Whisper instances failed")
        return None
    
    async def _transcribe_with_instance(self, base_url: str, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio using a specific Whisper instance
        
        Args:
            base_url: Base URL of the Whisper instance
            audio_data: Audio file data in bytes
            
        Returns:
            Transcribed text or None if failed
        """
        url = f"{base_url.rstrip('/')}/asr"
        
        # Prepare form data
        data = aiohttp.FormData()
        data.add_field('audio_file',
                      audio_data,
                      filename='audio.ogg',
                      content_type='audio/ogg')
        
        # Build query parameters
        params = {
            'output': self.output_format,
            'task': 'transcribe',
            'vad_filter': str(self.vad_filter).lower()
        }
        
        if self.language:
            params['language'] = self.language
        
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout for transcription
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, data=data, params=params) as response:
                    if response.status == 200:
                        if self.output_format == 'json':
                            result = await response.json()
                            # Extract text from JSON response
                            if isinstance(result, dict) and 'text' in result:
                                return result['text'].strip()
                            elif isinstance(result, dict) and 'segments' in result:
                                # Combine all segments
                                segments = result['segments']
                                text = ' '.join(seg.get('text', '').strip() for seg in segments)
                                return text.strip()
                        else:
                            # Plain text response
                            text = await response.text()
                            return text.strip()
                    else:
                        error_text = await response.text()
                        logger.error(f"Whisper API error from {base_url}: {response.status} - {error_text}")
                        return None
            except asyncio.TimeoutError:
                logger.error(f"Whisper request to {base_url} timed out")
                return None
            except Exception as e:
                logger.error(f"Whisper request to {base_url} failed: {str(e)}")
                return None


@requires_permission("chat")
class ChatCommand(Command):
    def __init__(self, privatemode_client: PrivateModeClient, permission_manager: PermissionManager, model: str = None):
        self.privatemode_client = privatemode_client
        self.permission_manager = permission_manager
        self.model = model
        self.conversations = {}
    
    def describe(self) -> str:
        return "Chat with AI assistant"
    
    async def handle(self, c: Context):
        logger.info(f"ChatCommand.handle called with message: {c.message.text}")
        
        message_text = c.message.text
        if not message_text:
            await c.send("Please provide a message to chat with the AI.")
            return
        
        # Get or create conversation history for this sender
        sender = c.message.source
        if sender not in self.conversations:
            self.conversations[sender] = []
        
        # Add user message to history
        self.conversations[sender].append({
            "role": "user",
            "content": message_text
        })
        
        # Keep only last 10 messages for context
        if len(self.conversations[sender]) > 10:
            self.conversations[sender] = self.conversations[sender][-10:]
        
        # Get AI response
        response = await self.privatemode_client.chat_completion(self.conversations[sender], self.model)
        
        # Add assistant response to history+50672831532
        self.conversations[sender].append({
            "role": "assistant",
            "content": response
        })
        
        # Send response
        await c.send(response)


@requires_permission("chat")
class ClearCommand(Command):
    def __init__(self, chat_command: ChatCommand, permission_manager: PermissionManager):
        self.chat_command = chat_command
        self.permission_manager = permission_manager
    
    def describe(self) -> str:
        return "Clear conversation history"
    
    async def handle(self, c: Context):
        sender = c.message.source
        if sender in self.chat_command.conversations:
            del self.chat_command.conversations[sender]
            await c.send("Conversation history cleared.")
        else:
            await c.send("No conversation history to clear.")


@requires_permission("models")
class ModelsCommand(Command):
    def __init__(self, privatemode_client: PrivateModeClient, permission_manager: PermissionManager):
        self.privatemode_client = privatemode_client
        self.permission_manager = permission_manager
    
    def describe(self) -> str:
        return "List available AI models"
    
    async def handle(self, c: Context):
        models = await self.privatemode_client.list_models()
        if models:
            models_text = "Available models:\n" + "\n".join(f"• {model}" for model in models)
        else:
            models_text = "No models available or unable to fetch model list."
        await c.send(models_text)


@requires_permission("help")
class HelpCommand(Command):
    def __init__(self, permission_manager: PermissionManager):
        self.permission_manager = permission_manager
    
    def describe(self) -> str:
        return "Show available commands"
    
    async def handle(self, c: Context):
        help_text = """Available commands:
!clear - Clear conversation history
!models - List available models
!help - Show this help message

Admin commands (admin only):
!adduser <phone> <role> [name] - Add a new user
!removeuser <phone> - Remove a user
!listusers [role] - List users (optionally filtered by role)
!listroles - List available roles

Features:
• Send text messages for AI chat
• Send voice messages for automatic transcription
• Forward voice messages from other chats for transcription

You can also send messages without commands for direct chat."""
        await c.send(help_text)


@requires_permission("admin")
class AddUserCommand(Command):
    def __init__(self, permission_manager: PermissionManager):
        self.permission_manager = permission_manager
    
    def describe(self) -> str:
        return "Add a new user (admin only)"
    
    async def handle(self, c: Context):
        # Parse command arguments
        parts = c.message.text.strip().split()
        if len(parts) < 3:
            await c.send("Usage: !adduser <phone> <role> [name]")
            return
        
        phone = parts[1]
        role = parts[2]
        name = " ".join(parts[3:]) if len(parts) > 3 else None
        
        # Validate phone number format
        if not phone.startswith('+'):
            await c.send("Phone number must start with + (e.g., +1234567890)")
            return
        
        # Add user
        added_by = c.message.source
        if self.permission_manager.add_user(phone, role, name, added_by):
            user_name = name or f"User {phone}"
            await c.send(f"✅ Added user {user_name} ({phone}) with role '{role}'")
        else:
            await c.send(f"❌ Failed to add user. Role '{role}' may not exist.")


@requires_permission("admin")
class RemoveUserCommand(Command):
    def __init__(self, permission_manager: PermissionManager):
        self.permission_manager = permission_manager
    
    def describe(self) -> str:
        return "Remove a user (admin only)"
    
    async def handle(self, c: Context):
        # Parse command arguments
        parts = c.message.text.strip().split()
        if len(parts) != 2:
            await c.send("Usage: !removeuser <phone>")
            return
        
        phone = parts[1]
        
        # Check if user exists
        users = self.permission_manager.list_users()
        if phone not in users:
            await c.send(f"❌ User {phone} not found")
            return
        
        # Prevent removing yourself
        if phone == c.message.source:
            await c.send("❌ You cannot remove yourself")
            return
        
        # Remove user
        user_name = users[phone].get("name", phone)
        if self.permission_manager.remove_user(phone):
            await c.send(f"✅ Removed user {user_name} ({phone})")
        else:
            await c.send(f"❌ Failed to remove user {phone}")


@requires_permission("admin")
class ListUsersCommand(Command):
    def __init__(self, permission_manager: PermissionManager):
        self.permission_manager = permission_manager
    
    def describe(self) -> str:
        return "List users (admin only)"
    
    async def handle(self, c: Context):
        # Parse optional role filter
        parts = c.message.text.strip().split()
        role_filter = parts[1] if len(parts) > 1 else None
        
        users = self.permission_manager.list_users(role_filter)
        
        if not users:
            filter_text = f" with role '{role_filter}'" if role_filter else ""
            await c.send(f"No users found{filter_text}")
            return
        
        # Format user list
        role_text = f" with role '{role_filter}'" if role_filter else ""
        header = f"Users{role_text}:\n"
        user_lines = []
        
        for phone, data in users.items():
            name = data.get("name", "Unknown")
            role = data.get("role", "unknown")
            added_at = data.get("added_at", "unknown")[:10]  # Just the date part
            user_lines.append(f"• {name} ({phone}) - {role} - added {added_at}")
        
        message = header + "\n".join(user_lines)
        await c.send(message)


@requires_permission("admin")
class ListRolesCommand(Command):
    def __init__(self, permission_manager: PermissionManager):
        self.permission_manager = permission_manager
    
    def describe(self) -> str:
        return "List available roles (admin only)"
    
    async def handle(self, c: Context):
        roles = self.permission_manager.list_roles()
        
        if not roles:
            await c.send("No roles configured")
            return
        
        # Format roles list
        role_lines = []
        for role_name, role_data in roles.items():
            description = role_data.get("description", "No description")
            permissions = role_data.get("permissions", [])
            perm_text = ", ".join(permissions)
            role_lines.append(f"• **{role_name}**: {description}\n  Permissions: {perm_text}")
        
        message = "Available roles:\n\n" + "\n\n".join(role_lines)
        await c.send(message)


def main():
    # Load configuration
    signal_service = os.getenv("SIGNAL_SERVICE", "localhost:8080")
    phone_number = os.getenv("SIGNAL_PHONE_NUMBER")
    
    if not phone_number:
        logger.error("SIGNAL_PHONE_NUMBER environment variable is required")
        return
    
    # PrivateMode API configuration
    privatemode_base_url = os.getenv("PRIVATEMODE_BASE_URL", "http://localhost:8080")
    model = os.getenv("PRIVATEMODE_MODEL", None)
    
    # Initialize PrivateMode client
    privatemode_client = PrivateModeClient(privatemode_base_url)
    
    # Whisper ASR configuration
    whisper_urls = os.getenv("WHISPER_ASR_URLS", None)
    whisper_client = None
    
    if whisper_urls:
        whisper_output_format = os.getenv("WHISPER_OUTPUT_FORMAT", "text")
        whisper_vad_filter = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
        whisper_language = os.getenv("WHISPER_LANGUAGE", None)
        
        # Initialize Whisper client
        whisper_client = WhisperClient(
            whisper_urls=whisper_urls,
            output_format=whisper_output_format,
            vad_filter=whisper_vad_filter,
            language=whisper_language
        )
        logger.info("Whisper ASR transcription enabled")
    else:
        logger.info("Whisper ASR not configured, voice transcription disabled")
    
    # Initialize Permission Manager (disable file watcher for debugging if needed)
    enable_watcher = os.getenv("ENABLE_FILE_WATCHER", "true").lower() == "true"
    permission_manager = PermissionManager(enable_file_watcher=enable_watcher)
    
    # Initialize Signal bot
    bot = SignalBot({
        "signal_service": signal_service,
        "phone_number": phone_number,
        "logging_level": logging.INFO,
        "download_attachments": True  # Enable attachment downloading
    })
    
    # Create command instances
    chat_command = ChatCommand(privatemode_client, permission_manager, model)
    clear_command = ClearCommand(chat_command, permission_manager)
    models_command = ModelsCommand(privatemode_client, permission_manager)
    help_command = HelpCommand(permission_manager)
    
    # Create admin command instances
    add_user_command = AddUserCommand(permission_manager)
    remove_user_command = RemoveUserCommand(permission_manager)
    list_users_command = ListUsersCommand(permission_manager)
    list_roles_command = ListRolesCommand(permission_manager)
    
    # Single message handler that processes all messages
    class MessageHandler(Command):
        def __init__(self, whisper_client: Optional[WhisperClient] = None):
            self.conversations = {}  # Store conversations here instead
            self.whisper_client = whisper_client
        
        def describe(self) -> str:
            return "Main message handler"
        
        async def handle(self, c: Context):
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
                                    'size': attachment.get('size')
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
                if not permission_manager.user_has_permission(sender, "chat"):
                    permission_manager.log_unauthorized_access(sender, "chat", message_text)
                    return
                
                await self.handle_chat(c, message_text)
        
        async def handle_command(self, c: Context, message_text: str):
            """Handle command messages that start with !"""
            sender = c.message.source
            command_parts = message_text.strip().split()
            command = command_parts[0].lower()
            
            # Route to appropriate command handler
            if command == "!clear":
                if not permission_manager.user_has_permission(sender, "chat"):
                    permission_manager.log_unauthorized_access(sender, "chat", message_text)
                    return
                await clear_command.handle(c)
            
            elif command == "!models":
                if not permission_manager.user_has_permission(sender, "models"):
                    permission_manager.log_unauthorized_access(sender, "models", message_text)
                    return
                await models_command.handle(c)
            
            elif command == "!help":
                if not permission_manager.user_has_permission(sender, "help"):
                    permission_manager.log_unauthorized_access(sender, "help", message_text)
                    return
                await help_command.handle(c)
            
            elif command == "!adduser":
                if not permission_manager.user_has_permission(sender, "admin"):
                    permission_manager.log_unauthorized_access(sender, "admin", message_text)
                    return
                await add_user_command.handle(c)
            
            elif command == "!removeuser":
                if not permission_manager.user_has_permission(sender, "admin"):
                    permission_manager.log_unauthorized_access(sender, "admin", message_text)
                    return
                await remove_user_command.handle(c)
            
            elif command == "!listusers":
                if not permission_manager.user_has_permission(sender, "admin"):
                    permission_manager.log_unauthorized_access(sender, "admin", message_text)
                    return
                await list_users_command.handle(c)
            
            elif command == "!listroles":
                if not permission_manager.user_has_permission(sender, "admin"):
                    permission_manager.log_unauthorized_access(sender, "admin", message_text)
                    return
                await list_roles_command.handle(c)
            
            else:
                # Unknown command - silently ignore
                return
        
        async def handle_chat(self, c: Context, message_text: str):
            """Handle regular chat messages (non-commands)"""
            sender = c.message.source
            
            # Get or create conversation history for this sender
            if sender not in self.conversations:
                self.conversations[sender] = []
            
            # Add user message to history
            self.conversations[sender].append({
                "role": "user",
                "content": message_text
            })
            
            # Keep only last 10 messages for context
            if len(self.conversations[sender]) > 10:
                self.conversations[sender] = self.conversations[sender][-10:]
            
            # Get AI response
            response = await privatemode_client.chat_completion(self.conversations[sender], model)
            
            # Add assistant response to history
            self.conversations[sender].append({
                "role": "assistant",
                "content": response
            })
            
            # Send response
            await c.send(response)
        
        async def handle_voice_message(self, c: Context):
            """Handle voice messages by transcribing them"""
            sender = c.message.source
            
            # Check if user has chat permission
            if not permission_manager.user_has_permission(sender, "chat"):
                permission_manager.log_unauthorized_access(sender, "voice_message", "Voice message")
                return
            
            # Check if Whisper client is configured
            if not self.whisper_client:
                await c.send("Voice transcription is not configured. Please contact the administrator.")
                return
            
            logger.info(f"Processing voice message from {sender}")
            
            try:
                audio_data = None
                
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
                transcription = await self.whisper_client.transcribe_audio(audio_data)
                
                if transcription:
                    # Check if this should trigger AI chat (starts with configured activation phrase)
                    should_chat = transcription.lower().strip().startswith(VOICE_ACTIVATION_PHRASE)
                    
                    if not should_chat:
                        # Just transcription request - send status and result
                        await c.send("Transcribing your message...")
                        await c.send(f"Transcription:\n{transcription}")
                    # If should_chat=True, we don't send any intermediate messages
                    
                    if should_chat:
                        # Remove activation phrase from the transcription and process as chat
                        activation_pattern = f'^{re.escape(VOICE_ACTIVATION_PHRASE)}[,\s]*'
                        chat_text = re.sub(activation_pattern, '', transcription, flags=re.IGNORECASE).strip()
                        
                        if chat_text:  # Only proceed if there's text after activation phrase
                            # Store in conversation history  
                            if sender not in self.conversations:
                                self.conversations[sender] = []
                            
                            self.conversations[sender].append({
                                "role": "user",
                                "content": chat_text
                            })
                            
                            # Keep only last 10 messages for context
                            if len(self.conversations[sender]) > 10:
                                self.conversations[sender] = self.conversations[sender][-10:]
                            
                            # Get AI response
                            ai_response = await privatemode_client.chat_completion(self.conversations[sender], model)
                            
                            # Add assistant response to history
                            self.conversations[sender].append({
                                "role": "assistant", 
                                "content": ai_response
                            })
                            
                            # Send AI response
                            await c.send(f"🤖 {ai_response}")
                        else:
                            await c.send("🤖 Yes? How can I help you?")
                else:
                    await c.send("Sorry, I couldn't transcribe the voice message. Please try again or check if the audio is clear.")
                    
            except Exception as e:
                logger.error(f"Error processing voice message: {str(e)}")
                await c.send("An error occurred while processing the voice message. Please try again.")
    
    # Register the single message handler
    message_handler = MessageHandler(whisper_client=whisper_client)
    
    # Also update the clear command to use the main handler's conversations
    clear_command.chat_command.conversations = message_handler.conversations
    
    bot.register(message_handler)
    
    logger.info(f"Starting Signal bot on {signal_service} with number {phone_number}")
    logger.info(f"Using PrivateMode API at {privatemode_base_url}")
    if model:
        logger.info(f"Using model: {model}")
    logger.info("Permission system enabled")
    if whisper_client:
        logger.info(f"Voice transcription enabled with {len(whisper_client.urls)} Whisper instance(s)")
        logger.info(f"Whisper instances: {', '.join(whisper_client.urls)}")
    
    try:
        bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        # Clean up file watcher
        permission_manager.stop_file_watcher()


if __name__ == "__main__":
    main()