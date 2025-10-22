"""
Chat commands for Signal Bot.

Provides AI chat functionality with conversation history.
"""

import logging
from signalbot import Command, Context
from ..permissions.manager import requires_permission

logger = logging.getLogger(__name__)

MAX_CONVERSATION_HISTORY = 10


@requires_permission("chat")
class ChatCommand(Command):
    def __init__(self, privatemode_client, permission_manager, model: str = None):
        self.privatemode_client = privatemode_client
        self.permission_manager = permission_manager
        self.model = model
        self.conversations = {}

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

        # Add user message to conversation history
        self.add_user_message(sender, message_text)

        # Get AI response
        response = await self.privatemode_client.chat_completion(self.conversations[sender], self.model)

        # Add assistant response to history
        self.add_assistant_message(sender, response)

        # Send response
        await c.send(response)


@requires_permission("chat")
class ClearCommand(Command):
    def __init__(self, chat_command: ChatCommand, permission_manager):
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