"""
Administrative commands for Signal Bot.

Provides user management, role management, and system information commands.
"""

import logging
from signalbot import Command, Context
from ..permissions.manager import requires_permission

logger = logging.getLogger(__name__)


@requires_permission("models")
class ModelsCommand(Command):
    def __init__(self, privatemode_client, permission_manager):
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
    def __init__(self, permission_manager):
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
    def __init__(self, permission_manager):
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
            await c.send(f"Added user {user_name} ({phone}) with role '{role}'")
        else:
            await c.send(f"Failed to add user. Role '{role}' may not exist.")


@requires_permission("admin")
class RemoveUserCommand(Command):
    def __init__(self, permission_manager):
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
            await c.send(f"User {phone} not found")
            return

        # Prevent removing yourself
        if phone == c.message.source:
            await c.send("You cannot remove yourself")
            return

        # Remove user
        user_name = users[phone].get("name", phone)
        if self.permission_manager.remove_user(phone):
            await c.send(f"Removed user {user_name} ({phone})")
        else:
            await c.send(f"Failed to remove user {phone}")


@requires_permission("admin")
class ListUsersCommand(Command):
    def __init__(self, permission_manager):
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
    def __init__(self, permission_manager):
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