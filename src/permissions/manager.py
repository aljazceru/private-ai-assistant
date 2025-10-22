"""
Permission manager for Signal Bot.

Provides role-based access control with file watching capabilities.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

if TYPE_CHECKING:
    from signalbot import Context

logger = logging.getLogger(__name__)


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

        async def wrapped_handle(self, c):
            from signalbot import Context  # Import here to avoid circular imports
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