"""
aichat agent runner.
Uses the aichat CLI (https://github.com/sigoden/aichat) for multi-provider LLM access.
"""

import re
import uuid
from typing import Callable

from ..config import config
from .base import AgentRunner, AgentCommand

_SENSITIVE_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "CLAUDEOR_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "COHERE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)


def clean_aichat_output(raw_output: str, original_prompt: str = "") -> str:
    """Clean aichat output."""
    if not config.aichat.clean_output:
        return raw_output
    cleaned = raw_output
    # Remove excessive newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


class AiChatRunner(AgentRunner):
    """Runner for aichat CLI."""

    def __init__(self):
        # Track session names generated during exec so parse_session_id can return them
        self._last_session_name: str | None = None

    @property
    def name(self) -> str:
        return "aichat"

    def build_exec_command(
        self,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> AgentCommand:
        """Build command for starting a new aichat session."""
        full_command = ["aichat"]

        # Model selection from config
        if config.aichat.model:
            full_command.extend(["-m", config.aichat.model])

        # Generate a session name for potential R2 resume
        session_name = f"owlex-{uuid.uuid4().hex[:12]}"
        self._last_session_name = session_name
        full_command.extend(["-s", session_name])

        return AgentCommand(
            command=full_command,
            prompt=prompt,  # Delivered via stdin
            cwd=working_directory,
            output_prefix="AiChat Output",
            not_found_hint="Please ensure aichat is installed. See: https://github.com/sigoden/aichat",
            stream=True,
            env_overrides=self._get_env_overrides(),
        )

    def build_resume_command(
        self,
        session_ref: str,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> AgentCommand:
        """Build command for resuming an existing aichat session."""
        # Validate session_ref to prevent flag injection
        if session_ref.startswith("-"):
            raise ValueError(f"Invalid session_ref: '{session_ref}' - cannot start with '-'")

        full_command = ["aichat"]

        # Model selection from config
        if config.aichat.model:
            full_command.extend(["-m", config.aichat.model])

        # Resume the named session
        full_command.extend(["-s", session_ref])

        return AgentCommand(
            command=full_command,
            prompt=prompt,  # Delivered via stdin
            cwd=working_directory,
            output_prefix="AiChat Output",
            not_found_hint="Please ensure aichat is installed. See: https://github.com/sigoden/aichat",
            stream=False,  # Non-streaming for R2 consistency
            env_overrides=self._get_env_overrides(),
        )

    def _get_env_overrides(self) -> dict[str, str] | None:
        """
        Default-safe behavior: clear common credential env vars before spawning aichat.
        Set AICHAT_ALLOW_ENV_CREDENTIALS=true to opt in to passthrough.
        """
        if config.aichat.allow_env_credentials:
            return None
        return {k: "" for k in _SENSITIVE_ENV_VARS}

    def get_output_cleaner(self) -> Callable[[str, str], str]:
        return clean_aichat_output

    async def parse_session_id(
        self,
        output: str,
        since_mtime: float | None = None,
        working_directory: str | None = None,
    ) -> str | None:
        """
        Return the session name used in the last exec command.

        aichat manages sessions by name via -s flag, so no filesystem
        discovery is needed - we just return the name we generated.
        """
        return self._last_session_name
