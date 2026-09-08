"""Provider-neutral contracts for OmniCare conversation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class LLMProviderError(RuntimeError):
    """A provider failure that is safe for ConversationManager to handle."""

    def __init__(self, message, code="provider_error"):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class LLMMessage:
    role: str
    text: str


@dataclass(frozen=True)
class LLMContext:
    intent: str
    recent_conversation: tuple[LLMMessage, ...]
    user_text: str
    system_instruction: str


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def generate_response(self, context: LLMContext) -> str:
        """Generate one complete response for the supplied conversation context."""

    def close(self) -> None:
        """Release provider resources, if any."""

