"""Conversation module."""

from .conversation_ai import (
    DEFAULT_FALLBACK_RESPONSE,
    DEFAULT_SYSTEM_INSTRUCTION,
    ConversationAI,
    ConversationManager,
    ConversationTurn,
)
from .gemini_client import (
    DEFAULT_GEMINI_MODEL,
    GeminiClient,
    GeminiClientError,
    GeminiProvider,
    GeminiProviderError,
)
from .intent_classifier import ConversationIntent, classify_intent
from .llm_provider import LLMContext, LLMMessage, LLMProvider, LLMProviderError
from .provider_factory import (
    DEFAULT_CONVERSATION_PROVIDER,
    ConversationProviderConfigurationError,
    create_conversation_manager,
    create_llm_provider,
)

__all__ = [
    "ConversationAI",
    "ConversationManager",
    "ConversationTurn",
    "ConversationIntent",
    "ConversationProviderConfigurationError",
    "DEFAULT_CONVERSATION_PROVIDER",
    "DEFAULT_FALLBACK_RESPONSE",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_SYSTEM_INSTRUCTION",
    "GeminiClient",
    "GeminiClientError",
    "GeminiProvider",
    "GeminiProviderError",
    "LLMContext",
    "LLMMessage",
    "LLMProvider",
    "LLMProviderError",
    "classify_intent",
    "create_conversation_manager",
    "create_llm_provider",
]
