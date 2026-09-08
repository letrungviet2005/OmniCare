"""Configuration boundary for selecting an OmniCare LLM provider."""

from __future__ import annotations

import os

from .gemini_client import GeminiProvider


DEFAULT_CONVERSATION_PROVIDER = "gemini"


class ConversationProviderConfigurationError(ValueError):
    pass


def create_llm_provider(provider_name=None, **kwargs):
    selected = (
        provider_name
        if provider_name is not None
        else os.getenv("CONVERSATION_PROVIDER", DEFAULT_CONVERSATION_PROVIDER)
    )
    normalized = str(selected or "").strip().casefold()
    if normalized == "gemini":
        return GeminiProvider(**kwargs)
    raise ConversationProviderConfigurationError(
        f"Unsupported CONVERSATION_PROVIDER '{selected}'. Supported providers: gemini."
    )


def create_conversation_manager(provider_name=None, **provider_kwargs):
    from .conversation_ai import ConversationManager

    return ConversationManager(
        provider=create_llm_provider(provider_name, **provider_kwargs)
    )
