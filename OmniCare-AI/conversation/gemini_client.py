"""Small synchronous adapter around the official Google Gen AI SDK."""

from __future__ import annotations

import os


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class GeminiClientError(RuntimeError):
    """Raised when Gemini cannot provide a usable response."""


class GeminiClient:
    def __init__(
        self,
        api_key=None,
        model=None,
        timeout_seconds=15,
        client=None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._client = client
        self._owns_client = client is None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise GeminiClientError("GEMINI_API_KEY is not configured")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiClientError("google-genai is not installed") from exc
        try:
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.timeout_seconds * 1000)
                ),
            )
        except Exception as exc:
            raise GeminiClientError("Gemini client initialization failed") from exc
        return self._client

    def generate(self, contents, system_instruction):
        client = self._ensure_client()
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    "system_instruction": system_instruction,
                    "temperature": 0.4,
                    "max_output_tokens": 120,
                    "automatic_function_calling": {"disable": True},
                },
            )
            text = getattr(response, "text", None)
        except Exception as exc:
            raise GeminiClientError("Gemini request failed") from exc
        if not isinstance(text, str) or not text.strip():
            raise GeminiClientError("Gemini returned an empty response")
        return " ".join(text.split())

    def close(self):
        if not self._owns_client or self._client is None:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
