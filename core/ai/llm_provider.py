"""
LLM provider abstraction for the platform.

This module defines a small, focused interface that wraps underlying
LLM providers (e.g., OpenAI, local models). The goal is to keep
call-sites stable while allowing you to swap providers via config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, Any, Optional
import os

from core.ai_analysis import AIAnalysisService


class ChatModel(Protocol):
    """Minimal interface for chat-style models."""

    def chat(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        ...


@dataclass
class OpenAIChatModel(ChatModel):
    """
    Thin adapter around the existing AIAnalysisService/OpenAI client.

    This keeps all OpenAI-specific configuration in one place and lets
    higher-level services depend only on the ChatModel protocol.
    """

    service: AIAnalysisService

    def chat(self, system_prompt: str, user_prompt: str, *, max_tokens: int = 512) -> str:
        if not self.service.client:
            return "AI analysis unavailable (provider not configured)."

        try:
            response = self.service.client.chat.completions.create(
                model=self.service.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.6,
            )
            return response.choices[0].message.content or ""
        except Exception as e:  # pragma: no cover - defensive
            return f"AI provider error: {str(e)[:200]}"


def get_default_chat_model() -> Optional[ChatModel]:
    """
    Return the default ChatModel instance based on environment/config.

    Currently this is just an OpenAI-backed model that reuses the
    existing AIAnalysisService. In the future you can add branches here
    for other providers or local models.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        service = AIAnalysisService()
        if not service.client:
            return None
        return OpenAIChatModel(service=service)

    # Placeholder for future providers (e.g., "local", "anthropic", etc.)
    return None


def summarize_text(text: str, *, context: str | None = None) -> str:
    """
    Convenience helper: summarize a blob of text using the default model.

    This is useful for report generation, notebook helpers, or piping
    log output into a quick natural-language explanation.
    """
    model = get_default_chat_model()
    if model is None:
        return "AI summarization unavailable (no LLM provider configured)."

    system = (
        "You are a concise financial analyst. Summarize the content for a "
        "professional user who understands markets and statistics."
    )
    if context:
        user = f"Context: {context}\n\nText:\n{text}"
    else:
        user = f"Text:\n{text}"

    return model.chat(system, user, max_tokens=256)

