"""
AI/LLM service abstractions.

This package provides a thin, provider-agnostic layer over concrete
LLM providers (e.g., OpenAI, local models). Existing AI analysis
modules should depend on these interfaces rather than directly on
third-party SDKs where possible.
"""

