from __future__ import annotations
from context_bench.config import parse_spec
from .base import LLMClient, LLMResponse, TransientError  # noqa: F401


def get_client(spec: str, cache_dir=None, rpm=None, num_ctx: int = 16384, **kw) -> LLMClient:
    provider, model = parse_spec(spec)
    common = dict(cache_dir=cache_dir, rpm=rpm)
    if provider == "ollama":
        from .ollama import OllamaClient
        return OllamaClient(model, num_ctx=num_ctx, **common, **kw)
    if provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(model, **common)
    if provider in ("openai", "openai_compat"):
        from .openai_compat import OpenAICompatClient
        return OpenAICompatClient(model, **common)
    raise ValueError(f"unknown provider {provider!r} (use ollama:, gemini:, openai:)")
