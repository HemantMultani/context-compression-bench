from __future__ import annotations
import os, time
from .base import LLMClient, LLMResponse, QuotaExhausted, TransientError


class GeminiClient(LLMClient):
    provider = "gemini"
    is_external = True

    def __init__(self, model: str = "gemini-flash-latest", api_key: str | None = None, **kw):
        super().__init__(**kw)
        from google import genai
        from google.genai import types
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env or the environment).")
        self.model = model
        self._genai, self._types = genai, types
        self._client = genai.Client(api_key=key)

    def _call(self, prompt, system, max_tokens, json_mode):
        types = self._types
        kw = dict(system_instruction=system, temperature=0, max_output_tokens=max_tokens,
                  response_mime_type="application/json" if json_mode else None)
        t = time.time()
        try:
            try:
                cfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0), **kw)
                r = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
            except Exception as e:
                if "thinking" not in str(e).lower():
                    raise
                r = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=types.GenerateContentConfig(**kw))
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and ("PerDay" in msg or "per day" in msg.lower()):
                raise QuotaExhausted(f"{self.model}: daily free-tier quota exhausted (resets daily); "
                                     "use another model, wait, or use a paid key") from e
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg:
                import re
                m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"'retryDelay': '(\d+)s'", msg)
                raise TransientError(msg[:120], float(m.group(1)) + 1 if m else None)
            raise
        u = r.usage_metadata
        if not r.text and not getattr(u, "candidates_token_count", None):
            raise TransientError("empty response (no candidate returned)", 3)
        return LLMResponse(r.text or "", getattr(u, "prompt_token_count", None),
                           getattr(u, "candidates_token_count", None), time.time() - t, self.model)
