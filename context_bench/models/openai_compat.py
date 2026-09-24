"""Any OpenAI-style /chat/completions endpoint (Azure OpenAI gateway, LiteLLM, vLLM, company proxy).
Configure with OPENAI_BASE_URL (e.g. https://host/v1) and OPENAI_API_KEY. Extra header via
OPENAI_EXTRA_HEADERS='{"api-key":"..."}' if your gateway needs it."""
from __future__ import annotations
import json, os, time, requests
from .base import LLMClient, LLMResponse, TransientError


class OpenAICompatClient(LLMClient):
    provider = "openai"
    is_external = True

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, **kw):
        super().__init__(**kw)
        self.model = model
        self.base = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            self.headers["Authorization"] = f"Bearer {key}"
        if os.environ.get("OPENAI_EXTRA_HEADERS"):
            self.headers.update(json.loads(os.environ["OPENAI_EXTRA_HEADERS"]))

    def _call(self, prompt, system, max_tokens, json_mode):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "temperature": 0, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t = time.time()
        for _ in range(2):
            try:
                r = requests.post(f"{self.base}/chat/completions", json=body, headers=self.headers, timeout=300)
            except requests.RequestException as e:
                raise TransientError(f"connection: {e}")
            if r.status_code == 400 and "max_tokens" in r.text and "max_tokens" in body:
                body["max_completion_tokens"] = body.pop("max_tokens"); continue   # newer models
            if r.status_code == 400 and "response_format" in body:
                body.pop("response_format"); continue                               # gateway lacks JSON mode
            break
        if r.status_code in (429, 500, 502, 503, 504):
            ra = r.headers.get("retry-after")
            raise TransientError(f"http {r.status_code}", float(ra) if ra and ra.isdigit() else None)
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        return LLMResponse(d["choices"][0]["message"]["content"] or "", u.get("prompt_tokens"),
                           u.get("completion_tokens"), time.time() - t, self.model)
