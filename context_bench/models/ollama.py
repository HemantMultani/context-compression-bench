from __future__ import annotations
import os, time, requests
from .base import LLMClient, LLMResponse, TransientError


class OllamaClient(LLMClient):
    provider = "ollama"
    is_external = False

    def __init__(self, model: str, host: str | None = None, num_ctx: int = 8192, **kw):
        super().__init__(**kw)
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.num_ctx = num_ctx

    def _call(self, prompt, system, max_tokens, json_mode):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "stream": False, "keep_alive": "30m",
                "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens}}
        if json_mode:
            body["format"] = "json"
        t = time.time()
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=900)
        except requests.RequestException as e:
            raise TransientError(f"ollama connection: {e}")
        if r.status_code >= 500:
            raise TransientError(f"ollama {r.status_code}")
        r.raise_for_status()
        d = r.json()
        return LLMResponse(d["message"]["content"], d.get("prompt_eval_count"), d.get("eval_count"),
                           time.time() - t, self.model)
