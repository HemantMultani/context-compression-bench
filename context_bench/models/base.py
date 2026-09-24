"""LLM abstraction. Providers implement `_call`; `LLMClient.generate` adds disk caching, rate limiting,
retry with backoff, and the one-time EXTERNAL API notice (privacy requirement)."""
from __future__ import annotations
import hashlib, json, sys, threading, time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LLMResponse:
    text: str
    input_tokens: int | None      # as reported by the provider (may be unreliable, e.g. Ollama KV-cache reuse)
    output_tokens: int | None
    latency_sec: float
    model: str
    cached: bool = False


class TransientError(RuntimeError):
    def __init__(self, msg, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class QuotaExhausted(RuntimeError):
    """A per-day/billing quota is used up: retrying cannot help until it resets."""


_announced: set[str] = set()


class LLMClient:
    provider: str = "base"
    model: str = ""
    is_external: bool = True

    def __init__(self, cache_dir: str | Path | None = None, rpm: float | None = None, max_retries: int = 9):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_interval = 60.0 / rpm if rpm else 0.0
        self._last = 0.0
        self._lock = threading.Lock()
        self.max_retries = max_retries
        self.calls = 0
        self.cache_hits = 0

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def announce(self) -> None:
        if self.is_external and self.label not in _announced:
            _announced.add(self.label)
            print(f"[EXTERNAL API] {self.label} will receive document content and test questions.",
                  file=sys.stderr, flush=True)

    def _call(self, prompt: str, system: str | None, max_tokens: int, json_mode: bool) -> LLMResponse:
        raise NotImplementedError

    def _key(self, prompt, system, max_tokens, json_mode) -> str:
        blob = json.dumps([self.provider, self.model, system, prompt, max_tokens, json_mode])
        return hashlib.sha256(blob.encode()).hexdigest()

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 1024,
                 json_mode: bool = False) -> LLMResponse:
        key = self._key(prompt, system, max_tokens, json_mode)
        if self.cache_dir:
            f = self.cache_dir / f"{key}.json"
            if f.is_file():
                self.cache_hits += 1
                d = json.loads(f.read_text())
                d["cached"] = True
                return LLMResponse(**d)
        self.announce()
        delay = 2.0
        for attempt in range(self.max_retries):
            with self._lock:
                wait = self._min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            try:
                resp = self._call(prompt, system, max_tokens, json_mode)
                break
            except TransientError as e:
                if attempt == self.max_retries - 1:
                    raise
                sleep = e.retry_after or delay
                print(f"[retry] {self.label}: {e} (sleep {sleep:.0f}s)", file=sys.stderr, flush=True)
                time.sleep(sleep)
                delay = min(delay * 2, 45)
        self.calls += 1
        if self.cache_dir:
            (self.cache_dir / f"{key}.json").write_text(json.dumps(asdict(resp)))
        return resp
