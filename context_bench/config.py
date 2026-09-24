"""Tiny .env loader and provider-spec parsing. No third-party deps."""
from __future__ import annotations
import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into os.environ (existing variables win). Never prints values."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split("  #")[0].strip().strip('"').strip("'")
        if v and k.strip() not in os.environ:
            os.environ[k.strip()] = v


def parse_spec(spec: str) -> tuple[str, str]:
    """'ollama:qwen2.5-coder:7b' -> ('ollama', 'qwen2.5-coder:7b'). Split on the FIRST colon only."""
    if ":" not in spec:
        raise ValueError(f"Model spec must look like 'provider:model', got {spec!r}")
    provider, model = spec.split(":", 1)
    return provider.strip().lower(), model.strip()
