"""Replaceable tokenizer. Default: tiktoken o200k_base (OpenAI-compatible baseline).

The final consuming model may tokenize differently, so every report states which tokenizer
produced the numbers. Use --tokenizer tiktoken:cl100k_base or hf:<model-name> to swap.
"""
from __future__ import annotations
from typing import Protocol


class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class TiktokenTokenizer:
    def __init__(self, encoding: str = "o200k_base"):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


class HFTokenizer:
    def __init__(self, model: str):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(model)
        self.name = f"hf:{model}"

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))


class CharApproxTokenizer:
    """Test/dev fallback only (~4 chars per token). Never a substitute for a real tokenizer."""
    name = "approx:chars/4"

    def count(self, text: str) -> int:
        return (len(text) + 3) // 4


def get_tokenizer(spec: str = "tiktoken:o200k_base") -> Tokenizer:
    kind, _, arg = spec.partition(":")
    if kind == "tiktoken":
        return TiktokenTokenizer(arg or "o200k_base")
    if kind == "hf":
        return HFTokenizer(arg)
    if kind == "approx":
        return CharApproxTokenizer()
    raise ValueError(f"Unknown tokenizer spec {spec!r}")
