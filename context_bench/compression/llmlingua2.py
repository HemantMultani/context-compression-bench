"""LLMLingua-2: BERT-style token classifier (keep/drop per token), distilled from GPT-4 (Microsoft).
API checked against llmlingua==0.2.x: PromptCompressor(model_name, use_llmlingua2=True, device_map),
compress_prompt(context, rate, force_tokens, drop_consecutive, force_reserve_digit)."""
from __future__ import annotations
from ._hf import require_llmlingua, check_model_cached, pick_device

DEFAULT_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
FORCE_TOKENS = ["\n", ".", "?", "!"]


class LLMLingua2Compressor:
    name = "llmlingua2"

    def __init__(self, model: str = DEFAULT_MODEL, preserve_digits: bool = False,
                 allow_download: bool = False, device: str | None = None):
        require_llmlingua()
        check_model_cached(model, allow_download)
        from llmlingua import PromptCompressor
        self.model = model
        self.params = {"preserve_digits": preserve_digits, "force_tokens": FORCE_TOKENS,
                       "drop_consecutive": True, "device": device or pick_device()}
        self._pc = PromptCompressor(model_name=model, use_llmlingua2=True, device_map=self.params["device"])
        self._digits = preserve_digits

    def compress(self, text: str, rate: float) -> str:
        r = self._pc.compress_prompt(text, rate=rate, force_tokens=FORCE_TOKENS,
                                     drop_consecutive=True, force_reserve_digit=self._digits)
        return r["compressed_prompt"]
