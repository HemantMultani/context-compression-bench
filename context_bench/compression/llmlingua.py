"""Original LLMLingua (v1): drops low-perplexity tokens under a causal LM.
Default scorer is GPT-2 (small, fast); the paper used LLaMA-7B. Override with --v1-model.
Question-agnostic settings only (no question/instruction), sentence/context filters off."""
from __future__ import annotations
from ._hf import require_llmlingua, check_model_cached, pick_device

DEFAULT_MODEL = "openai-community/gpt2"


class LLMLinguaCompressor:
    name = "llmlingua"

    def __init__(self, model: str = DEFAULT_MODEL, allow_download: bool = False, device: str | None = None):
        require_llmlingua()
        check_model_cached(model, allow_download)
        from llmlingua import PromptCompressor
        self.model = model
        self.params = {"use_sentence_level_filter": False, "use_context_level_filter": False,
                       "device": device or pick_device()}
        self._pc = PromptCompressor(model_name=model, device_map=self.params["device"])

    def compress(self, text: str, rate: float) -> str:
        r = self._pc.compress_prompt([text], instruction="", question="", rate=rate,
                                     use_sentence_level_filter=False, use_context_level_filter=False,
                                     condition_compare=False)
        return r["compressed_prompt"]


def get_compressor(method: str, **kw):
    if method == "llmlingua":
        return LLMLinguaCompressor(**kw)
    if method == "llmlingua2":
        from .llmlingua2 import LLMLingua2Compressor
        return LLMLingua2Compressor(**kw)
    raise ValueError(f"unknown compression method {method!r}")
