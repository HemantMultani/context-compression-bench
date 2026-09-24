"""Shared helpers: dependency detection, model-cache check (never download silently), device choice."""
from __future__ import annotations


class MissingDependency(RuntimeError):
    pass


class ModelNotCached(RuntimeError):
    pass


APPROX_SIZE = {
    "microsoft/llmlingua-2-xlm-roberta-large-meetingbank": "~2.2 GB",
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank": "~0.7 GB",
    "openai-community/gpt2": "~0.5 GB",
    "NousResearch/Llama-2-7b-hf": "~13 GB",
}


def require_llmlingua():
    try:
        import llmlingua  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise MissingDependency(
            f"LLMLingua is not installed ({e}). Install with:\n  pip install -e '.[compress]'\n"
            "(pins transformers==4.44.2, which the llmlingua package was written against)") from e


def check_model_cached(repo_id: str, allow_download: bool) -> None:
    from huggingface_hub import try_to_load_from_cache
    hit = try_to_load_from_cache(repo_id, "config.json")
    if isinstance(hit, str) or allow_download:
        return
    raise ModelNotCached(
        f"Model '{repo_id}' ({APPROX_SIZE.get(repo_id, 'size unknown')}) is not in the local Hugging Face "
        f"cache and would need to be downloaded from huggingface.co.\nRe-run with --allow-download to "
        f"permit it (only the model files are downloaded; your document is not uploaded).")


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
