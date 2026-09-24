import pytest
from context_bench.compression.base import _segments, compress_document, make_result
from context_bench.compression import _hf
from conftest import FakeCompressor


def test_segments_cover_document_exactly(small_doc):
    for cc in (False, True):
        segs = _segments(small_doc, cc)
        assert "".join(t for _, t in segs) == small_doc.text


def test_headings_and_code_kept_verbatim(small_doc, tok):
    res = compress_document(small_doc, FakeCompressor(), tok, 0.5)
    for h in ("# Handbook", "## 1. Alpha Rules", "## 2. Beta Rules", "### 2.1 Beta Exceptions"):
        assert h + "\n" in res.text
    assert "```python\n# not a heading\nx = 1\n```" in res.text
    assert res.compressed_tokens < res.original_tokens


def test_compress_code_flag(small_doc, tok):
    off = compress_document(small_doc, FakeCompressor(), tok, 0.4, compress_code=False)
    on = compress_document(small_doc, FakeCompressor(), tok, 0.4, compress_code=True)
    assert "x = 1\n```" in off.text                      # code fence untouched by default
    assert on.params["compress_code"] is True and on.text != off.text


def test_ratio_definitions_and_fields(small_doc, tok):
    res = compress_document(small_doc, FakeCompressor(), tok, 0.7)
    assert res.retained_ratio == pytest.approx(res.compressed_tokens / res.original_tokens)
    assert res.reduction == pytest.approx(1 - res.retained_ratio)
    assert res.original_tokens == tok.count(small_doc.text)
    assert res.compressed_tokens == tok.count(res.text)
    meta = res.meta()
    for k in ("method", "model", "target_retained_ratio", "original_tokens", "compressed_tokens", "retained_ratio",
              "reduction", "runtime_sec", "tokenizer", "params"):
        assert k in meta
    assert "text" not in meta and res.runtime_sec >= 0


def test_prose_rate_is_adjusted_for_verbatim_parts(synth_doc, tok):
    res = compress_document(synth_doc, FakeCompressor(), tok, 0.8)
    assert res.prose_rate_requested < 0.8 < 1.0
    assert abs(res.retained_ratio - 0.8) < 0.08


def test_ratio_one_changes_nothing(small_doc, tok):
    assert compress_document(small_doc, FakeCompressor(), tok, 1.0).text == small_doc.text


def test_make_result_zero_original():
    assert make_result("m", "x", 0.5, 0, 0, 0.0, "t", {}).retained_ratio == 1.0


def test_missing_dependency_message(monkeypatch):
    import builtins
    real = builtins.__import__
    def fake(name, *a, **k):
        if name == "llmlingua":
            raise ImportError("no llmlingua")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(_hf.MissingDependency, match="pip install"):
        _hf.require_llmlingua()


def test_uncached_model_is_never_downloaded_silently(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    with pytest.raises(_hf.ModelNotCached, match="--allow-download"):
        _hf.check_model_cached("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", allow_download=False)
    _hf.check_model_cached("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", allow_download=True)
