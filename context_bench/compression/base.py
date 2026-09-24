"""Compression interface + structure-aware document driver.

Definitions used everywhere (and repeated in the report):
    retained_ratio = compressed_tokens / original_tokens
    reduction      = 1 - retained_ratio
The requested target is a *retained* ratio for the WHOLE document. Headings and code fences are kept
verbatim, so the prose is compressed harder to hit the whole-document target; the achieved ratio is
always measured afterwards and may differ from the request (LLMLingua often misses its target).
"""
from __future__ import annotations
import re, time
from dataclasses import dataclass, field, asdict
from typing import Protocol

from context_bench.analysis.markdown_parser import Document

_FENCE = re.compile(r"^\s*(```|~~~)")


class Compressor(Protocol):
    name: str
    model: str
    params: dict

    def compress(self, text: str, rate: float) -> str: ...


@dataclass
class CompressionResult:
    method: str
    model: str
    target_retained_ratio: float
    original_tokens: int
    compressed_tokens: int
    retained_ratio: float
    reduction: float
    runtime_sec: float
    tokenizer: str
    params: dict = field(default_factory=dict)
    verbatim_tokens: int = 0
    prose_tokens_before: int = 0
    prose_tokens_after: int = 0
    prose_rate_requested: float = 1.0
    text: str = ""

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def make_result(method, model, target, orig_tokens, comp_tokens, runtime, tokenizer_name, params, **kw):
    ratio = comp_tokens / orig_tokens if orig_tokens else 1.0
    return CompressionResult(method, model, target, orig_tokens, comp_tokens, ratio, 1 - ratio,
                             runtime, tokenizer_name, params, **kw)


def _segments(doc: Document, compress_code: bool):
    """Yield (kind, text) with kind in {'verbatim','prose'} covering the document exactly, in order."""
    fence = None
    buf: list[str] = []
    kind = "prose"

    def flush():
        nonlocal buf
        if buf:
            yield_list.append((kind, "".join(buf)))
            buf = []

    yield_list: list[tuple[str, str]] = []
    heading_lines = {s.start_line for s in doc.sections if s.level > 0}
    for i, raw in enumerate(doc.lines, 1):
        m = _FENCE.match(raw)
        if i in heading_lines and fence is None:
            flush(); kind = "verbatim"; buf.append(raw); flush(); kind = "prose"
            continue
        if m:
            if fence is None:
                flush(); fence = m.group(1); kind = "prose" if compress_code else "verbatim"
                buf.append(raw)
            else:
                buf.append(raw)
                if fence == m.group(1):
                    fence = None
                    flush(); kind = "prose"
            continue
        if fence is None and not raw.strip() and kind == "prose":
            buf.append(raw)          # blank lines stay attached to prose runs
            continue
        buf.append(raw)
    flush()
    return yield_list


def compress_document(doc: Document, compressor: Compressor, tokenizer, target_ratio: float,
                      compress_code: bool = False) -> CompressionResult:
    orig_tokens = tokenizer.count(doc.text)
    segs = _segments(doc, compress_code)
    assert "".join(t for _, t in segs) == doc.text, "segmenter must cover the document exactly"
    # Budget against the SUM of segment token counts (tokenisation at segment boundaries can differ
    # slightly from the whole-document count), so a target of 1.0 is exactly a no-op.
    compressible = [t for k, t in segs if k == "prose" and t.strip()]
    prose = sum(tokenizer.count(t) for t in compressible)
    seg_total = sum(tokenizer.count(t) for _, t in segs)
    verb = seg_total - prose
    budget = target_ratio * seg_total - verb
    prose_rate = 1.0 if prose == 0 or target_ratio >= 1.0 else min(1.0, max(0.05, budget / prose))
    out, t0 = [], time.perf_counter()
    for kind, text in segs:
        if kind == "prose" and text.strip() and prose_rate < 1.0:
            lead = text[: len(text) - len(text.lstrip("\n"))]
            trail = "\n" * (len(text) - len(text.rstrip("\n")))
            body = compressor.compress(text.strip("\n"), prose_rate).strip("\n")
            out.append(lead + body + trail)
        else:
            out.append(text)
    runtime = time.perf_counter() - t0
    result_text = "".join(out)
    comp_tokens = tokenizer.count(result_text)
    return make_result(compressor.name, compressor.model, target_ratio, orig_tokens, comp_tokens, runtime,
                       tokenizer.name, {**compressor.params, "compress_code": compress_code},
                       verbatim_tokens=verb, prose_tokens_before=prose,
                       prose_tokens_after=comp_tokens - verb, prose_rate_requested=prose_rate,
                       text=result_text)
