from __future__ import annotations
from context_bench.analysis.markdown_parser import Document
from .markdown_index import heuristic_plan
from .llm_splitter import SplitFailed, llm_plan
from .plan import SplitPlan, normalize_and_validate


def build_plan(doc: Document, tokenizer, client=None, max_section_tokens: int = 1200, strict: bool = False,
               **kw) -> SplitPlan:
    """LLM-planned split when a client is given, deterministic heuristic otherwise."""
    if client is None:
        plan = heuristic_plan(doc, tokenizer, max_section_tokens)
    else:
        plan = llm_plan(doc, tokenizer, client, max_tokens=max_section_tokens + 300, strict=strict, **kw)
    errs = normalize_and_validate(plan, len(doc.lines))
    if errs:
        raise ValueError(f"invalid split plan: {errs}")
    return plan
