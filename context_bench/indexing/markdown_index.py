"""Deterministic (no-LLM) splitter: headings -> sections, oversized sections chunked by paragraph.
Also the fallback when the LLM splitter's plan cannot be validated."""
from __future__ import annotations
import re
from collections import Counter
from context_bench.analysis.markdown_parser import Document, find_headings
from context_bench.indexing.plan import SectionSpec, SplitPlan

_STOP = set("the a an of to in on at for and or but is are was were be been this that these those it its as by "
            "with from which who has have had do does did not no all any each must should shall may can will "
            "when where what how if then than also use used using section sections see note".split())
_REF = re.compile(r"(?:§\s*|[Ss]ections?\s+)(\d+(?:\.\d+)*(?:(?:,|\s+and|\s+or)\s+\d+(?:\.\d+)*)*)")
_NUM = re.compile(r"\d+(?:\.\d+)*")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _chunk_lines(doc: Document, start: int, end: int, max_tokens: int, tokenizer) -> list[tuple[int, int]]:
    """Split [start,end] at blank lines outside code fences so each chunk is <= max_tokens (best effort)."""
    if tokenizer.count(doc.slice(start, end)) <= max_tokens:
        return [(start, end)]
    cuts, fence = [], None
    for ln in range(start, end):
        raw = doc.lines[ln - 1]
        m = _FENCE.match(raw)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        if fence is None and not raw.strip():
            cuts.append(ln)
    chunks, cur_start = [], start
    for i, ln in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else end
        if tokenizer.count(doc.slice(cur_start, ln)) >= max_tokens // 2 and \
           tokenizer.count(doc.slice(cur_start, nxt)) > max_tokens:
            chunks.append((cur_start, ln))
            cur_start = ln + 1
    chunks.append((cur_start, end))
    return [c for c in chunks if c[0] <= c[1]]


def _first_sentence(body: str, max_words: int = 25) -> str:
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    body = re.sub(r"^\s*[#>|`-].*$", " ", body, flags=re.M)
    text = " ".join(body.split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    sent = (m.group(1) if m else text)
    w = sent.split()
    return " ".join(w[:max_words]) + ("…" if len(w) > max_words else "")


def _key_terms(body: str, n: int = 6) -> list[str]:
    ticks = re.findall(r"`([^`\n]{2,40})`", body)
    plain = re.sub(r"```.*?```", " ", body, flags=re.S)
    caps = [m.group(0) for m in re.finditer(r"\b[A-Z][A-Za-z0-9_]{2,}\b", plain)
            if not re.search(r"(^|[.!?:#*\-|>]\s*|\n\s*)$", plain[max(0, m.start() - 4):m.start()])
            and plain[m.start():m.end()].lower() not in _STOP]
    c = Counter(t for t in ticks + caps if t.lower() not in _STOP)
    return [t for t, _ in c.most_common(n)]


def heuristic_plan(doc: Document, tokenizer, max_section_tokens: int = 1200) -> SplitPlan:
    specs: list[SectionSpec] = []
    for sec in doc.sections:
        chunks = _chunk_lines(doc, sec.start_line, sec.end_line, max_section_tokens, tokenizer)
        for i, (a, b) in enumerate(chunks):
            body = doc.slice(a, b)
            sid = sec.id if i == 0 else f"{sec.id}-{i + 1}"
            title = sec.title if i == 0 else f"{sec.title} (part {i + 1})"
            specs.append(SectionSpec(id=sid, title=title, start_line=a, end_line=b,
                                     parent=sec.parent if i == 0 else sec.id,
                                     summary=_first_sentence(body), key_terms=_key_terms(body)))
    plan = SplitPlan(specs, "heuristic")
    _add_cross_refs(doc, plan)
    return plan


def _add_cross_refs(doc: Document, plan: SplitPlan) -> None:
    by_num = {s.doc_number: s.id for s in doc.sections if s.doc_number}
    pos_ids = {s.id for s in plan.sections}
    titles = {s.id: s.title for s in doc.sections if s.level > 0 and len(s.title) >= 8}
    for spec in plan.sections:
        body = doc.slice(spec.start_line, spec.end_line)
        found: dict[str, str] = {}
        for m in _REF.finditer(body):
            for num in _NUM.findall(m.group(1)):
                target = by_num.get(num) or (f"S{num}" if f"S{num}" in pos_ids else None)
                if target and target != spec.id and not target.startswith(spec.id + "-"):
                    found.setdefault(target, f"references section {num}")
        for tid, title in titles.items():
            if tid != spec.id and not spec.id.startswith(tid + "-") and title.lower() in body.lower():
                found.setdefault(tid, f"mentions '{title}'")
        spec.related = [{"id": t, "relation": "see_also", "reason": r}
                        for t, r in found.items() if t in pos_ids]
