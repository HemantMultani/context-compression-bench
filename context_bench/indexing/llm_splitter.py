"""LLM-planned split. The model decides WHERE to cut and writes routing metadata; the tool slices the
original text by line number, so the model can never alter or drop content.

Validation: exact cover of all lines, unique ids, valid related targets. One repair retry with the
error list; if it still fails, fall back to the deterministic heuristic splitter (recorded in the plan)."""
from __future__ import annotations
import json, re

from context_bench.analysis.markdown_parser import Document, find_headings
from context_bench.indexing.markdown_index import heuristic_plan, _chunk_lines
from context_bench.indexing.plan import SectionSpec, SplitPlan, normalize_and_validate, RELATIONS

SYSTEM = "You are a meticulous document architect. You output only valid JSON."

class SplitFailed(RuntimeError):
    """The LLM could not produce a valid split plan (only raised with strict=True)."""


PROMPT = """You are indexing a document so that an AI agent can navigate it like a graph: read a short index first, then fetch only the sections it needs.

The document has already been divided into {n_blocks} numbered blocks, [B1] to [B{n_blocks}] (a block is a heading or a paragraph/list/code block). Group CONSECUTIVE blocks into sections.

RULES
1. Sections must be contiguous ranges of blocks that cover EVERY block from B1 to B{n_blocks} exactly once, in order (no gaps, no overlaps). The first section starts at block 1 and the last ends at block {n_blocks}.
2. Cut at semantic boundaries (a topic, rule group, procedure). Never cut inside a code block, table, or list. Prefer sections of roughly {min_tok}-{max_tok} tokens; very short adjacent topics may be merged, very long topics must be split.
3. Existing headings are hints, not law: the document may have missing, flat or misleading headings. Use your own judgment about the real structure.
4. `parent` expresses hierarchy (id of the enclosing section, or null for top level). Parents must be sections in your plan.
5. `summary`: one factual sentence (max 30 words) saying what the section contains. Do not invent content.
6. `use_when`: which kind of task or question should make an agent read this section.
7. `key_terms`: up to 8 exact names, identifiers or concepts from the section.
8. `related`: ONLY when the text of this section explicitly depends on, is an exception to, or overrides another section, or points to it. Use relation one of: {relations}. Give a short reason.
9. Ids are "S1", "S2", ... in document order.

OUTPUT: a JSON object exactly of this shape and nothing else (start_block and end_block are block numbers, plain integers):
{{"sections": [{{"id": "S1", "title": "...", "start_block": 1, "end_block": 6, "parent": null,
  "summary": "...", "use_when": "...", "key_terms": ["..."],
  "related": [{{"id": "S4", "relation": "exception_to", "reason": "..."}}]}}]}}

DOCUMENT BLOCKS:
{numbered}
"""


_FENCE = re.compile(r"^\s*(```|~~~)")


def make_blocks(doc: Document) -> list[tuple[int, int]]:
    """Blocks = headings and blank-line-separated paragraphs (fences kept whole). They cover every
    line exactly once: block i runs from its first line to the line before the next block starts."""
    heads = {ln for ln, _, _ in find_headings(doc.lines)}
    starts, fence, prev_blank = [], None, True
    for i, raw in enumerate(doc.lines, 1):
        m = _FENCE.match(raw)
        blank = not raw.strip()
        if fence is None and not blank and (prev_blank or i in heads):
            starts.append(i)
        elif fence is None and i in heads:
            starts.append(i)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        prev_blank = blank and fence is None
    if not starts or starts[0] != 1:
        starts.insert(0, 1)
    starts = sorted(set(starts))
    return [(a, (starts[k + 1] - 1) if k + 1 < len(starts) else len(doc.lines)) for k, a in enumerate(starts)]


def _numbered(doc: Document, blocks, lo: int, hi: int) -> str:
    """Render blocks lo..hi (1-based, inclusive) as '[B12] text'."""
    return "\n".join(f"[B{i}] " + doc.slice(*blocks[i - 1]).rstrip("\n") for i in range(lo, hi + 1))


def _parse_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _windows(doc: Document, blocks, tokenizer, window_tokens: int) -> list[tuple[int, int]]:
    """Windows of consecutive blocks (1-based, inclusive) of about window_tokens each."""
    wins, lo, cur = [], 1, 0
    for i, (a, b) in enumerate(blocks, 1):
        n = tokenizer.count(doc.slice(a, b))
        if cur and cur + n > window_tokens:
            wins.append((lo, i - 1)); lo, cur = i, 0
        cur += n
    wins.append((lo, len(blocks)))
    return wins


def _block_errors(data, lo: int, hi: int) -> list[str]:
    """Exact-cover check in block space (this is what a model can actually see and count)."""
    errs, prev = [], lo - 1
    secs = sorted(data["sections"], key=lambda s: int(s["start_block"]))
    for s in secs:
        a, b = int(s["start_block"]), int(s["end_block"])
        if not (lo <= a <= b <= hi):
            errs.append(f"{s.get('id')}: blocks {a}-{b} outside {lo}..{hi}")
        elif a != prev + 1:
            errs.append(f"{s.get('id')}: starts at B{a} but previous section ended at B{prev}"
                        + (" (gap)" if a > prev + 1 else " (overlap)"))
        prev = max(prev, b)
    if prev != hi:
        errs.append(f"sections end at B{prev} but the document has {hi} blocks; the last section must end at B{hi}")
    return errs


def _to_specs(data, blocks) -> list[SectionSpec]:
    specs = []
    for s in data["sections"]:
        a, b = int(s["start_block"]), int(s["end_block"])
        specs.append(SectionSpec(
            id=str(s["id"]), title=str(s.get("title", s["id"])), start_line=blocks[a - 1][0],
            end_line=blocks[b - 1][1], parent=s.get("parent") or None,
            summary=str(s.get("summary", "")), use_when=str(s.get("use_when", "")),
            key_terms=list(s.get("key_terms") or []), related=list(s.get("related") or [])))
    return specs


def _plan_window(client, doc, blocks, lo, hi, tokenizer, min_tok, max_tok, prefix, raw, notes, max_out):
    n = hi - lo + 1
    body = _numbered(doc, blocks, lo, hi)
    prompt = PROMPT.format(n_blocks=hi, min_tok=min_tok, max_tok=max_tok, relations=", ".join(RELATIONS),
                           numbered=body)
    if lo > 1:
        prompt += (f"\nNOTE: this is only part of a longer document. Blocks run from B{lo} to B{hi}: your "
                   f"first section starts at B{lo} and your last ends at B{hi}.")
    errors: list[str] = []
    resp_text = ""
    for attempt in range(2):
        p = prompt if attempt == 0 else (
            prompt + "\n\nYOUR PREVIOUS ANSWER WAS INVALID:\n" + "\n".join(f"- {e}" for e in errors) +
            f"\n\nPrevious answer:\n{resp_text}\n\nReturn the full corrected JSON.")
        resp_text = client.generate(p, system=SYSTEM, max_tokens=max_out, json_mode=True).text
        raw.append(resp_text)
        try:
            data = _parse_json(resp_text)
            errors = _block_errors(data, lo, hi)
            if errors:
                continue
            specs = _to_specs(data, blocks)
        except Exception as e:                                # noqa: BLE001
            errors = [f"response was not valid plan JSON: {type(e).__name__}: {e}"]
            continue
        for sp in specs:
            sp.id = f"{prefix}{sp.id}" if prefix else sp.id
            if sp.parent:
                sp.parent = f"{prefix}{sp.parent}" if prefix else sp.parent
            for r in sp.related:
                if isinstance(r, dict) and r.get("id") and prefix:
                    r["id"] = f"{prefix}{r['id']}"
        return specs
    raise ValueError("; ".join(errors[:6]))


def llm_plan(doc: Document, tokenizer, client, min_tokens: int = 300, max_tokens: int = 1500,
             window_tokens: int = 100_000, max_out_tokens: int = 12_000, strict: bool = False) -> SplitPlan:
    """strict=True raises SplitFailed instead of falling back to the heuristic splitter (used by experiments,
    so an 'LLM index' arm can never silently be a heading-based index)."""
    raw: list[str] = []
    notes: list[str] = []
    try:
        specs: list[SectionSpec] = []
        blocks = make_blocks(doc)
        wins = _windows(doc, blocks, tokenizer, window_tokens)
        if len(wins) > 1:
            notes.append(f"document exceeded window ({window_tokens} tokens): planned in {len(wins)} windows; "
                         "cross-window related links come from heuristic references only")
        for wi, (a, b) in enumerate(wins, 1):
            specs += _plan_window(client, doc, blocks, a, b, tokenizer, min_tokens, max_tokens,
                                  f"W{wi}-" if len(wins) > 1 else "", raw, notes, max_out_tokens)
        plan = SplitPlan(specs, "llm", notes, raw)
        errs = normalize_and_validate(plan, len(doc.lines))
        if errs:
            raise ValueError("; ".join(errs[:6]))
    except Exception as e:                                    # noqa: BLE001
        if strict:
            raise SplitFailed(str(e)) from e
        import sys
        print(f"[WARNING] LLM split plan unusable ({str(e)[:160]}); falling back to the heuristic splitter.",
              file=sys.stderr, flush=True)
        fb = heuristic_plan(doc, tokenizer, max_tokens)
        fb.source = "heuristic-fallback"
        fb.notes = notes + [f"LLM plan rejected after retry: {e}; used heuristic splitter"]
        fb.raw = raw
        return fb
    _chunk_oversized(doc, plan, tokenizer, max_tokens)
    return plan


def _chunk_oversized(doc: Document, plan: SplitPlan, tokenizer, max_tokens: int) -> None:
    out: list[SectionSpec] = []
    for s in plan.sections:
        chunks = _chunk_lines(doc, s.start_line, s.end_line, int(max_tokens * 1.5), tokenizer)
        if len(chunks) == 1:
            out.append(s); continue
        plan.notes.append(f"{s.id} exceeded {int(max_tokens * 1.5)} tokens; chunked into {len(chunks)} parts")
        for i, (a, b) in enumerate(chunks):
            out.append(SectionSpec(id=s.id if i == 0 else f"{s.id}-{i + 1}",
                                   title=s.title if i == 0 else f"{s.title} (part {i + 1})",
                                   start_line=a, end_line=b, parent=s.parent if i == 0 else s.id,
                                   summary=s.summary, use_when=s.use_when, key_terms=s.key_terms,
                                   related=s.related if i == 0 else []))
    plan.sections = out
