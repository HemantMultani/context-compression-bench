"""Split plan: the contract between splitters (heuristic or LLM) and the mechanical slicer.

A splitter only decides boundaries + routing metadata. Text is always sliced from the original by
line numbers, so section files concatenate back to the original byte-for-byte.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field, asdict
from pathlib import Path

RELATIONS = ("depends_on", "exception_to", "see_also", "overrides")


@dataclass
class SectionSpec:
    id: str
    title: str
    start_line: int
    end_line: int
    parent: str | None = None
    summary: str = ""
    use_when: str = ""
    key_terms: list[str] = field(default_factory=list)
    related: list[dict] = field(default_factory=list)   # {"id","relation","reason"}


@dataclass
class SplitPlan:
    sections: list[SectionSpec]
    source: str                      # "heuristic" | "llm" | "heuristic-fallback"
    notes: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)   # raw LLM responses, kept for reproducibility

    def by_id(self) -> dict[str, SectionSpec]:
        return {s.id: s for s in self.sections}

    def to_json(self) -> dict:
        return {"source": self.source, "notes": self.notes, "raw": self.raw,
                "sections": [asdict(s) for s in self.sections]}

    @classmethod
    def from_json(cls, d: dict) -> "SplitPlan":
        return cls([SectionSpec(**s) for s in d["sections"]], d.get("source", "?"), d.get("notes", []), d.get("raw", []))


def normalize_and_validate(plan: SplitPlan, n_lines: int) -> list[str]:
    """Repair harmless problems in place (order, small gaps, first/last line) and return the
    list of remaining errors. Empty list == plan is a valid exact cover of lines 1..n_lines."""
    errors: list[str] = []
    secs = plan.sections
    if not secs:
        return ["plan has no sections"]
    ids = [s.id for s in secs]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        errors.append(f"duplicate section ids: {dup}")
    for s in secs:
        if not isinstance(s.start_line, int) or not isinstance(s.end_line, int):
            errors.append(f"{s.id}: start_line/end_line must be integers")
    if errors:
        return errors
    secs.sort(key=lambda s: s.start_line)
    for s in secs:
        if s.end_line < s.start_line:
            errors.append(f"{s.id}: end_line {s.end_line} < start_line {s.start_line}")
        if s.start_line < 1 or s.end_line > n_lines:
            errors.append(f"{s.id}: lines {s.start_line}-{s.end_line} outside 1..{n_lines}")
    if errors:
        return errors
    # repair gaps (attach to previous section) ; overlaps are errors
    if secs[0].start_line != 1:
        plan.notes.append(f"repaired: first section started at line {secs[0].start_line}; extended to 1")
        secs[0].start_line = 1
    for prev, cur in zip(secs, secs[1:]):
        if cur.start_line <= prev.end_line:
            errors.append(f"overlap: {prev.id} ends at {prev.end_line} but {cur.id} starts at {cur.start_line}")
        elif cur.start_line > prev.end_line + 1:
            plan.notes.append(f"repaired: gap lines {prev.end_line + 1}-{cur.start_line - 1} attached to {prev.id}")
            prev.end_line = cur.start_line - 1
    if secs[-1].end_line != n_lines and not errors:
        plan.notes.append(f"repaired: last section ended at {secs[-1].end_line}; extended to {n_lines}")
        secs[-1].end_line = n_lines
    idset = set(ids)
    for s in secs:
        if s.parent is not None and s.parent not in idset:
            plan.notes.append(f"repaired: {s.id} had unknown parent {s.parent!r}; cleared")
            s.parent = None
        clean = []
        for r in s.related or []:
            if isinstance(r, dict) and r.get("id") in idset and r.get("id") != s.id:
                if r.get("relation") not in RELATIONS:
                    r["relation"] = "see_also"
                clean.append({"id": r["id"], "relation": r["relation"], "reason": str(r.get("reason", ""))})
            else:
                plan.notes.append(f"repaired: dropped related entry {r!r} on {s.id} (unknown target)")
        s.related = clean
        s.key_terms = [str(k) for k in (s.key_terms or [])]
    return errors


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "section"


def section_text(doc, spec: SectionSpec) -> str:
    return doc.slice(spec.start_line, spec.end_line)


def _tree_lines(plan: SplitPlan) -> list[tuple[int, SectionSpec]]:
    children: dict[str | None, list[SectionSpec]] = {}
    for s in plan.sections:
        children.setdefault(s.parent, []).append(s)
    out: list[tuple[int, SectionSpec]] = []

    def walk(pid, depth):
        for s in children.get(pid, []):
            out.append((depth, s))
            walk(s.id, depth + 1)
    walk(None, 0)
    return out


def render_index(plan: SplitPlan, doc, tokenizer, title: str = "Context Index") -> str:
    """Compact routing document for an agent: one entry per section, indented by hierarchy.
    Never includes section bodies (only a one-line summary); line ranges live in graph.json."""
    parts = [f"# {title}",
             "Navigation index. Each entry: [id] title (size in tokens): what it covers. Request sections by id; "
             "do not assume content that is not listed here.",
             ""]
    for depth, s in _tree_lines(plan):
        toks = tokenizer.count(section_text(doc, s))
        line = f"{'  ' * depth}- [{s.id}] {s.title} ({toks} tok)"
        if s.summary:
            line += f": {s.summary}"
        if s.use_when:
            line += f" Use when: {s.use_when.rstrip('.')}."
        if s.key_terms:
            line += f" Terms: {', '.join(s.key_terms)}."
        if s.related:
            line += " Related: " + ", ".join(f"{r['id']} ({r['relation']})" for r in s.related) + "."
        parts.append(line)
    return "\n".join(parts).rstrip() + "\n"


def write_index_dir(out_dir: Path, doc, plan: SplitPlan, tokenizer) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "sections").mkdir(parents=True, exist_ok=True)
    for old in (out_dir / "sections").glob("*.md"):
        old.unlink()
    nodes, edges = [], []
    for s in plan.sections:
        body = section_text(doc, s)
        (out_dir / "sections" / f"{s.id}-{slug(s.title)}.md").write_text(body)
        nodes.append({"id": s.id, "title": s.title, "tokens": tokenizer.count(body),
                      "lines": [s.start_line, s.end_line], "summary": s.summary,
                      "use_when": s.use_when, "key_terms": s.key_terms})
        if s.parent:
            edges.append({"from": s.parent, "to": s.id, "type": "parent_of"})
        for r in s.related:
            edges.append({"from": s.id, "to": r["id"], "type": r["relation"], "reason": r.get("reason", "")})
    index_md = render_index(plan, doc, tokenizer)
    (out_dir / "index.md").write_text(index_md)
    (out_dir / "graph.json").write_text(json.dumps({"nodes": nodes, "edges": edges}, indent=2))
    (out_dir / "split_plan.json").write_text(json.dumps(plan.to_json(), indent=2))
    return {"sections": len(plan.sections), "index_tokens": tokenizer.count(index_md),
            "source": plan.source, "edges": len(edges)}
