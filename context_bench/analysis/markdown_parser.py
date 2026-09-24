"""Markdown heading parser: ATX headings only, ignores headings inside ``` / ~~~ fences.

Each Section owns the lines from its heading up to the next heading of ANY level (`own`), which
makes sections a flat, contiguous cover of the document. Hierarchy is kept via parent/children.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_DOCNUM = re.compile(r"^(?:§\s*)?(\d+(?:\.\d+)*)[.):]?\s+")


@dataclass
class Section:
    id: str
    title: str
    level: int                 # 0 for preamble
    start_line: int            # 1-based, inclusive
    end_line: int              # 1-based, inclusive
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    doc_number: str | None = None   # number written in the heading, e.g. "7.3"

    @property
    def n_lines(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass
class Document:
    text: str
    lines: list[str]           # with line endings preserved
    sections: list[Section]

    def slice(self, start_line: int, end_line: int) -> str:
        return "".join(self.lines[start_line - 1:end_line])

    def section_text(self, sec: Section) -> str:
        return self.slice(sec.start_line, sec.end_line)

    def by_id(self) -> dict[str, Section]:
        return {s.id: s for s in self.sections}

    @property
    def heading_count(self) -> int:
        return sum(1 for s in self.sections if s.level > 0)


def find_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return (line_no_1based, level, title) for real headings (outside code fences)."""
    out, fence = [], None
    for i, raw in enumerate(lines, 1):
        line = raw.rstrip("\r\n")
        m = _FENCE.match(line)
        if m:
            marker = m.group(1)
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence:
            continue
        h = _HEADING.match(line)
        if h:
            out.append((i, len(h.group(1)), h.group(2).strip()))
    return out


def parse(text: str) -> Document:
    lines = text.splitlines(keepends=True)
    heads = find_headings(lines)
    sections: list[Section] = []
    if not lines:
        return Document(text, lines, sections)

    first = heads[0][0] if heads else len(lines) + 1
    if first > 1 and "".join(lines[:first - 1]).strip():
        sections.append(Section("S0", "Preamble", 0, 1, first - 1))

    # A lone H1 that opens the document is the document title: it becomes S0 and the remaining
    # headings are numbered from S1, so ids line up with numbered headings ("## 9. Logging" -> S9).
    h1s = [h for h in heads if h[1] == 1]
    title_mode = len(h1s) == 1 and heads[0][1] == 1 and not sections
    counters = [0] * 7
    stack: list[tuple[Section, int]] = []
    for idx, (ln, level, title) in enumerate(heads):
        end = (heads[idx + 1][0] - 1) if idx + 1 < len(heads) else len(lines)
        m = _DOCNUM.match(title)
        if title_mode and idx == 0:
            sections.append(Section("S0", title, level, ln, end, None, doc_number=None))
            continue
        eff = level - 1 if title_mode else level
        counters[eff] += 1
        for l in range(eff + 1, 7):
            counters[l] = 0
        while stack and stack[-1][1] >= eff:
            stack.pop()
        parent = stack[-1][0] if stack else None
        sid = f"{parent.id}.{counters[eff]}" if parent else f"S{counters[eff]}"
        sec = Section(sid, title, level, ln, end, parent.id if parent else None,
                      doc_number=m.group(1) if m else None)
        if parent:
            parent.children.append(sid)
        sections.append(sec)
        stack.append((sec, eff))
    # heading-less prefix lines belonging to no section (blank only) are attached to first section
    if sections and sections[0].start_line > 1:
        sections[0].start_line = 1
    return Document(text, lines, sections)


def stats(doc: Document, tokenizer) -> dict:
    text = doc.text
    return {
        "characters": len(text),
        "words_approx": len(text.split()),
        "lines": len(doc.lines),
        "headings": doc.heading_count,
        "sections": len(doc.sections),
        "tokens": tokenizer.count(text),
        "tokenizer": tokenizer.name,
    }
