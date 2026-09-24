"""Condense arm: an LLM rewrites each section into terse, agent-readable form (key: value, `cond -> rule`,
short bullets, no filler prose) WITHOUT dropping information.

Unlike LLMLingua (which forces a fixed drop rate per 512-token chunk regardless of density) the model
decides what is redundant, so dense sections shrink less. Nothing is trusted blindly: every section's output
must pass mechanical checks (all numbers, identifiers, code spans, acronyms, quoted strings, headings and
condition/negation markers from the original are still present). One repair retry with the missing items,
then the section is kept VERBATIM. So hard facts are preserved by construction; what can still be lost is
nuance the checks cannot see, which is what the tests measure."""
from __future__ import annotations
import re, time

SYSTEM = "You are a precise technical editor. You output only the rewritten text."

PROMPT = """Rewrite the section below so that an AI coding agent can read it using as few tokens as possible, WITHOUT losing any information.

The reader is a capable agent, not a human. It does not need full sentences, politeness, transitions, restated examples or repeated explanations. It DOES need every rule, value, condition, exception, identifier and cross-reference.

RULES
- Keep EVERY number, unit, version, identifier, name, path, code span, acronym and quoted string exactly as written.
- Keep every condition and exception ("only when", "unless", "except", "never", "must not", "at most", "at least"). Make them explicit with markers such as NEVER, MUST, ONLY, EXCEPT, UNLESS, IF ... THEN, or "->".
- Prefer compact forms: `key: value`, `condition -> requirement`, short bullets. State a repeated rule once.
- Keep every heading line (lines starting with #) exactly as written, in the same order.
- Keep each number on the same line as the words it quantifies (e.g. "tables > 10 million rows: CONCURRENTLY"). Never list numbers on their own line.
- Keep every distinct rule, including "how to correct" and "who is not exempt" clauses. Do not add anything that is not in the section. Do not explain what you did. Output only the rewritten section.
{feedback}
SECTION:
{text}
"""

_NUM = re.compile(r"\d+(?:[.,:/\-]\d+)*")
_CODE = re.compile(r"`([^`\n]+)`")
_IDENT = re.compile(r"(?<![\w])(?:[A-Za-z]+(?:[_./\-][A-Za-z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+|/[A-Za-z][\w/\-]*)(?![\w])")
_ACRO = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")
_QUOTE = re.compile(r'"([^"\n]{2,60})"')
_NEG_ORIG = re.compile(r"\b(never|must not|not allowed|forbidden|cannot|may not|prohibited|do not|does not)\b", re.I)
_NEG_OK = re.compile(r"\b(never|not|no|forbid\w*|prohibit\w*|cannot|without)\b|n't|[!¬≠✗✘]", re.I)
_COND_ORIG = re.compile(r"\b(only|except|unless|exception)\b", re.I)
_COND_OK = re.compile(r"\b(only|except|unless|exception|if|when|but|otherwise)\b|->|→|=>|\bexcl", re.I)


def _n(s: str) -> str:
    return re.sub(r"[`*]", "", s.lower())


def hard_tokens(text: str) -> set[str]:
    """Facts that must survive verbatim (lower-cased, markup stripped)."""
    plain = text
    out = set(_NUM.findall(plain))
    out |= {m.strip() for m in _CODE.findall(plain)}
    out |= set(_IDENT.findall(plain))
    out |= set(_ACRO.findall(plain))
    out |= {m.strip() for m in _QUOTE.findall(plain)}
    return {_n(t).strip(" .,:;") for t in out if len(t.strip(" .,:;")) >= 1}


def missing_facts(original: str, condensed: str) -> list[str]:
    """Mechanical loss check. Empty list == every hard fact, heading and marker is still present."""
    cn = _n(condensed)
    miss = sorted(t for t in hard_tokens(original) if t not in cn)
    for line in original.splitlines():
        if re.match(r"^#{1,6}\s", line) and line.strip() not in condensed:
            miss.append(f"heading: {line.strip()}")
    miss += orphaned_numbers(original, condensed)
    if _NEG_ORIG.search(original) and not _NEG_OK.search(condensed):
        miss.append("negation (never / must not / forbidden)")
    if _COND_ORIG.search(original) and not _COND_OK.search(condensed):
        miss.append("condition/exception marker (only / except / unless)")
    return miss


_STOP = set("a an the of to in on at for and or but is are was were be been this that these those it its as by with from "
            "which who has have had do does did not no all any each must should shall may can will when where what how if "
            "then than also use used using only more most less than into over under per via etc".split())
_WORD = re.compile(r"\d+(?:\.\d+)*|[a-z_][a-z0-9_]*")


def _stems(words) -> set[str]:
    return {w[:5] for w in words if len(w) >= 3 and w not in _STOP and not w[0].isdigit()}


def orphaned_numbers(original: str, condensed: str, radius: int = 3) -> list[str]:
    """A number must keep its meaning: at least one of the content words next to it in the original must appear on
    the same line as it in the rewrite (lines with real content of their own are given the benefit of the doubt, so
    synonym swaps are not flagged). Catches numbers dumped on a stray line to satisfy the fact check (e.g.
    "10; 7.3"), which passes a plain presence check but has lost what the number quantifies."""
    ow = _WORD.findall(_n(original))
    cond_lines = [_WORD.findall(_n(l)) for l in condensed.splitlines()]
    out = []
    for n in sorted({w for w in ow if w[0].isdigit()}):
        neigh = set()
        for i, w in enumerate(ow):
            if w == n:
                neigh |= _stems(ow[max(0, i - radius):i] + ow[i + 1:i + 1 + radius])
        if not neigh:
            continue
        lines = [ln for ln in cond_lines if n in ln]
        # flag only "stray" lines: no neighbouring word AND almost no other content (a synonym swap such as
        # "3 times" -> "3 retries" is fine as long as the line still says what the number is about)
        if lines and not any((neigh & _stems(ln)) or len(_stems(ln)) >= 2 for ln in lines):
            out.append(f"number {n} lost its context (was next to: {', '.join(sorted(neigh)[:4])})")
    return out


def _clean(out: str) -> str:
    out = out.strip()
    m = re.fullmatch(r"```(?:markdown|md|text)?\n(.*)\n```", out, re.S)
    return (m.group(1) if m else out).strip()


def condense_section(client, text: str, tok, min_tokens: int = 80, log=print) -> tuple[str, dict]:
    n0 = tok.count(text)
    if n0 < min_tokens:
        return text, {"status": "skipped_small", "orig_tokens": n0, "new_tokens": n0}
    feedback, missing = "", []
    for attempt in (1, 2):
        prompt = PROMPT.format(feedback=feedback, text=text.strip("\n"))
        try:
            out = _clean(client.generate(prompt, system=SYSTEM, max_tokens=int(n0 * 1.3) + 120).text)
        except Exception as e:                                # noqa: BLE001
            return text, {"status": "fallback_verbatim", "orig_tokens": n0, "new_tokens": n0, "attempts": attempt,
                          "missing": [f"model error: {type(e).__name__}: {str(e)[:120]}"]}
        missing = missing_facts(text, out)
        n1 = tok.count(out)
        if not missing and n1 < n0:
            return out, {"status": "ok", "orig_tokens": n0, "new_tokens": n1, "attempts": attempt}
        if not missing:
            missing = ["output was not shorter than the original"]
        feedback = ("\nYOUR PREVIOUS ATTEMPT WAS REJECTED. It " +
                    ("dropped these items that MUST appear exactly: " + "; ".join(missing[:25])
                     if any(m != "output was not shorter than the original" for m in missing)
                     else "was not shorter than the original") + ". Fix this: keep them all and be shorter.\n")
    return text, {"status": "fallback_verbatim", "orig_tokens": n0, "new_tokens": n0, "attempts": 2, "missing": missing[:25]}


def condense_document(doc, units, client, tok, min_tokens: int = 80, log=print) -> tuple[str, dict]:
    """units: sections with start_line/end_line/id covering the document in order (an index plan's sections)."""
    t0, pieces, rows = time.perf_counter(), [], []
    for i, u in enumerate(units, 1):
        text = doc.slice(u.start_line, u.end_line)
        out, info = condense_section(client, text, tok, min_tokens, log)
        info["id"] = u.id
        rows.append(info)
        pieces.append(out.rstrip("\n") + "\n\n")
        log(f"[condense] {i}/{len(units)} {u.id}: {info['status']} {info['orig_tokens']} -> {info['new_tokens']} tok"
            + (f" (missing: {', '.join(info.get('missing', [])[:4])})" if info["status"] == "fallback_verbatim" else ""))
    result = "".join(pieces).rstrip("\n") + "\n"
    return result, {"sections": rows, "runtime_sec": time.perf_counter() - t0,
                    "fallback_sections": [r["id"] for r in rows if r["status"] == "fallback_verbatim"]}
