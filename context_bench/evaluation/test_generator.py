"""Test-case generation from the ORIGINAL document, with mechanical grounding.

Every test carries critical_facts that must be short phrases actually present in the source sections
it cites; tests failing that check are rejected (not silently kept), so the answer key can't contain
invented facts. Same schema for LLM-generated, manually written and draft (no-LLM) tests."""
from __future__ import annotations
import json, math, re
from collections import Counter

CATEGORIES = {
    "A": ("direct_recall", "Direct factual recall: the answer is stated explicitly in one place."),
    "B": ("rule_application", "Rule application: describe a concrete situation; the answer is which documented rule applies and what it requires."),
    "C": ("exception_handling", "Exception handling: ask about a documented exception or edge case and the condition it applies under."),
    "D": ("multi_section", "Multi-section reasoning: the answer needs facts from TWO OR MORE different sections."),
    "E": ("negative_constraint", "Negative / constraint: ask what must NOT be done or what is forbidden or never allowed."),
    "F": ("terminology_entity", "Terminology / entity: ask what a project-specific name, identifier, term or component is or means."),
    "G": ("numerical_exact", "Numerical / exact value: ask for an exact threshold, limit, duration, version or configuration value."),
}
DIFFICULTIES = ("easy", "medium", "hard")
REQUIRED = ("id", "category", "question", "expected_answer", "source_sections", "critical_facts", "difficulty")

SYSTEM = "You write rigorous evaluation questions from documents. You output only valid JSON."
PROMPT = """Below are sections of a document, each starting with its [ID]. Write {k} test question(s) of this category:

CATEGORY {cat}: {desc}

REQUIREMENTS
- The question must be answerable ONLY from the sections below and must test retrieval of specific facts, rules, constraints, exceptions or procedures. Do NOT ask for a summary or opinions.
- The question must be self-contained: never say "according to section X" or refer to ids.
- expected_answer: concise, and derived strictly from the document. Do not invent facts.
- critical_facts: 1-4 SHORT phrases (values, names, conditions) copied EXACTLY, character for character, from the document, that a correct answer must contain.
- source_sections: the [ID]s the answer is drawn from.
- difficulty: easy | medium | hard.
{extra}
Return JSON exactly like:
{{"tests": [{{"question": "...", "expected_answer": "...", "source_sections": ["S3"], "critical_facts": ["..."], "difficulty": "easy"}}]}}

SECTIONS:
{sections}
"""


def norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[`*_]", "", s)
    return " ".join(re.sub(r"[^\w\s./:%+\-<>=]", " ", s).split())


def fact_in_text(fact: str, text: str, threshold: float = 0.8) -> bool:
    """Exact (normalised) substring, or >=threshold of the fact's tokens present in the text."""
    nf, nt = norm(fact), norm(text)
    if not nf:
        return False
    if nf in nt:
        return True
    toks = nf.split()
    if len(toks) < 3:
        return False
    have = set(nt.split())
    return sum(t in have for t in toks) / len(toks) >= threshold


def validate_test(t: dict) -> list[str]:
    errs = [f"missing field {k}" for k in REQUIRED if k not in t]
    if errs:
        return errs
    if t["category"] not in CATEGORIES:
        errs.append(f"bad category {t['category']!r}")
    if t["difficulty"] not in DIFFICULTIES:
        errs.append(f"bad difficulty {t['difficulty']!r}")
    for k in ("question", "expected_answer"):
        if not isinstance(t[k], str) or not t[k].strip():
            errs.append(f"{k} must be a non-empty string")
    for k in ("source_sections", "critical_facts"):
        if not isinstance(t[k], list) or not t[k] or not all(isinstance(x, str) and x.strip() for x in t[k]):
            errs.append(f"{k} must be a non-empty list of strings")
    return errs


def validate_tests(tests: list[dict]) -> list[str]:
    out = []
    for t in tests:
        out += [f"{t.get('id', '?')}: {e}" for e in validate_test(t)]
    return out


def ground_check(t: dict, section_texts: dict[str, str]) -> list[str]:
    errs = []
    unknown = [s for s in t["source_sections"] if s not in section_texts]
    if unknown:
        return [f"unknown source sections {unknown}"]
    src = "\n".join(section_texts[s] for s in t["source_sections"])
    for f in t["critical_facts"]:
        if not fact_in_text(f, src):
            errs.append(f"critical fact not found in cited sections: {f!r}")
    if t["category"] == "G" and not any(re.search(r"\d", f) for f in t["critical_facts"]):
        errs.append("numerical test has no digit in its critical facts")
    if t["category"] == "D" and len(set(t["source_sections"])) < 2:
        errs.append("multi-section test cites fewer than 2 sections")
    return errs


def _batches(plan_sections, section_texts, tokenizer, max_tokens):
    batches, cur, cur_t = [], [], 0
    for s in plan_sections:
        n = tokenizer.count(section_texts[s.id])
        if cur and cur_t + n > max_tokens:
            batches.append(cur); cur, cur_t = [], 0
        cur.append(s); cur_t += n
    if cur:
        batches.append(cur)
    return batches


def generate_tests(plan, section_texts: dict[str, str], tokenizer, client, num_tests: int = 35,
                   batch_tokens: int = 24_000, log=print) -> tuple[list[dict], list[dict]]:
    """Return (accepted, rejected). One LLM call per (category x batch)."""
    cats = list(CATEGORIES)
    per_cat = [num_tests // len(cats) + (1 if i < num_tests % len(cats) else 0) for i in range(len(cats))]
    batches = _batches(plan.sections, section_texts, tokenizer, batch_tokens)
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    counters: Counter = Counter()
    for cat, want in zip(cats, per_cat):
        name, desc = CATEGORIES[cat]
        for bi, batch in enumerate(batches):
            k = max(1, math.ceil(want / len(batches)))
            body = "\n\n".join(f"[{s.id}] {s.title}\n{section_texts[s.id]}" for s in batch)
            extra = "- Multi-section: cite at least two different section IDs.\n" if cat == "D" else ""
            if accepted:   # help the model avoid repeats
                extra += "- Avoid these existing questions:\n" + "\n".join(
                    f"  * {t['question']}" for t in accepted[-12:]) + "\n"
            resp = client.generate(PROMPT.format(k=k + 1, cat=cat, desc=desc, extra=extra, sections=body),
                                   system=SYSTEM, max_tokens=4000, json_mode=True)
            try:
                items = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip()))["tests"]
            except Exception as e:                                # noqa: BLE001
                log(f"[generator] category {cat}: unparsable response ({e})")
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                t = {"id": "", "category": cat, "question": it.get("question", ""),
                     "expected_answer": it.get("expected_answer", ""),
                     "source_sections": [str(x) for x in it.get("source_sections", [])],
                     "critical_facts": [str(x) for x in it.get("critical_facts", [])],
                     "difficulty": it.get("difficulty", "medium"), "origin": "generated"}
                errs = validate_test({**t, "id": "tmp"}) or ground_check(t, section_texts)
                key = norm(t["question"])
                if not errs and key in seen:
                    errs = ["duplicate question"]
                if errs:
                    rejected.append({**t, "reasons": errs}); continue
                if sum(1 for a in accepted if a["category"] == cat) >= want:
                    continue
                seen.add(key)
                counters[cat] += 1
                t["id"] = f"{cat}-{counters[cat]:02d}"
                accepted.append(t)
    return accepted, rejected


_CAND = re.compile(r"\b(must not|must|never|shall|only|except|at most|at least|forbidden|required|always|"
                   r"not allowed|no more than|maximum|minimum)\b|\d", re.I)


def draft_tests(plan, section_texts: dict[str, str]) -> list[dict]:
    """No-LLM fallback: candidate facts (rule/number sentences) for a human to turn into questions."""
    out, n = [], 0
    for s in plan.sections:
        for sent in re.split(r"(?<=[.!?])\s+", " ".join(section_texts[s.id].split())):
            if len(sent) > 25 and _CAND.search(sent) and not sent.startswith("#"):
                n += 1
                nums = re.findall(r"`[^`]+`|\b\d[\d,.:%]*\b(?:\s?[a-zA-Z]+)?", sent)[:4]
                out.append({"id": f"DRAFT-{n:03d}", "category": "A", "question": "TODO: turn this rule into a question",
                            "expected_answer": sent, "source_sections": [s.id],
                            "critical_facts": nums or [sent[:60]], "difficulty": "medium", "origin": "draft"})
    return out
