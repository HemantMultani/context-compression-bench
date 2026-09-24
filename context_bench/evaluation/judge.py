"""LLM judge: labels each answer pass/partial/fail against the expected answer and critical facts.
It never sees the representation. Results are MODEL-JUDGED, not ground truth; the objective
fact-match (string containment of critical facts in the answer) is reported next to it."""
from __future__ import annotations
import json, re
import re
from context_bench.evaluation.test_generator import norm

_STOPW = set("a an the of to in on at for and or but is are was were be been this that these those it its as by with from "
             "which who has have had do does did not no all any each must should shall may can will when where what how if "
             "then than also use used using only more most less than".split())


def fact_in_answer(fact: str, answer: str) -> bool:
    """Lenient objective check that an ANSWER conveys a critical fact (a proxy, not truth).

    Answers are free-form and often terse ("8" for "at most 8 downstream calls"), so exact phrase matching
    would under-count. If the fact contains numbers, all of them must appear in the answer; otherwise at
    least 70% of its content words (compared by 5-letter stems) must appear. Numbers matched in the wrong
    context are a known weakness, which is why judged and objective scores are both reported."""
    toks = lambda x: [t for t in (w.strip(".,:;") for w in norm(x).split()) if t]   # "15." must equal "15"
    ft, at = toks(fact), toks(answer)
    if not ft:
        return False
    a_set = set(at)
    nums = [t for t in ft if re.search(r"\d", t)]
    if nums:
        return all(n in a_set for n in nums)
    stems = {t[:5] for t in ft if t not in _STOPW}
    if not stems:
        return norm(fact) in norm(answer)
    a_stems = {w[:5] for w in at}
    return len(stems & a_stems) / len(stems) >= 0.7

SYSTEM = "You are a strict, fair grader. You output only valid JSON."
PROMPT = """Grade each candidate answer against the reference.

For each item:
- "pass": the candidate states ALL critical facts correctly and does not contradict the reference.
- "partial": it states some but not all critical facts, or is right but omits a required condition/exception.
- "fail": it is wrong, contradicts the reference, misses the key facts, or says it cannot find the answer (e.g. NOT FOUND).
Judge on meaning, not wording. Do not reward extra unrelated information. Do not use outside knowledge.

Return JSON: {{"grades": [{{"id": "<item id>", "verdict": "pass|partial|fail", "reason": "<max 15 words>"}}]}}

ITEMS:
{items}
"""


def objective_verdict(answer: str, test: dict) -> tuple[str, float]:
    facts = test["critical_facts"]
    hit = sum(fact_in_answer(f, answer) for f in facts)
    r = hit / len(facts)
    return ("pass" if r == 1 else "partial" if r > 0 else "fail"), r


def judge_answers(answers, tests_by_id: dict, client, batch: int = 16, log=print, on_batch=None) -> dict:
    """Return {(model,rep,test_id): {"verdict","reason"}} for answers without errors."""
    todo = [a for a in answers if a.error is None]
    out: dict[tuple, dict] = {}
    failures = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        items = []
        for j, a in enumerate(chunk):
            t = tests_by_id[a.test_id]
            items.append({"id": str(j), "question": t["question"], "reference_answer": t["expected_answer"],
                          "critical_facts": t["critical_facts"], "candidate_answer": a.answer[:1500]})
        try:
            resp = client.generate(PROMPT.format(items=json.dumps(items, indent=1)), system=SYSTEM,
                                   max_tokens=1500, json_mode=True)
            grades = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip()))["grades"]
        except Exception as e:                                # noqa: BLE001
            log(f"[judge] batch {i // batch}: failed ({type(e).__name__}: {str(e)[:100]}); left ungraded")
            failures += 1
            if failures >= 2:
                log("[judge] two batches failed; stopping. Re-run `evaluate`/`run` later: graded batches are "
                    "saved and cached, only the missing ones will be asked again.")
                break
            continue
        failures = 0
        for g in grades:
            try:
                a = chunk[int(g["id"])]
            except (KeyError, ValueError, IndexError):
                continue
            v = str(g.get("verdict", "")).lower()
            if v in ("pass", "partial", "fail"):
                out[(a.model, a.rep, a.test_id)] = {"verdict": v, "reason": str(g.get("reason", ""))}
        if on_batch:
            on_batch(out)          # persist progress: a quota failure mid-way must not lose earlier grades
    return out
