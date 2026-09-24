"""Aggregate answers into the metrics the report shows.

Objective (deterministic): token counts, retained ratio, fact survival in the representation,
fact match in the answer, navigation hit rates.  Model-judged: pass/partial/fail from the judge.
score = (pass + 0.5 * partial) / n.  Definitions: retained_ratio = tokens / original_tokens;
reduction = 1 - retained_ratio."""
from __future__ import annotations
from collections import defaultdict
from statistics import mean

from context_bench.evaluation.judge import objective_verdict
from context_bench.evaluation.test_generator import CATEGORIES, fact_in_text


def _rates(verdicts: list[str]) -> dict:
    n = len(verdicts)
    if not n:
        return {"n": 0, "accuracy": None, "partial": None, "fail": None, "score": None}
    p, pa, f = (verdicts.count(x) for x in ("pass", "partial", "fail"))
    return {"n": n, "accuracy": p / n, "partial": pa / n, "fail": f / n, "score": (p + 0.5 * pa) / n}


def _overlap(a, b) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def nav_coverage(rep, test, fetched: list[str]) -> tuple[bool, bool]:
    """(hit_any, hit_all): did the fetched sections cover the lines holding the answer? Uses line ranges so it is
    comparable across different indexes; falls back to id equality for tests without source_lines."""
    src = test.get("source_lines")
    got = [rep.ranges[i] for i in fetched if i in rep.ranges]
    if src and rep.ranges:
        return (any(_overlap(s, g) for s in src for g in got),
                all(any(_overlap(s, g) for g in got) for s in src))
    ids = set(test["source_sections"])
    return bool(ids & set(fetched)), ids <= set(fetched)


def fact_survival(rep, test) -> float:
    txt = rep.all_text()
    return mean(fact_in_text(f, txt) for f in test["critical_facts"])


def aggregate(answers, judgments: dict, tests: list[dict], reps: list, tok, original_tokens: int) -> dict:
    tests_by_id = {t["id"]: t for t in tests}
    reps_by_name = {r.name: r for r in reps}
    groups = defaultdict(list)
    for a in answers:
        groups[(a.model, a.rep)].append(a)
    rows = []
    for (model, rname), ans in sorted(groups.items()):
        rep = reps_by_name[rname]
        ok = [a for a in ans if a.error is None]
        row = {"model": model, "rep": rname, "answered": len(ok), "errors": len(ans) - len(ok),
               "context_tokens": rep.context_tokens(tok),
               "context_retained_ratio": rep.context_tokens(tok) / original_tokens,
               "context_reduction": 1 - rep.context_tokens(tok) / original_tokens}
        if ok:
            row.update(mean_output_tokens=mean(a.output_tokens for a in ok),
                       mean_latency_sec=mean(a.latency_sec for a in ok), mean_turns=mean(a.turns for a in ok))
            if any(a.input_tokens for a in ok):      # imported answers have no measured token usage
                row.update(mean_input_tokens=mean(a.input_tokens for a in ok),
                           mean_total_tokens=mean(a.input_tokens + a.output_tokens for a in ok))
        # objective
        obj = [objective_verdict(a.answer, tests_by_id[a.test_id])[0] for a in ok]
        row["objective"] = _rates(obj)
        row["objective_fact_match"] = mean(objective_verdict(a.answer, tests_by_id[a.test_id])[1] for a in ok) if ok else None
        row["fact_survival_in_representation"] = mean(fact_survival(rep, tests_by_id[a.test_id]) for a in ok) if ok else None
        # judged
        jv = [judgments[(a.model, a.rep, a.test_id)]["verdict"] for a in ok if (a.model, a.rep, a.test_id) in judgments]
        row["judged"] = _rates(jv)
        both = [(judgments[k]["verdict"], objective_verdict(a.answer, tests_by_id[a.test_id])[0])
                for a in ok for k in [(a.model, a.rep, a.test_id)] if k in judgments]
        row["judge_objective_agreement"] = mean(j == o for j, o in both) if both else None
        if rep.kind == "navigate" and ok and any(a.turns for a in ok):      # skipped when navigation was not recorded
            cov = [nav_coverage(rep, tests_by_id[a.test_id], a.fetched) for a in ok]
            row["nav_hit_any"] = mean(c[0] for c in cov)
            row["nav_hit_all"] = mean(c[1] for c in cov)
            row["mean_sections_fetched"] = mean(len(a.fetched) for a in ok)
        # by category (judged if available else objective)
        bycat = {}
        for c in CATEGORIES:
            cs = [a for a in ok if tests_by_id[a.test_id]["category"] == c]
            js = [judgments[(a.model, a.rep, a.test_id)]["verdict"] for a in cs if (a.model, a.rep, a.test_id) in judgments]
            os_ = [objective_verdict(a.answer, tests_by_id[a.test_id])[0] for a in cs]
            bycat[c] = {"judged": _rates(js), "objective": _rates(os_)}
        row["by_category"] = bycat
        rows.append(row)
    # paired losses: questions the ORIGINAL answered correctly that this arm does not (judged verdict if graded, else objective)
    verd = {}
    for a in answers:
        if a.error is None:
            k = (a.model, a.rep, a.test_id)
            verd[k] = judgments[k]["verdict"] if k in judgments else objective_verdict(a.answer, tests_by_id[a.test_id])[0]
    for r in rows:
        if r["rep"] == "original":
            r["regressions"] = None
            continue
        base = [tid for (m, rep, tid), v in verd.items()
                if m == r["model"] and rep == "original" and v == "pass" and (m, r["rep"], tid) in verd]
        r["regressions"] = {"n_base": len(base),
                            "ids": sorted(t for t in base if verd[(r["model"], r["rep"], t)] != "pass")}
    # per-question input-token reduction relative to the 'original' representation, same model
    base = {r["model"]: r.get("mean_input_tokens") for r in rows if r["rep"] == "original"}
    for r in rows:
        b = base.get(r["model"])
        if b and r.get("mean_input_tokens") is not None:
            r["input_token_reduction_vs_original"] = 1 - r["mean_input_tokens"] / b
    return {"rows": rows}


def pareto(rows: list[dict], score_key: str | None = None) -> dict[str, list[str]]:
    """Per model: representations not dominated in (mean_input_tokens lower, score higher).
    Uses judged scores when the model has any, otherwise the objective scores."""
    out = {}
    for model in sorted({r["model"] for r in rows}):
        key = score_key or ("judged" if any(r["model"] == model and r["judged"]["score"] is not None for r in rows) else "objective")
        rs = [r for r in rows if r["model"] == model and r.get("mean_input_tokens") is not None
              and r[key]["score"] is not None]
        score_key_ = key
        front = []
        for r in rs:
            dominated = any(o is not r and o["mean_input_tokens"] <= r["mean_input_tokens"]
                            and o[score_key_]["score"] >= r[score_key_]["score"]
                            and (o["mean_input_tokens"] < r["mean_input_tokens"] or o[score_key_]["score"] > r[score_key_]["score"])
                            for o in rs)
            if not dominated:
                front.append(r["rep"])
        out[model] = front
    return out
