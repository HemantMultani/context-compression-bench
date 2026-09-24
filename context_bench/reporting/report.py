"""Neutral, human-readable report. Does not declare a 'best' method; shows trade-offs and the
Pareto-efficient set, and lists every failed/partial test so information loss can be inspected."""
from __future__ import annotations
from context_bench.evaluation.metrics import pareto
from context_bench.evaluation.test_generator import CATEGORIES

DEFS = ("**Definitions.** `retained_ratio = compressed_tokens / original_tokens`; "
        "`reduction = 1 - retained_ratio`. Token counts use the tokenizer named below and may differ from "
        "your target model's tokenizer. `score = (pass + 0.5 x partial) / n`.")


def pct(x, d=0):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=0):
    return "-" if x is None else f"{x:,.{d}f}"


def generate_report(res: dict) -> str:
    st, rows = res["document"], res["aggregate"]["rows"]
    L: list[str] = []
    A = L.append
    A(f"# Context compression report: {res['name']}\n")
    A(f"_Generated {res['generated_at']}. Experimental benchmark: token reduction alone is not success._\n")
    # -- executive summary
    A("## Executive summary\n")
    A(f"- Original document: **{st['tokens']:,} tokens** ({st['tokenizer']}), {st['sections']} sections, "
      f"{st['lines']:,} lines.")
    A(f"- Tests: {res['tests']['total']} ({res['tests']['by_origin']}), the same tests for every representation.")
    A(f"- Representations tested: {', '.join('`' + r + '`' for r in res['representations'])}.")
    front = pareto(rows)
    for model, reps in front.items():
        A(f"- Answering model `{model}`: representations not dominated on (input tokens, judged score): "
          f"{', '.join('`' + r + '`' for r in reps) or 'n/a'}.")
    A("- No method is declared best: pick using the trade-off table below and your tolerance for lost facts.")
    A("- Judged scores come from an LLM judge and are **model-judged, not objective truth**; objective "
      "fact-match is shown alongside.\n")
    A(DEFS + "\n")
    # -- stats
    A("## Document statistics\n")
    A("| characters | words (approx) | lines | headings | sections | tokens |\n|---|---|---|---|---|---|")
    A(f"| {st['characters']:,} | {st['words_approx']:,} | {st['lines']:,} | {st['headings']} | "
      f"{st['sections']} | {st['tokens']:,} |\n")
    A("Characters, words and tokens are different measurements; only tokens approximate model cost.\n")
    # -- structure
    if res.get("index"):
        A("## Indexing\n")
        A("| variant | splitter | sections | index tokens | edges | notes |\n|---|---|---|---|---|---|")
        for k, v in res["index"].items():
            n = "; ".join(v.get("notes", []))[:160] or "-"
            src = "**FAILED - arm not evaluated**" if v["source"] == "FAILED" else v["source"]
            A(f"| {k} | {src} | {v['sections']} | {v['index_tokens']:,} | {v['edges']} | {n} |")
        A("")
    # -- compression
    A("## Compression results (objective)\n")
    A("| method | model | target retained | achieved retained | reduction | tokens | runtime |\n|---|---|---|---|---|---|---|")
    for c in res["compression"]:
        tgt = "-" if c["target_retained_ratio"] is None else f"{c['target_retained_ratio']:.2f}"
        A(f"| {c['method']} | {c['model'].split('/')[-1]} | {tgt} | "
          f"{c['retained_ratio']:.3f} | {pct(c['reduction'], 1)} | {c['original_tokens']:,} -> "
          f"{c['compressed_tokens']:,} | {c['runtime_sec']:.1f}s |")
    A("\nCompressors often miss the requested ratio; the achieved ratio is what counts. Headings and code "
      "blocks are kept verbatim. `condensed` has no target: the model decides what is redundant.")
    for c in res["compression"]:
        if c["method"] == "condensed" and c.get("params", {}).get("authored_by"):
            mm = c["params"].get("mechanical_check_missing", [])
            A(f"\n`condensed` was authored externally by {c['params']['authored_by']}; mechanical fact check: "
              + ("PASS" if not mm else f"{len(mm)} issue(s): {', '.join(mm[:8])}") + ".")
        elif c["method"] == "condensed":
            fb = c.get("fallback_sections", [])
            A(f"\n`condensed`: {len(fb)} of {len(c.get('sections', []))} sections were kept verbatim because the rewrite failed "
              f"the mechanical fact checks or was not shorter" + (f" ({', '.join(fb)})." if fb else "."))
    A("")
    # -- evaluation
    if rows:
        A("## Evaluation results\n")
        for model in sorted({r["model"] for r in rows}):
            A(f"### Answering model: `{model}`\n")
            A("| representation | context tokens | mean input tokens / question | input-token reduction | "
              "pass | partial | fail | score (judged) | score (objective) | fact survival | answered |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
            for r in [x for x in rows if x["model"] == model]:
                j, o = r["judged"], r["objective"]
                A(f"| {r['rep']} | {r['context_tokens']:,} | {num(r.get('mean_input_tokens'))} | "
                  f"{pct(r.get('input_token_reduction_vs_original'))} | {pct(j['accuracy'])} | {pct(j['partial'])} | "
                  f"{pct(j['fail'])} | {pct(j['score'])} | {pct(o['score'])} | "
                  f"{pct(r['fact_survival_in_representation'])} | {r['answered']}"
                  f"{' (+' + str(r['errors']) + ' errors)' if r['errors'] else ''} |")
            A("")
            nav = [x for x in rows if x["model"] == model and "nav_hit_any" in x]
            if nav:
                A("Navigation behaviour (does the model fetch the sections that hold the answer?):\n")
                A("| representation | fetched any needed section | fetched all needed | mean sections fetched | mean turns |\n|---|---|---|---|---|")
                for r in nav:
                    A(f"| {r['rep']} | {pct(r['nav_hit_any'])} | {pct(r['nav_hit_all'])} | "
                      f"{r['mean_sections_fetched']:.1f} | {r['mean_turns']:.1f} |")
                A("")
            agree = ", ".join(f"{r['rep']} {pct(r['judge_objective_agreement'])}" for r in rows
                              if r['model'] == model and r['judge_objective_agreement'] is not None)
            A("Objective vs judged: `fact survival` = share of each test's critical facts still present in the "
              "representation (strict phrase match; an upper bound on what any reader could recover); "
              "`score (objective)` = critical facts conveyed by the answer under a lenient number/keyword proxy "
              "(imperfect for free-form answers). "
              + (f"Judge vs objective agreement: {agree}." if agree else "No judge was run, so only objective scores exist.")
              + "\n")
        # -- paired losses
        A("## Loss versus the original (paired, per question)\n")
        A("For each arm: of the questions the ORIGINAL answered correctly, how many does the arm get wrong? Zero means "
          "no measured loss on these tests: it is **not proof** of zero loss (few questions, small model).\n")
        for model in sorted({r["model"] for r in rows}):
            A(f"### `{model}`\n")
            A("| arm | input-token reduction | original-correct questions | regressed | regressed ids |\n|---|---|---|---|---|")
            clean = []
            for r in [x for x in rows if x["model"] == model and x.get("regressions")]:
                g = r["regressions"]
                if not g["ids"]:
                    clean.append((r.get("input_token_reduction_vs_original") or 0, r["rep"]))
                A(f"| {r['rep']} | {pct(r.get('input_token_reduction_vs_original'))} | {g['n_base']} | {len(g['ids'])} | "
                  f"{', '.join(g['ids']) or '-'} |")
            A("")
            if clean:
                A("Arms with no measured regression, largest token reduction first: " +
                  ", ".join(f"`{n}` ({pct(v)})" for v, n in sorted(clean, reverse=True)) + ".\n")
            else:
                A("No arm is free of regressions on these tests.\n")
        # -- by category
        A("## Results by test category (judged score; objective in brackets)\n")
        for model in sorted({r["model"] for r in rows}):
            A(f"### `{model}`\n")
            reps = [r for r in rows if r["model"] == model]
            A("| category | " + " | ".join(r["rep"] for r in reps) + " |\n|---|" + "---|" * len(reps))
            for c, (name, _) in CATEGORIES.items():
                cells = []
                for r in reps:
                    b = r["by_category"][c]
                    cells.append(f"{pct(b['judged']['score'])} ({pct(b['objective']['score'])}) n={b['objective']['n']}")
                A(f"| {c} {name} | " + " | ".join(cells) + " |")
            A("")
    # -- failures
    A("## Failed or partially preserved facts\n")
    fails = res.get("failures", [])
    if not fails:
        A("_None recorded (or evaluation not run)._\n")
    for f in fails[: res.get("max_failures_shown", 150)]:
        A(f"**[{f['rep']}] {f['test_id']} ({f['category']}) - {f['verdict']} ({f['source']})** - model `{f['model']}`\n")
        A(f"- Question: {f['question']}")
        A(f"- Expected answer: {f['expected']}")
        A(f"- Critical facts: {'; '.join(f['facts'])}")
        A(f"- Model answer: {f['answer'] or '(empty)'}")
        A(f"- Source section(s): {', '.join(f['sections'])}")
        if f.get("reason"):
            A(f"- Judge note: {f['reason']}")
        A(f"- Representation tested: `{f['rep']}`\n")
    if len(fails) > res.get("max_failures_shown", 150):
        A(f"_{len(fails) - res.get('max_failures_shown', 150)} more in results.json._\n")
    # -- repro
    A("## Reproducibility\n")
    m = res["manifest"]
    A(f"- Document SHA-256: `{m['document_sha256']}`; tests SHA-256: `{m.get('tests_sha256', '-')}`")
    A(f"- Tokenizer: `{m['tokenizer']}`; models: {m['models']}")
    A(f"- Python {m['python']} on {m['platform']}; packages: {m['packages']}")
    A(f"- Full parameters and timestamps in `manifest.json`.\n")
    return "\n".join(L)
