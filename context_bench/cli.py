"""context-compression-bench CLI.  python -m context_bench <command> ...

Nothing leaves your machine unless you pass a non-local model spec (gemini:, openai:); such calls print
an [EXTERNAL API] notice. ollama: models run locally. Analysis, indexing, compression and manual
evaluation need no API at all."""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

from context_bench.analysis.tokenizer import get_tokenizer
from context_bench.compression._hf import MissingDependency, ModelNotCached
from context_bench.config import load_env, parse_spec
from context_bench.experiment import DEFAULT_RATIO, DEFAULT_RATIOS, Experiment
from context_bench.models import get_client


def _common(p):
    p.add_argument("doc", help="path to a Markdown/text document")
    p.add_argument("--root", default="experiments", help="experiments directory (default: experiments/)")
    p.add_argument("--tokenizer", default="tiktoken:o200k_base", help="tiktoken:<enc> | hf:<model>")
    p.add_argument("--env", default=".env", help=".env file with API keys (default: .env)")


def _ratio_args(p):
    p.add_argument("--target-ratio", type=float, default=None,
                   help="a single retained ratio = compressed/original tokens (overrides the default sweep)")
    p.add_argument("--target-tokens", type=int, default=None, help="alternative: target compressed token count")
    p.add_argument("--ratios", default=None,
                   help="comma list of retained ratios to compare (default 0.9,0.8,0.7: LLMLingua needs a rate, so we "
                        "sweep it and let the results show how much can be cut)")


def _model_args(p, index=True):
    if index:
        p.add_argument("--index-model", default=None, help="LLM that plans the split, e.g. gemini:gemini-flash-latest")
    p.add_argument("--max-section-tokens", type=int, default=1200)
    p.add_argument("--rpm", type=float, default=8, help="requests/minute limit for external APIs (default 8)")


def ratios_from(a, exp: Experiment) -> list[float]:
    if a.ratios:
        return [float(x) for x in a.ratios.split(",")]
    if a.target_tokens:
        return [round(a.target_tokens / exp.tok.count(exp.text), 4)]
    return list(DEFAULT_RATIOS)


def _arms_arg(p):
    p.add_argument("--arms", default=None, help="comma list of arm names to include (default: all), e.g. original,condensed,llmlingua2@0.9")


def _arms(a):
    return [x.strip() for x in a.arms.split(",")] if getattr(a, "arms", None) else None


def _filter_arms(reps, a):
    arms = _arms(a)
    if not arms:
        return reps
    have = {r.name for r in reps}
    unknown = [x for x in arms if x not in have]
    if unknown:
        sys.exit(f"unknown arm(s) {unknown}; available: {sorted(have)}")
    return [r for r in reps if r.name in arms]


def client(spec: str, exp: Experiment, rpm=None):
    provider, _ = parse_spec(spec)
    return get_client(spec, cache_dir=exp.cache, rpm=rpm if provider != "ollama" else None)


def make_exp(a) -> Experiment:
    load_env(a.env)
    return Experiment(a.doc, a.root, get_tokenizer(a.tokenizer))


def make_compressor(method, a):
    from context_bench.compression.llmlingua import get_compressor
    kw = {"allow_download": a.allow_download}
    if method == "llmlingua2":
        kw["preserve_digits"] = a.preserve_digits
        if a.v2_model:
            kw["model"] = a.v2_model
    elif a.v1_model:
        kw["model"] = a.v1_model
    return get_compressor(method, **kw)


def _add_compress_args(p):
    p.add_argument("--allow-download", action="store_true",
                   help="permit downloading missing Hugging Face models (model files only; your document is not uploaded)")
    p.add_argument("--preserve-digits", action="store_true", help="LLMLingua-2: never drop tokens containing digits")
    p.add_argument("--compress-code", action="store_true", help="also compress fenced code blocks (default: keep verbatim)")
    p.add_argument("--v1-model", default=None, help="scoring LM for LLMLingua v1 (default openai-community/gpt2)")
    p.add_argument("--v2-model", default=None, help="LLMLingua-2 model (default xlm-roberta-large-meetingbank)")


# ---------------- commands ----------------
def cmd_analyze(a):
    exp = make_exp(a)
    s = exp.stats()
    print(f"Document: {a.doc}\n  characters:      {s['characters']:,}\n  words (approx):  {s['words_approx']:,}\n"
          f"  lines:           {s['lines']:,}\n  headings:        {s['headings']}\n  sections:        {s['sections']}\n"
          f"  tokens:          {s['tokens']:,}   ({s['tokenizer']})\n"
          "Characters and words are NOT tokens; only the token count approximates model cost.")
    print("\nStructure:")
    for sec in exp.doc.sections:
        print("  " + "  " * max(0, sec.level - 1) + f"[{sec.id}] {sec.title}  (lines {sec.start_line}-{sec.end_line})")


def cmd_index(a):
    exp = make_exp(a)
    c = client(a.index_model, exp, a.rpm) if a.index_model else None
    info = exp.index(c, a.max_section_tokens)
    for k, v in info.items():
        if v["source"] == "FAILED":
            print(f"{k}: FAILED - {v['notes'][0][:200]}")
            continue
        print(f"{k}: source={v['source']} sections={v['sections']} index_tokens={v['index_tokens']:,} -> {exp.dir / k}")
        for n in v.get("notes", []):
            print("   note:", n)


def cmd_compress(a):
    exp = make_exp(a)
    ratios = ratios_from(a, exp)
    for m in (["llmlingua", "llmlingua2"] if a.method == "both" else [a.method]):
        comp = make_compressor(m, a)
        exp.compress(comp, ratios, a.compress_code)
        if m == "llmlingua2" and a.combined:
            plan = exp.load_plan("indexed")
            if plan is None:
                print("[compress] --combined needs an LLM index first: run `index --index-model ...`.")
            else:
                exp.compress_sections(comp, a.combined_ratio, plan, a.compress_code)


def cmd_condense(a):
    exp = make_exp(a)
    exp.condense(client(a.condense_model, exp, a.rpm), a.min_section_tokens)


def cmd_generate_tests(a):
    exp = make_exp(a)
    c = client(a.gen_model, exp, a.rpm) if a.gen_model else None
    exp.make_tests(c, a.num_tests, a.tests)


def _select(tests, max_tests):
    if not max_tests or len(tests) <= max_tests:
        return tests
    by = defaultdict(list)
    for t in tests:
        by[t["category"]].append(t)
    out, i = [], 0
    while len(out) < max_tests and any(by.values()):
        for cat in sorted(by):
            if by[cat] and len(out) < max_tests:
                out.append(by[cat].pop(0))
    return out


def export_manual(exp: Experiment, reps, tests):
    d = exp.dir / "manual"
    (d / "prompts").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in reps:
        for t in tests:
            if r.kind == "static":
                body = (f"{ev_sys()}\n\n<document>\n{r.text}\n</document>\n\nQuestion: {t['question']}\n")
            else:
                body = (f"NAVIGATION representation '{r.name}'. Give the model ONLY the index below, let it choose "
                        f"sections, then paste just those sections from {exp.dir / 'indexed' / 'sections'}.\n\n"
                        f"<index>\n{r.index_md}\n</index>\n\nQuestion: {t['question']}\n")
            (d / "prompts" / f"{r.name.replace('@', '_')}__{t['id']}.txt").write_text(body)
            rows.append({"rep": r.name, "test_id": t["id"], "question": t["question"], "expected_answer": t["expected_answer"],
                         "critical_facts": " | ".join(t["critical_facts"]), "model_answer": "", "verdict": ""})
    with (d / "manual_verdicts.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"Manual evaluation files written to {d}/ : prompts/ (one per representation x test) and "
          "manual_verdicts.csv (fill model_answer and verdict = pass|partial|fail, then run `import-manual`).")


def ev_sys():
    from context_bench.evaluation.evaluator import ANSWER_SYS
    return ANSWER_SYS


def cmd_import_manual(a):
    from context_bench.evaluation.evaluator import Answer
    exp = make_exp(a)
    p = exp.dir / "manual" / "manual_verdicts.csv"
    judged, n = exp.load_judgments(), 0
    out = exp.dir / "answers" / "manual.jsonl"
    (exp.dir / "answers").mkdir(exist_ok=True)
    with out.open("w") as fh:
        for row in csv.DictReader(p.open()):
            v = row["verdict"].strip().lower()
            if v in ("pass", "partial", "fail") and row["model_answer"].strip():
                ans = Answer("manual:human", row["rep"], row["test_id"], row["model_answer"],
                             exp.tok.count(row["model_answer"]), 0, None, None, 0.0, 1, [], None)
                fh.write(json.dumps(ans.__dict__) + "\n")
                judged[("manual:human", row["rep"], row["test_id"])] = {"verdict": v, "reason": "human verdict"}
                n += 1
    (exp.dir / "judgments.json").write_text(json.dumps({"|".join(k): v for k, v in judged.items()}, indent=2))
    print(f"imported {n} manual verdicts; run `report` to include them.")


def cmd_evaluate(a):
    exp = make_exp(a)
    ratios = ratios_from(a, exp) if (a.ratios or a.target_ratio or a.target_tokens) else exp_ratios(exp)
    tests = _select(exp.load_tests(), a.max_tests)
    if not tests or any(t.get("question", "").startswith("TODO") for t in tests):
        sys.exit("No usable tests: run `generate-tests` (or pass --tests) first; draft tests need a real question.")
    reps = _filter_arms(exp.representations(ratios), a)
    if a.manual or not a.answer_models:
        export_manual(exp, reps, tests)
        return
    answer_clients = [client(s, exp, a.rpm) for s in a.answer_models.split(",")]
    judge = client(a.judge_model, exp, a.rpm) if a.judge_model else None
    exp.evaluate(reps, tests, answer_clients, judge)
    print("evaluation finished; run `report`.")


def exp_ratios(exp) -> list[float]:
    rs = sorted({round(m["target_retained_ratio"], 4) for m in exp.compression_metas()
                 if m.get("target_retained_ratio") is not None}, reverse=True)
    return rs or [DEFAULT_RATIO]


def models_info(a) -> dict:
    return {"index": getattr(a, "index_model", None), "generator": getattr(a, "gen_model", None),
            "answer": (getattr(a, "answer_models", None) or "").split(",") if getattr(a, "answer_models", None) else [],
            "judge": getattr(a, "judge_model", None)}


def cmd_report(a):
    exp = make_exp(a)
    res = exp.build_results(exp_ratios(exp), models_info(a), {"cli": sys.argv[1:]}, _arms(a))
    print(f"wrote {exp.dir / 'report.md'} and results.json")


def cmd_export_questions(a):
    exp = make_exp(a)
    qp, kp = exp.export_questions()
    print(f"wrote {qp} (ask these in fresh chats, one per arm) and {kp} (grade against it)")


def cmd_import_index(a):
    exp = make_exp(a)
    exp.import_index(a.plan, a.label)


def cmd_import_condensed(a):
    exp = make_exp(a)
    exp.import_condensed(a.condensed, a.label)


def cmd_import_answers(a):
    exp = make_exp(a)
    r = exp.import_answers(a.answers, a.arm, a.model)
    print(f"imported {r['imported']} answers for arm {a.arm} (model {a.model})")
    for k in ("missing", "unknown", "empty"):
        if r[k]:
            print(f"  {k}: {', '.join(r[k])}")


def cmd_judging_sheet(a):
    exp = make_exp(a)
    print(f"wrote {exp.judging_sheet(_arms(a), blind=a.blind, model=a.model)}: grade it (yourself or with a coding agent), save verdicts as JSON, "
          "then run `import-verdicts`.")


def cmd_import_verdicts(a):
    exp = make_exp(a)
    r = exp.import_verdicts(json.loads(Path(a.verdicts).read_text()), a.judge_label, blind=a.blind,
                            only_arms=[x.strip() for x in a.only_arms.split(",")] if a.only_arms else None,
                            model=a.model)
    print(f"imported {r['imported']} verdicts (judge: {a.judge_label})")
    if r["unknown"]:
        print("  no answer found for:", ", ".join(r["unknown"][:15]))
    if r["ungraded"]:
        print(f"  {len(r['ungraded'])} answers still ungraded (they fall back to the objective score)")


def cmd_run(a):
    exp = make_exp(a)
    s = exp.stats()
    print(f"[run] {a.doc}: {s['tokens']:,} tokens ({s['tokenizer']}), {s['sections']} sections")
    ratios = ratios_from(a, exp)
    ic = client(a.index_model, exp, a.rpm) if a.index_model else None
    exp.index(ic, a.max_section_tokens)
    if not a.skip_compression:
        for m in a.methods.split(","):
            comp = make_compressor(m, a)
            exp.compress(comp, ratios, a.compress_code)
            plan = exp.load_plan("indexed")
            if m == "llmlingua2" and plan is not None:
                exp.compress_sections(comp, a.combined_ratio, plan, a.compress_code)
            del comp
    if a.condense_model:
        exp.condense(client(a.condense_model, exp, a.rpm), a.min_section_tokens)
    all_ratios = sorted(set(ratios) | {a.combined_ratio}, reverse=True)
    gc = client(a.gen_model, exp, a.rpm) if a.gen_model else None
    if not exp.tests_path.is_file() or a.tests or a.regen_tests:
        exp.make_tests(gc, a.num_tests, a.tests)
    tests = _select(exp.load_tests(), a.max_tests)
    reps = _filter_arms(exp.representations(all_ratios), a)
    if not a.answer_models or any(t.get("question", "").startswith("TODO") for t in tests):
        export_manual(exp, reps, tests) if tests else None
        print("[run] no answering model (or only draft tests): stopped before automatic evaluation.")
    else:
        judge = client(a.judge_model, exp, a.rpm) if a.judge_model else None
        exp.evaluate(reps, tests, [client(x, exp, a.rpm) for x in a.answer_models.split(",")], judge)
    exp.build_results(all_ratios, models_info(a), {"cli": sys.argv[1:]}, _arms(a))
    print(f"[run] report: {exp.dir / 'report.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="context_bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="document statistics + structure"); _common(p); p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("index", help="split + index (LLM-planned if --index-model, else heuristic)")
    _common(p); _model_args(p); p.set_defaults(fn=cmd_index)

    p = sub.add_parser("compress", help="compress with LLMLingua / LLMLingua-2")
    _common(p); _ratio_args(p); _add_compress_args(p)
    p.add_argument("--method", choices=["llmlingua", "llmlingua2", "both"], default="llmlingua2")
    p.add_argument("--combined", action="store_true", help="also compress each LLM-indexed section (needs `index --index-model`)")
    p.add_argument("--combined-ratio", type=float, default=0.8, help="retained ratio for the combined arm (default 0.8)")
    p.set_defaults(fn=cmd_compress)

    p = sub.add_parser("condense", help="LLM rewrite into terse agent-readable form with mechanical fact checks")
    _common(p); _model_args(p, index=False)
    p.add_argument("--condense-model", required=True, help="e.g. ollama:<model>, gemini:<model>, openai:<model>")
    p.add_argument("--min-section-tokens", type=int, default=80, help="sections smaller than this stay verbatim")
    p.set_defaults(fn=cmd_condense)

    p = sub.add_parser("generate-tests", help="generate/import test cases from the ORIGINAL document")
    _common(p); _model_args(p, index=False)
    p.add_argument("--gen-model", default=None, help="e.g. gemini:gemini-flash-latest or ollama:<model>; none = draft only")
    p.add_argument("--num-tests", type=int, default=35)
    p.add_argument("--tests", default=None, help="hand-written tests JSON to include (schema in README)")
    p.set_defaults(fn=cmd_generate_tests)

    p = sub.add_parser("evaluate", help="run every representation against the same tests")
    _common(p); _ratio_args(p); _model_args(p, index=False); _arms_arg(p)
    p.add_argument("--answer-models", default=None, help="comma list, e.g. ollama:qwen2.5-coder:7b-instruct-q4_K_M,gemini:gemini-flash-latest")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--manual", action="store_true", help="export prompts + verdict CSV instead of calling a model")
    p.add_argument("--max-tests", type=int, default=None)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("export-questions", help="questions.md + answer_key.md for manual runs in Copilot/Claude/Codex")
    _common(p); p.set_defaults(fn=cmd_export_questions)

    p = sub.add_parser("judging-sheet", help="write judging_sheet.md: every arm's answers per question, for a human/agent judge")
    _common(p); _arms_arg(p)
    p.add_argument("--blind", action="store_true", help="hide arm names (per-question shuffled labels arm-1, arm-2...)")
    p.add_argument("--model", default=None, help="only this answering model's answers (e.g. antigravity:gemini-3.8-flash-high)")
    p.set_defaults(fn=cmd_judging_sheet)

    p = sub.add_parser("import-answers", help="load answers written outside the tool (### [test-id] blocks in a .md file)")
    _common(p); p.add_argument("answers"); p.add_argument("--arm", required=True); p.add_argument("--model", required=True)
    p.set_defaults(fn=cmd_import_answers)

    p = sub.add_parser("import-index", help="use an externally authored split plan (JSON) as the `indexed` arm")
    _common(p); p.add_argument("plan"); p.add_argument("--label", required=True, help="who authored it, e.g. claude-sonnet-5")
    p.set_defaults(fn=cmd_import_index)

    p = sub.add_parser("import-condensed", help="use an externally authored condensed document as the `condensed` arm")
    _common(p); p.add_argument("condensed"); p.add_argument("--label", required=True)
    p.set_defaults(fn=cmd_import_condensed)

    p = sub.add_parser("import-verdicts", help="load verdicts JSON {test: {pass:[arms], partial:[...], fail:[...]}}")
    _common(p); p.add_argument("verdicts", help="verdicts JSON file")
    p.add_argument("--judge-label", default="manual", help="who judged, e.g. claude-code or copilot (recorded)")
    p.add_argument("--blind", action="store_true", help="verdicts use the anonymous labels from `judging-sheet --blind`")
    p.add_argument("--only-arms", default=None, help="import verdicts for these arms only (comma list)")
    p.add_argument("--model", default=None, help="apply verdicts only to this answering model's answers")
    p.set_defaults(fn=cmd_import_verdicts)

    p = sub.add_parser("import-manual", help="load manual_verdicts.csv into the results"); _common(p); p.set_defaults(fn=cmd_import_manual)

    p = sub.add_parser("report", help="build report.md / results.json from saved artifacts")
    _common(p); _arms_arg(p); p.add_argument("--index-model"); p.add_argument("--gen-model"); p.add_argument("--answer-models"); p.add_argument("--judge-model")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("run", help="full experiment: index, compress, tests, evaluate, report")
    _common(p); _ratio_args(p); _model_args(p); _add_compress_args(p); _arms_arg(p)
    p.add_argument("--methods", default="llmlingua2", help="llmlingua2 (default), llmlingua, or both comma-separated")
    p.add_argument("--condense-model", default=None, help="adds the `condensed` arm (LLM rewrite with fact checks)")
    p.add_argument("--min-section-tokens", type=int, default=80)
    p.add_argument("--combined-ratio", type=float, default=0.8)
    p.add_argument("--skip-compression", action="store_true")
    p.add_argument("--gen-model", default=None); p.add_argument("--answer-models", default=None)
    p.add_argument("--judge-model", default=None); p.add_argument("--tests", default=None)
    p.add_argument("--num-tests", type=int, default=35); p.add_argument("--max-tests", type=int, default=None)
    p.add_argument("--regen-tests", action="store_true")
    p.set_defaults(fn=cmd_run)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (MissingDependency, ModelNotCached) as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(2)
