"""Pipeline steps operating on an experiment directory:

experiments/<doc-name>/
    original.md  compressed/{llmlingua,llmlingua2,combined}/  indexed/  indexed_heuristic/
    test_cases.json  answers/<model>.jsonl  judgments.json  results.json  report.md  manifest.json  .cache/
"""
from __future__ import annotations
import hashlib, json, platform, re, shutil, sys, time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from context_bench.analysis.markdown_parser import parse, stats
from context_bench.compression.base import compress_document, make_result
from context_bench.compression.condense import condense_document, missing_facts
from context_bench.evaluation import evaluator as ev
from context_bench.evaluation.judge import judge_answers, objective_verdict
from context_bench.evaluation.metrics import aggregate
from context_bench.evaluation.test_generator import (draft_tests, generate_tests, ground_check,
                                                     validate_tests)
from context_bench.indexing import SplitFailed, build_plan
from context_bench.indexing.plan import SplitPlan, normalize_and_validate, section_text, write_index_dir
from context_bench.reporting.report import generate_report

DEFAULT_RATIO = 0.8
DEFAULT_RATIOS = [0.9, 0.8, 0.7]     # a sweep: the data, not a forced target, decides how much can be cut


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rname(method: str, ratio: float) -> str:
    return f"{method}@{ratio:g}"


class Experiment:
    def __init__(self, doc_path: str | Path, root: str | Path = "experiments", tokenizer=None):
        self.doc_path = Path(doc_path)
        self.name = self.doc_path.stem
        self.dir = Path(root) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tok = tokenizer
        self.text = self.doc_path.read_text()
        (self.dir / "original.md").write_text(self.text)
        self.doc = parse(self.text)
        self.cache = self.dir / ".cache"

    # ---------- analysis / index ----------
    def stats(self) -> dict:
        return stats(self.doc, self.tok)

    def index(self, client=None, max_section_tokens: int = 1200, log=print) -> dict:
        """Two independent arms: `indexed_heuristic/` (deterministic, always written) and `indexed/` (LLM-planned,
        only when a client is given AND the plan validates). A failed LLM split never leaves a mislabeled arm."""
        info_path = self.dir / "index_info.json"
        out = json.loads(info_path.read_text()) if info_path.is_file() else {}
        h = build_plan(self.doc, self.tok, None, max_section_tokens)
        info = write_index_dir(self.dir / "indexed_heuristic", self.doc, h, self.tok)
        info["notes"] = h.notes
        out["indexed_heuristic"] = info
        if client:
            try:
                plan = build_plan(self.doc, self.tok, client, max_section_tokens, strict=True)
            except SplitFailed as e:
                log(f"[index] ERROR: the LLM split FAILED ({str(e)[:300]}). The `indexed` (LLM) arm will NOT be "
                    "evaluated; only `indexed_heuristic` exists. Use a stronger --index-model and re-run `index`.")
                shutil.rmtree(self.dir / "indexed", ignore_errors=True)
                out["indexed"] = {"source": "FAILED", "sections": 0, "index_tokens": 0, "edges": 0, "notes": [str(e)[:400]]}
            else:
                info = write_index_dir(self.dir / "indexed", self.doc, plan, self.tok)
                info["notes"] = plan.notes
                out["indexed"] = info
                log(f"[index] LLM split ok: {info['sections']} sections, index {info['index_tokens']:,} tokens")
        info_path.write_text(json.dumps(out, indent=2))
        return out

    def import_index(self, plan_path, label: str, log=print) -> dict:
        """Use a split plan written by someone/something else (a person, a coding agent) as the `indexed` arm.
        Same schema as split_plan.json: {"sections": [{id,title,start_line,end_line,parent,summary,use_when,
        key_terms,related:[{id,relation,reason}]}]}. It is validated as an exact cover of the document; the tool
        slices the text, so the author can only choose boundaries and metadata."""
        data = json.loads(Path(plan_path).read_text())
        plan = SplitPlan.from_json({"sections": data["sections"], "source": f"external:{label}", "notes": []})
        errs = normalize_and_validate(plan, len(self.doc.lines))
        if errs:
            raise ValueError("invalid plan: " + "; ".join(errs[:8]))
        info = write_index_dir(self.dir / "indexed", self.doc, plan, self.tok)
        info["notes"] = plan.notes + [f"plan authored externally by {label}"]
        info_path = self.dir / "index_info.json"
        out = json.loads(info_path.read_text()) if info_path.is_file() else {}
        out["indexed"] = info
        info_path.write_text(json.dumps(out, indent=2))
        log(f"[index] imported plan from {label}: {info['sections']} sections, index {info['index_tokens']:,} tokens")
        for n in plan.notes:
            log(f"[index] note: {n}")
        return info

    def import_condensed(self, path, label: str, log=print) -> dict:
        """Use a condensed document written by someone/something else as the `condensed` arm. The same mechanical
        loss checks as the automatic arm are run and reported (missing numbers/identifiers/headings/markers)."""
        text = Path(path).read_text()
        missing = missing_facts(self.text, text)
        d = self.dir / "compressed" / "condensed"
        d.mkdir(parents=True, exist_ok=True)
        (d / "condensed.md").write_text(text)
        meta = make_result("condensed", label, None, self.tok.count(self.text), self.tok.count(text), 0.0, self.tok.name,
                           {"authored_by": label, "mechanical_check_missing": missing}).meta()
        meta["fallback_sections"], meta["sections"] = [], []
        (d / "condensed.json").write_text(json.dumps({"meta": meta}, indent=2))
        log(f"[condense] imported document from {label}: retained {meta['retained_ratio']:.3f} "
            f"({meta['original_tokens']:,} -> {meta['compressed_tokens']:,} tokens); mechanical check: "
            + ("PASS" if not missing else f"{len(missing)} issue(s): {', '.join(missing[:12])}"))
        return meta

    def load_plan(self, which: str = "indexed") -> SplitPlan | None:
        p = self.dir / which / "split_plan.json"
        return SplitPlan.from_json(json.loads(p.read_text())) if p.is_file() else None

    # ---------- compression ----------
    def compress(self, compressor, ratios: list[float], compress_code: bool = False, log=print) -> list[dict]:
        metas = []
        d = self.dir / "compressed" / compressor.name
        d.mkdir(parents=True, exist_ok=True)
        for r in ratios:
            res = compress_document(self.doc, compressor, self.tok, r, compress_code)
            (d / f"r{r:.2f}.md").write_text(res.text)
            (d / f"r{r:.2f}.json").write_text(json.dumps(res.meta(), indent=2))
            metas.append(res.meta())
            log(f"[compress] {compressor.name} target={r:.2f} achieved retained={res.retained_ratio:.3f} "
                f"({res.original_tokens:,} -> {res.compressed_tokens:,} tokens) in {res.runtime_sec:.1f}s")
        return metas

    def compress_sections(self, compressor, ratio: float, plan: SplitPlan, compress_code: bool = False,
                          log=print) -> dict:
        """'combined': compress each indexed section separately (headings/code kept verbatim)."""
        secs, t0, orig, comp = {}, time.perf_counter(), 0, 0
        for s in plan.sections:
            body = section_text(self.doc, s)
            sub = parse(body)
            res = compress_document(sub, compressor, self.tok, ratio, compress_code)
            secs[s.id] = res.text
            orig += res.original_tokens; comp += res.compressed_tokens
        meta = make_result("combined(" + compressor.name + ")", compressor.model, ratio, orig, comp,
                           time.perf_counter() - t0, self.tok.name, compressor.params).meta()
        d = self.dir / "compressed" / "combined"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"r{ratio:.2f}.json").write_text(json.dumps({"meta": meta, "sections": secs}, indent=2))
        log(f"[compress] combined target={ratio:.2f} achieved retained={meta['retained_ratio']:.3f} "
            f"({orig:,} -> {comp:,} tokens; section text only, index excluded)")
        return meta

    def condense(self, client, min_section_tokens: int = 80, log=print) -> dict:
        """LLM rewrite into terse agent-readable form, section by section, with mechanical fact checks."""
        llm_plan_ = self.load_plan("indexed")
        plan = llm_plan_ or self.load_plan("indexed_heuristic") or build_plan(self.doc, self.tok, None)
        text, info = condense_document(self.doc, plan.sections, client, self.tok, min_section_tokens, log)
        d = self.dir / "compressed" / "condensed"
        d.mkdir(parents=True, exist_ok=True)
        (d / "condensed.md").write_text(text)
        meta = make_result("condensed", client.model, None, self.tok.count(self.text), self.tok.count(text),
                           info["runtime_sec"], self.tok.name,
                           {"min_section_tokens": min_section_tokens,
                            "units": "LLM-index sections" if llm_plan_ else "heading-based sections",
                            "checks": "numbers, identifiers, code spans, acronyms, quotes, headings, negation/condition markers"}).meta()
        meta["fallback_sections"] = info["fallback_sections"]
        meta["sections"] = info["sections"]
        (d / "condensed.json").write_text(json.dumps({"meta": meta}, indent=2))
        log(f"[condense] retained {meta['retained_ratio']:.3f} ({meta['original_tokens']:,} -> {meta['compressed_tokens']:,} tokens); "
            f"{len(info['fallback_sections'])} of {len(info['sections'])} sections kept verbatim (failed the fact checks or not shorter)")
        return meta

    def compression_metas(self) -> list[dict]:
        out = []
        for p in sorted((self.dir / "compressed").glob("*/*.json")):
            d = json.loads(p.read_text())
            out.append(d["meta"] if "meta" in d else d)
        return out

    # ---------- representations ----------
    def _nav_rep(self, name: str, which: str, sections: dict[str, str] | None = None) -> ev.Representation | None:
        plan = self.load_plan(which)
        if plan is None:
            return None
        d = self.dir / which
        return ev.Representation(
            name, "navigate", index_md=(d / "index.md").read_text(),
            sections=sections if sections is not None else {s.id: section_text(self.doc, s) for s in plan.sections},
            titles={s.id: s.title for s in plan.sections},
            related={s.id: [(r["id"], r["relation"]) for r in s.related] for s in plan.sections},
            ranges={s.id: (s.start_line, s.end_line) for s in plan.sections})

    def representations(self, ratios: list[float]) -> list[ev.Representation]:
        reps = [ev.Representation("original", "static", text=self.text)]
        for method in ("llmlingua", "llmlingua2"):
            for r in ratios:
                p = self.dir / "compressed" / method / f"r{r:.2f}.md"
                if p.is_file():
                    reps.append(ev.Representation(rname(method, r), "static", text=p.read_text()))
        cp = self.dir / "compressed" / "condensed" / "condensed.md"
        if cp.is_file():
            reps.append(ev.Representation("condensed", "static", text=cp.read_text()))
        for which in ("indexed", "indexed_heuristic"):
            rep = self._nav_rep(which, which)
            if rep:
                reps.append(rep)
        for r in ratios:
            p = self.dir / "compressed" / "combined" / f"r{r:.2f}.json"
            if p.is_file():
                rep = self._nav_rep(rname("combined", r), "indexed", json.loads(p.read_text())["sections"])
                if rep:
                    reps.append(rep)
        return reps

    def guard(self) -> ev.LeakGuard:
        plan = self.load_plan("indexed") or build_plan(self.doc, self.tok, None)
        return ev.LeakGuard(self.text, {s.id: section_text(self.doc, s) for s in plan.sections})

    # ---------- tests ----------
    @property
    def tests_path(self) -> Path:
        return self.dir / "test_cases.json"

    def load_tests(self) -> list[dict]:
        return json.loads(self.tests_path.read_text())["tests"] if self.tests_path.is_file() else []

    def make_tests(self, client=None, num_tests: int = 35, manual_file: str | None = None, log=print) -> dict:
        """Manual tests cite the heading-based section ids printed by `analyze`; generated tests cite the ids of
        the LLM index when there is one. Every test also stores `source_lines` (original line ranges) so
        navigation can be scored against ANY index by line overlap."""
        hplan = build_plan(self.doc, self.tok, None)
        htexts = {s.id: section_text(self.doc, s) for s in hplan.sections}
        plan = self.load_plan("indexed") or hplan
        texts = {s.id: section_text(self.doc, s) for s in plan.sections}
        lines_of = lambda pl, ids: [[pl.by_id()[i].start_line, pl.by_id()[i].end_line] for i in ids if i in pl.by_id()]
        tests, rejected = [], []
        if manual_file:
            manual = json.loads(Path(manual_file).read_text())
            manual = manual["tests"] if isinstance(manual, dict) else manual
            errs = validate_tests(manual)
            if errs:
                raise ValueError("manual tests invalid:\n  " + "\n  ".join(errs))
            for t in manual:
                t.setdefault("origin", "manual")
                unknown = [x for x in t["source_sections"] if x not in htexts]
                if unknown:
                    log(f"[tests] WARNING manual test {t['id']} cites unknown section ids {unknown} (use the ids "
                        "printed by `analyze`); not grounded")
                else:
                    g = ground_check(t, htexts)
                    if g:
                        log(f"[tests] WARNING manual test {t['id']} not grounded in its sections: {g}")
                    t["source_lines"] = lines_of(hplan, t["source_sections"])
                tests.append(t)
        if client:
            gen, rejected = generate_tests(plan, texts, self.tok, client, num_tests, log=log)
            ids = {t["id"] for t in tests}
            for t in gen:
                while t["id"] in ids:
                    t["id"] += "g"
                ids.add(t["id"]); t["source_lines"] = lines_of(plan, t["source_sections"]); tests.append(t)
        elif not manual_file:
            tests = draft_tests(hplan, htexts)
            for t in tests:
                t["source_lines"] = lines_of(hplan, t["source_sections"])
            log("[tests] no generator model configured: wrote DRAFT candidates (question = TODO). Edit "
                "test_cases.json (or supply --tests) before evaluating.")
        payload = {"generated_at": now(), "document_sha256": sha256(self.text),
                   "generator": client.label if client else None, "tests": tests}
        self.tests_path.write_text(json.dumps(payload, indent=2))
        (self.dir / "test_cases.rejected.json").write_text(json.dumps(rejected, indent=2))
        log(f"[tests] {len(tests)} tests kept, {len(rejected)} rejected by grounding/schema checks")
        return {"kept": len(tests), "rejected": len(rejected)}

    # ---------- evaluation ----------
    def evaluate(self, reps, tests, answer_clients, judge_client=None, log=print) -> None:
        (self.dir / "answers").mkdir(exist_ok=True)
        guard = self.guard()
        for c in answer_clients:
            safe = c.label.replace(":", "_").replace("/", "_")
            ev.run_answers(reps, tests, c, self.tok, guard, self.dir / "answers" / f"{safe}.jsonl", log=log)
        if judge_client:
            answers = self.load_answers()
            judged = self.load_judgments()
            todo = [a for a in answers if a.error is None and (a.model, a.rep, a.test_id) not in judged]
            def save(partial):
                merged = {**judged, **partial}
                (self.dir / "judgments.json").write_text(json.dumps({"|".join(k): v for k, v in merged.items()}, indent=2))
            new = judge_answers(todo, {t["id"]: t for t in tests}, judge_client, log=log, on_batch=save)
            judged.update(new)
            save({})
            log(f"[judge] graded {len(new)} of {len(todo)} pending answers with {judge_client.label}")

    def load_answers(self) -> list[ev.Answer]:
        out = []
        for p in sorted((self.dir / "answers").glob("*.jsonl")):
            for line in p.read_text().splitlines():
                out.append(ev.Answer(**json.loads(line)))
        return out

    def load_judgments(self) -> dict:
        p = self.dir / "judgments.json"
        if not p.is_file():
            return {}
        return {tuple(k.split("|", 2)): v for k, v in json.loads(p.read_text()).items()}

    def import_answers(self, path, arm: str, model_label: str) -> dict:
        """Load answers written outside the tool (e.g. an agent that wrote answers.md in `### [test-id]` blocks).
        Token usage and navigation were not measured, so they are stored as unknown (0 input tokens, 0 turns); only
        the size of the arm's context is known and shown in the report."""
        parts = re.split(r"(?m)^###\s*\[([^\]]+)\]\s*$", Path(path).read_text())
        got = {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
        tests = {t["id"]: t for t in self.load_tests()}
        reps = {r.name for r in self.representations(DEFAULT_RATIOS)}
        if arm not in reps:
            raise ValueError(f"unknown arm {arm!r}; available: {sorted(reps)}")
        rows = [ev.Answer(model_label, arm, tid, txt, 0, self.tok.count(txt), None, None, 0.0, 0, [], None)
                for tid, txt in got.items() if tid in tests and txt]
        (self.dir / "answers").mkdir(exist_ok=True)
        out = self.dir / "answers" / (model_label.replace(":", "_").replace("/", "_") + ".jsonl")
        keep = []
        if out.is_file():           # replace this arm's earlier rows for the same model
            keep = [l for l in out.read_text().splitlines() if json.loads(l)["rep"] != arm]
        out.write_text("\n".join(keep + [json.dumps(r.__dict__) for r in rows]) + "\n")
        return {"imported": len(rows), "missing": sorted(set(tests) - set(got)),
                "unknown": sorted(set(got) - set(tests)), "empty": sorted(k for k, v in got.items() if not v)}

    # ---------- judging by a person / coding agent ----------
    def judging_sheet(self, arms: list[str] | None = None, blind: bool = False, model: str | None = None) -> Path:
        """One block per test: question, answer key and every arm's answer (identical answers merged), so a
        human or coding agent can grade all arms of a question at a glance."""
        tests = {t["id"]: t for t in self.load_tests()}
        answers = [a for a in self.load_answers() if a.error is None and (not arms or a.rep in arms)
                   and (not model or a.model == model)]
        by, key = {}, {}
        if blind:      # hide which arm produced which answer: per-question shuffled anonymous labels
            import random
            for tid in {a.test_id for a in answers}:
                names = sorted({a.rep for a in answers if a.test_id == tid})
                order = random.Random(f"blind-{tid}").sample(names, len(names))
                key[tid] = {rep: f"arm-{i + 1}" for i, rep in enumerate(order)}
        for a in answers:
            label = key[a.test_id][a.rep] if blind else a.rep
            by.setdefault(a.test_id, {}).setdefault((a.model, " ".join(a.answer.split())), []).append(label)
        L = ["# Judging sheet",
             "Grade every arm's answer as pass / partial / fail against the expected answer and critical facts.",
             "pass = all critical facts stated correctly, nothing contradicted; partial = some facts or a required "
             "condition/exception missing; fail = wrong, contradictory, or NOT FOUND.",
             'Save verdicts as JSON: {"<test id>": {"pass": ["<arm>", ...], "partial": [...], "fail": [...]}} '
             "and import with `import-verdicts`.", ""]
        for tid in [t for t in tests if t in by]:
            t = tests[tid]
            L += [f"### {tid} [{t['category']}, {t['difficulty']}] {t['question']}",
                  f"- Expected: {t['expected_answer']}", f"- Critical facts: {' | '.join(t['critical_facts'])}"]
            for (model, text), reps in by[tid].items():
                L.append(f"- **{', '.join(reps)}** -> {text[:600] or '(empty)'}")
            L.append("")
        path = self.dir / "judging_sheet.md"
        path.write_text("\n".join(L))
        if blind:
            (self.dir / "judging_blind_key.json").write_text(json.dumps({t: {v: k for k, v in m.items()} for t, m in key.items()}))
        return path

    def import_verdicts(self, data: dict, label: str, blind: bool = False, only_arms: list[str] | None = None,
                        model: str | None = None) -> dict:
        if blind:      # translate anonymous labels back to arm names
            kp = json.loads((self.dir / "judging_blind_key.json").read_text())
            data = {t: {v: [kp[t][l] for l in ls if l in kp[t]] for v, ls in g.items()} for t, g in data.items() if t in kp}
        if only_arms:
            data = {t: {v: [r for r in rs if r in only_arms] for v, rs in g.items()} for t, g in data.items()}
        judged = self.load_judgments()
        answers = [a for a in self.load_answers() if a.error is None and (not model or a.model == model)]
        n, unknown = 0, set()
        for tid, groups in data.items():
            for verdict in ("pass", "partial", "fail"):
                for rep in groups.get(verdict, []):
                    hit = [a for a in answers if a.test_id == tid and a.rep == rep]
                    if not hit:
                        unknown.add(f"{tid}/{rep}")
                    for a in hit:
                        judged[(a.model, a.rep, a.test_id)] = {"verdict": verdict, "reason": f"judge: {label}"}
                        n += 1
        (self.dir / "judgments.json").write_text(json.dumps({"|".join(k): v for k, v in judged.items()}, indent=2))
        ungraded = [f"{a.test_id}/{a.rep}" for a in answers if (a.model, a.rep, a.test_id) not in judged]
        return {"imported": n, "unknown": sorted(unknown), "ungraded": ungraded}

    def export_questions(self) -> tuple[Path, Path]:
        """questions.md (paste one at a time into a FRESH chat per arm) and answer_key.md (for grading)."""
        tests = self.load_tests()
        q = ["# Questions",
             "For each arm you test (original / condensed / llmlingua2@R / indexed folder): open a NEW chat with only that "
             "arm's file(s) available, then ask each question below. Suggested prefix for every question:",
             "> Using only the provided documentation, answer concisely but include every specific value, condition and exception it gives.",
             ""]
        key = ["# Answer key", "Grade each answer pass / partial / fail: pass = all critical facts stated correctly and nothing "
               "contradicted; partial = some facts or a required condition/exception missing; fail = wrong, contradictory or missing.", ""]
        for i, t in enumerate(tests, 1):
            q.append(f"{i}. [{t['id']}] {t['question']}")
            key += [f"### {i}. [{t['id']}] ({t['category']}, {t['difficulty']}) {t['question']}",
                    f"- Expected: {t['expected_answer']}", f"- Critical facts: {' | '.join(t['critical_facts'])}", ""]
        qp, kp = self.dir / "questions.md", self.dir / "answer_key.md"
        qp.write_text("\n".join(q) + "\n"); kp.write_text("\n".join(key))
        return qp, kp

    # ---------- results / report ----------
    def build_results(self, ratios: list[float], models_info: dict, params: dict, arms: list[str] | None = None) -> dict:
        tests = self.load_tests()
        reps = self.representations(ratios)
        answers, judged = self.load_answers(), self.load_judgments()
        if arms:
            reps = [r for r in reps if r.name in arms]
            answers = [a for a in answers if a.rep in arms]
        reps_used = [r for r in reps if any(a.rep == r.name for a in answers)] or reps
        agg = aggregate(answers, judged, tests, reps, self.tok, self.tok.count(self.text)) if answers else {"rows": []}
        by_id = {t["id"]: t for t in tests}
        failures = []
        for a in answers:
            if a.error is not None or a.test_id not in by_id:
                continue
            t = by_id[a.test_id]
            j = judged.get((a.model, a.rep, a.test_id))
            verdict, source = (j["verdict"], "model-judged") if j else (objective_verdict(a.answer, t)[0], "objective fact match")
            if verdict != "pass":
                failures.append({"rep": a.rep, "model": a.model, "test_id": a.test_id, "category": t["category"],
                                 "verdict": verdict, "source": source, "question": t["question"],
                                 "expected": t["expected_answer"], "facts": t["critical_facts"],
                                 "answer": a.answer, "sections": t["source_sections"],
                                 "reason": (j or {}).get("reason", "")})
        order = {r.name: i for i, r in enumerate(reps)}
        failures.sort(key=lambda f: (order.get(f["rep"], 99), f["model"], f["test_id"]))
        from collections import Counter
        manifest = self.manifest(models_info, params, tests)
        res = {"name": self.name, "generated_at": now(), "document": self.stats(),
               "tests": {"total": len(tests), "by_origin": dict(Counter(t.get("origin", "?") for t in tests))},
               "representations": [r.name for r in reps_used],
               "compression": self.compression_metas(),
               "index": json.loads((self.dir / "index_info.json").read_text()) if (self.dir / "index_info.json").is_file() else {},
               "aggregate": agg, "failures": failures, "manifest": manifest}
        (self.dir / "results.json").write_text(json.dumps(res, indent=2))
        (self.dir / "report.md").write_text(generate_report(res))
        return res

    def manifest(self, models_info: dict, params: dict, tests: list[dict]) -> dict:
        pk = {}
        for n in ("llmlingua", "torch", "transformers", "tiktoken", "google-genai"):
            try:
                pk[n] = metadata.version(n)
            except metadata.PackageNotFoundError:
                pk[n] = "not installed"
        m = {"document": str(self.doc_path), "document_sha256": sha256(self.text),
             "tests_sha256": sha256(self.tests_path.read_text()) if self.tests_path.is_file() else None,
             "num_tests": len(tests), "tokenizer": self.tok.name, "models": models_info, "parameters": params,
             "timestamp": now(), "python": sys.version.split()[0], "platform": platform.platform(),
             "packages": pk}
        (self.dir / "manifest.json").write_text(json.dumps(m, indent=2))
        return m
