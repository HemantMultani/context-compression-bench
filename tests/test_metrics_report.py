import json
from context_bench.evaluation import evaluator as ev
from context_bench.evaluation.judge import judge_answers, objective_verdict
from context_bench.evaluation.metrics import _rates, aggregate, pareto
from context_bench.reporting.report import generate_report
from conftest import FakeLLM

TESTS = [
    {"id": "A-01", "category": "A", "question": "timeout?", "expected_answer": "30 seconds",
     "source_sections": ["S1"], "critical_facts": ["30 seconds"], "difficulty": "easy"},
    {"id": "G-01", "category": "G", "question": "retry?", "expected_answer": "250 ms and 2 seconds",
     "source_sections": ["S2"], "critical_facts": ["250 milliseconds", "2 seconds"], "difficulty": "hard"},
]


def ans(model, rep, tid, text, i=100):
    return ev.Answer(model, rep, tid, text, i, 10, None, None, 0.5, 1, [], None)


def test_rates():
    r = _rates(["pass", "pass", "partial", "fail"])
    assert (r["n"], r["accuracy"], r["partial"], r["fail"], r["score"]) == (4, .5, .25, .25, .625)
    assert _rates([])["score"] is None


def test_objective_answer_matching_is_lenient_but_not_blind():
    from context_bench.evaluation.judge import fact_in_answer
    assert fact_in_answer("at most 8 downstream synchronous calls", "8")
    assert fact_in_answer("base delay of 250 milliseconds", "250 ms base delay")
    assert not fact_in_answer("at most 8 downstream synchronous calls", "12")
    assert fact_in_answer("must never use SELECT *", "You must never use SELECT * in queries")
    assert not fact_in_answer("reporting replica", "the primary database only")
    assert not fact_in_answer("", "anything")
    assert fact_in_answer("PostgreSQL 15", "The standard database engine is PostgreSQL 15.")   # trailing period
    assert fact_in_answer("rejected with HTTP 422", "The server rejects the POST request with HTTP 422.")
    assert fact_in_answer("the coverage requirement in Section 8.1", "... and the coverage requirement in Section 8.1.")


def test_objective_verdict():
    t = TESTS[1]
    assert objective_verdict("250 milliseconds and 2 seconds", t) == ("pass", 1.0)
    assert objective_verdict("about 250 milliseconds", t) == ("partial", 0.5)
    assert objective_verdict("NOT FOUND", t) == ("fail", 0.0)


def make_reps():
    return [ev.Representation("original", "static", text="timeout 30 seconds; 250 milliseconds cap 2 seconds " * 5),
            ev.Representation("llmlingua2@0.5", "static", text="timeout 30 seconds")]


def test_aggregate_and_pareto(tok):
    answers = [ans("m", "original", "A-01", "30 seconds", 400), ans("m", "original", "G-01", "250 milliseconds and 2 seconds", 400),
               ans("m", "llmlingua2@0.5", "A-01", "30 seconds", 100), ans("m", "llmlingua2@0.5", "G-01", "NOT FOUND", 100)]
    judged = {("m", a.rep, a.test_id): {"verdict": objective_verdict(a.answer, {x["id"]: x for x in TESTS}[a.test_id])[0], "reason": ""}
              for a in answers}
    res = aggregate(answers, judged, TESTS, make_reps(), tok, original_tokens=tok.count(make_reps()[0].text))
    rows = {r["rep"]: r for r in res["rows"]}
    assert rows["original"]["judged"]["score"] == 1.0 and rows["llmlingua2@0.5"]["judged"]["score"] == 0.5
    assert rows["llmlingua2@0.5"]["input_token_reduction_vs_original"] == 0.75
    assert rows["llmlingua2@0.5"]["context_reduction"] > 0.5
    assert rows["llmlingua2@0.5"]["fact_survival_in_representation"] == 0.5      # "30 seconds" kept, G-01 facts lost
    assert rows["llmlingua2@0.5"]["by_category"]["G"]["judged"]["score"] == 0.0
    assert rows["original"]["judge_objective_agreement"] == 1.0
    assert set(pareto(res["rows"])["m"]) == {"original", "llmlingua2@0.5"}   # each is best on one axis


def test_judge_batches_and_ignores_garbage(tok):
    answers = [ans("m", "r", "A-01", "30 seconds"), ans("m", "r", "G-01", "no idea")]
    calls = []
    def fn(prompt, system, j):
        calls.append(prompt)
        return json.dumps({"grades": [{"id": "0", "verdict": "PASS", "reason": "ok"}, {"id": "1", "verdict": "maybe"}, {"id": "9", "verdict": "pass"}]})
    out = judge_answers(answers, {t["id"]: t for t in TESTS}, FakeLLM(fn))
    assert len(calls) == 1 and "candidate_answer" in calls[0] and "llmlingua" not in calls[0]   # one batch; judge never sees the representation
    assert out == {("m", "r", "A-01"): {"verdict": "pass", "reason": "ok"}}
    assert judge_answers(answers, {t["id"]: t for t in TESTS}, FakeLLM(lambda *a: "garbage"), log=lambda *_: None) == {}


def test_report_contents(tok):
    reps = make_reps()
    answers = [ans("m", "original", "A-01", "30 seconds", 400), ans("m", "llmlingua2@0.5", "A-01", "NOT FOUND", 100),
               ans("m", "original", "G-01", "250 milliseconds and 2 seconds", 400), ans("m", "llmlingua2@0.5", "G-01", "250 milliseconds", 100)]
    judged = {("m", "llmlingua2@0.5", "A-01"): {"verdict": "fail", "reason": "abstained"}}
    agg = aggregate(answers, judged, TESTS, reps, tok, 100)
    fails = [{"rep": "llmlingua2@0.5", "model": "m", "test_id": "A-01", "category": "A", "verdict": "fail",
              "source": "model-judged", "question": "timeout?", "expected": "30 seconds", "facts": ["30 seconds"],
              "answer": "NOT FOUND", "sections": ["S1"], "reason": "abstained"}]
    res = {"name": "doc", "generated_at": "now", "document": {"tokens": 100, "tokenizer": "t", "sections": 2, "lines": 10,
           "characters": 1, "words_approx": 1, "headings": 2},
           "tests": {"total": 2, "by_origin": {"manual": 2}}, "representations": ["original", "llmlingua2@0.5"],
           "compression": [{"method": "llmlingua2", "model": "org/m", "target_retained_ratio": .5, "retained_ratio": .6,
                            "reduction": .4, "original_tokens": 100, "compressed_tokens": 60, "runtime_sec": 1.0}],
           "index": {}, "aggregate": agg, "failures": fails,
           "manifest": {"document_sha256": "x", "tokenizer": "t", "models": {}, "python": "3", "platform": "p", "packages": {}}}
    md = generate_report(res)
    for needle in ("## Executive summary", "## Document statistics", "## Compression results", "## Evaluation results",
                   "## Results by test category", "## Failed or partially preserved facts", "retained_ratio = compressed_tokens",
                   "model-judged, not objective truth", "Expected answer: 30 seconds", "Model answer: NOT FOUND",
                   "Source section(s): S1", "Representation tested: `llmlingua2@0.5`", "achieved retained"):
        assert needle in md, needle
    assert "best" not in md.lower().replace("no method is declared best", "")


def test_navigation_coverage_uses_line_overlap_across_indexes():
    from context_bench.evaluation.metrics import nav_coverage
    rep = ev.Representation("indexed", "navigate", ranges={"S7": (100, 140), "S8": (141, 170)})
    test = {"source_sections": ["S7.3"], "source_lines": [[125, 132]]}          # cited under a different plan's ids
    assert nav_coverage(rep, test, ["S7"]) == (True, True)
    assert nav_coverage(rep, test, ["S8"]) == (False, False)
    two = {"source_sections": ["S7.3", "S8.1"], "source_lines": [[125, 132], [150, 155]]}
    assert nav_coverage(rep, two, ["S7"]) == (True, False)
    legacy = ev.Representation("indexed", "navigate")                            # no ranges: id fallback
    assert nav_coverage(legacy, {"source_sections": ["S1"]}, ["S1"]) == (True, True)


def test_paired_regressions_vs_original(tok):
    reps = make_reps()
    answers = [ans("m", "original", "A-01", "30 seconds"), ans("m", "original", "G-01", "250 milliseconds and 2 seconds"),
               ans("m", "llmlingua2@0.5", "A-01", "30 seconds"), ans("m", "llmlingua2@0.5", "G-01", "NOT FOUND")]
    res = aggregate(answers, {}, TESTS, reps, tok, 100)
    rows = {r["rep"]: r for r in res["rows"]}
    assert rows["original"]["regressions"] is None
    assert rows["llmlingua2@0.5"]["regressions"] == {"n_base": 2, "ids": ["G-01"]}
    # a question the original also failed is not counted as a regression
    answers[1] = ans("m", "original", "G-01", "NOT FOUND")
    rows = {r["rep"]: r for r in aggregate(answers, {}, TESTS, reps, tok, 100)["rows"]}
    assert rows["llmlingua2@0.5"]["regressions"] == {"n_base": 1, "ids": []}
