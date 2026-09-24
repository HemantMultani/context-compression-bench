"""Whole pipeline with fake LLMs and a fake compressor: no network, no model downloads."""
import json
import pytest
from context_bench.experiment import Experiment
from context_bench.indexing import SplitFailed, build_plan
from conftest import FakeCompressor, FakeLLM, SYNTH, block_plan_llm, terse_llm


def answering_llm():
    """Behaves like a navigating agent that always finds section 'Part 4' (retries) and answers from what it sees."""
    def fn(prompt, system, j):
        if "<index>" in prompt and "Which sections must you read" in prompt:
            ids = __import__("re").findall(r"\[(S\d+)\] Part (\d+)", prompt)
            return json.dumps({"read": [i for i, n in ids][:2]})
        if "Sections you requested" in prompt:
            return json.dumps({"answer": "base delay of 250 milliseconds, capped at 2 seconds"})
        return "base delay of 250 milliseconds, capped at 2 seconds" if "250" in prompt else "NOT FOUND"
    return FakeLLM(fn, model="agent")


def test_full_pipeline(tok, tmp_path, manual_tests):
    exp = Experiment(SYNTH, tmp_path, tok)
    info = exp.index(block_plan_llm(), 1200, log=lambda *_: None)
    assert info["indexed"]["source"] == "llm" and info["indexed_heuristic"]["source"] == "heuristic"
    assert (exp.dir / "indexed" / "index.md").is_file() and (exp.dir / "indexed_heuristic" / "index.md").is_file()
    comp = FakeCompressor(); comp.name = "llmlingua2"
    exp.compress(comp, [0.9, 0.8], log=lambda *_: None)
    exp.compress_sections(comp, 0.8, exp.load_plan("indexed"), log=lambda *_: None)
    meta = exp.condense(terse_llm(), 40, log=lambda *_: None)
    assert meta["method"] == "condensed" and meta["target_retained_ratio"] is None and meta["retained_ratio"] < 1
    (tmp_path / "manual.json").write_text(json.dumps({"tests": manual_tests}))
    exp.make_tests(None, manual_file=str(tmp_path / "manual.json"), log=lambda *_: None)
    tests = exp.load_tests()
    assert len(tests) == len(manual_tests) and all(t.get("source_lines") for t in tests)
    reps = exp.representations([0.9, 0.8])
    assert [r.name for r in reps] == ["original", "llmlingua2@0.9", "llmlingua2@0.8", "condensed", "indexed",
                                      "indexed_heuristic", "combined@0.8"]
    judge = FakeLLM(lambda p, s, j: json.dumps({"grades": [{"id": str(i), "verdict": "partial", "reason": "r"} for i in range(16)]}), model="judge")
    exp.evaluate(reps, tests[:6], [answering_llm()], judge, log=lambda *_: None)
    res = exp.build_results([0.9, 0.8], {"answer": ["agent"]}, {})
    assert (exp.dir / "report.md").is_file() and (exp.dir / "results.json").is_file() and (exp.dir / "manifest.json").is_file()
    rows = {r["rep"]: r for r in res["aggregate"]["rows"]}
    assert set(rows) == {r.name for r in reps} and rows["original"]["answered"] == 6
    assert rows["original"]["regressions"] is None and "ids" in rows["condensed"]["regressions"]
    md = (exp.dir / "report.md").read_text()
    assert "Loss versus the original" in md and "Failed or partially preserved facts" in md and "condensed" in md
    m = json.loads((exp.dir / "manifest.json").read_text())
    import hashlib
    assert m["document_sha256"] == hashlib.sha256(SYNTH.read_bytes()).hexdigest() and m["tests_sha256"]
    # arms filter
    res2 = exp.build_results([0.9, 0.8], {}, {}, arms=["original", "condensed"])
    assert {r["rep"] for r in res2["aggregate"]["rows"]} == {"original", "condensed"}
    # resumability: a second evaluate call asks the answering model nothing new
    llm = answering_llm()
    exp.evaluate(reps, tests[:6], [llm], None, log=lambda *_: None)
    assert llm.prompts == []


def test_failed_llm_split_leaves_no_mislabeled_arm(tok, tmp_path):
    exp = Experiment(SYNTH, tmp_path, tok)
    logs = []
    info = exp.index(FakeLLM(lambda *a: "garbage"), 1200, log=logs.append)
    assert info["indexed"]["source"] == "FAILED" and info["indexed_heuristic"]["source"] == "heuristic"
    assert not (exp.dir / "indexed").exists() and (exp.dir / "indexed_heuristic" / "index.md").is_file()
    assert any("FAILED" in l and "will NOT be evaluated" in l for l in logs)
    assert [r.name for r in exp.representations([0.8])] == ["original", "indexed_heuristic"]
    with pytest.raises(SplitFailed):
        build_plan(exp.doc, tok, FakeLLM(lambda *a: "garbage"), strict=True)
    # a later good run recovers cleanly and replaces the FAILED marker
    info = exp.index(block_plan_llm(), 1200, log=lambda *_: None)
    assert info["indexed"]["source"] == "llm" and (exp.dir / "indexed" / "index.md").is_file()


def test_no_llm_path_still_produces_artifacts(tok, tmp_path):
    exp = Experiment(SYNTH, tmp_path, tok)
    fc = FakeCompressor(); fc.name = "llmlingua2"
    info = exp.index(None); exp.compress(fc, [0.8], log=lambda *_: None)
    assert list(info) == ["indexed_heuristic"]
    exp.make_tests(None, log=lambda *_: None)                        # draft candidates only
    tests = exp.load_tests()
    assert tests and all(t["question"].startswith("TODO") for t in tests)
    res = exp.build_results([0.8], {}, {})
    assert res["aggregate"]["rows"] == [] and (exp.dir / "report.md").is_file()


def test_judging_sheet_and_verdict_import(tok, tmp_path, manual_tests):
    exp = Experiment(SYNTH, tmp_path, tok)
    exp.index(None)
    (tmp_path / "manual.json").write_text(json.dumps({"tests": manual_tests}))
    exp.make_tests(None, manual_file=str(tmp_path / "manual.json"), log=lambda *_: None)
    tests = exp.load_tests()[:2]
    from context_bench.evaluation import evaluator as ev
    reps = [ev.Representation("original", "static", text="doc 250"), ev.Representation("condensed", "static", text="doc")]
    exp.evaluate(reps, tests, [FakeLLM(lambda p, s, j: "same answer", model="m")], None, log=lambda *_: None)
    sheet = exp.judging_sheet().read_text()
    assert "**original, condensed** -> same answer" in sheet and tests[0]["question"] in sheet   # identical answers merged
    r = exp.import_verdicts({tests[0]["id"]: {"pass": ["original"], "fail": ["condensed", "nonexistent"]}}, "claude-code")
    assert r["imported"] == 2 and r["unknown"] == [f"{tests[0]['id']}/nonexistent"]
    assert set(r["ungraded"]) == {f"{tests[1]['id']}/original", f"{tests[1]['id']}/condensed"}
    j = exp.load_judgments()
    assert j[("ollama:m".replace("ollama", "fake"), "original", tests[0]["id"])]["verdict"] == "pass"
    assert "claude-code" in j[("fake:m", "condensed", tests[0]["id"])]["reason"]


def test_export_questions_has_no_answers_in_the_question_file(tok, tmp_path, manual_tests):
    exp = Experiment(SYNTH, tmp_path, tok)
    exp.index(None)
    (tmp_path / "manual.json").write_text(json.dumps({"tests": manual_tests}))
    exp.make_tests(None, manual_file=str(tmp_path / "manual.json"), log=lambda *_: None)
    qp, kp = exp.export_questions()
    q, k = qp.read_text(), kp.read_text()
    assert all(t["question"] in q and t["question"] in k for t in manual_tests)
    assert all(t["expected_answer"] not in q for t in manual_tests) and "Expected:" in k and "Expected:" not in q


def test_import_external_index_and_condensed_and_blind_judging(tok, tmp_path, manual_tests):
    from context_bench.evaluation import evaluator as ev
    exp = Experiment(SYNTH, tmp_path, tok)
    n = len(exp.doc.lines)
    plan = {"sections": [{"id": "S1", "title": "First half", "start_line": 1, "end_line": 100, "parent": None, "summary": "a",
                          "use_when": "b", "key_terms": [], "related": [{"id": "S2", "relation": "see_also", "reason": "x"}]},
                         {"id": "S2", "title": "Second half", "start_line": 101, "end_line": n, "parent": None,
                          "summary": "c", "use_when": "d", "key_terms": [], "related": []}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    info = exp.import_index(tmp_path / "plan.json", "someone", log=lambda *_: None)
    assert info["sections"] == 2 and exp.load_plan("indexed").source == "external:someone"
    (tmp_path / "bad.json").write_text(json.dumps({"sections": [{**plan["sections"][0], "end_line": 120}, plan["sections"][1]]}))
    with pytest.raises(ValueError, match="overlap"):     # overlapping sections are refused
        exp.import_index(tmp_path / "bad.json", "someone", log=lambda *_: None)
    cond = SYNTH.read_text().replace("This handbook is entirely fictional and exists only as a synthetic test document for compression and indexing experiments. ", "")
    (tmp_path / "c.md").write_text(cond)
    meta = exp.import_condensed(tmp_path / "c.md", "someone", log=lambda *_: None)
    assert meta["params"]["mechanical_check_missing"] == [] and meta["retained_ratio"] < 1
    (tmp_path / "c2.md").write_text(cond.replace("250", "some"))
    assert any("250" in m for m in exp.import_condensed(tmp_path / "c2.md", "someone", log=lambda *_: None)["params"]["mechanical_check_missing"])
    # blind judging round trip
    (tmp_path / "manual.json").write_text(json.dumps({"tests": manual_tests}))
    exp.make_tests(None, manual_file=str(tmp_path / "manual.json"), log=lambda *_: None)
    tests = exp.load_tests()[:2]
    reps = [ev.Representation("original", "static", text="a 250"), ev.Representation("condensed", "static", text="a")]
    answers = {"original": "orig answer", "condensed": "cond answer"}
    exp.evaluate(reps, tests, [FakeLLM(lambda p, s, j: "orig answer" if "a 250" in p else "cond answer", model="m")], None, log=lambda *_: None)
    sheet = exp.judging_sheet(blind=True).read_text()
    assert "original" not in sheet.split("### ", 1)[1] and "condensed ->" not in sheet and "arm-1" in sheet
    key = {v: k for k, v in json.loads((exp.dir / "judging_blind_key.json").read_text())[tests[0]["id"]].items()}   # arm -> label
    orig_label, cond_label = key["original"], key["condensed"]
    r = exp.import_verdicts({tests[0]["id"]: {"pass": [orig_label], "fail": [cond_label]}}, "me", blind=True)
    j = exp.load_judgments()
    assert j[("fake:m", "original", tests[0]["id"])]["verdict"] == "pass" and j[("fake:m", "condensed", tests[0]["id"])]["verdict"] == "fail"
    exp.import_verdicts({tests[1]["id"]: {"pass": list(json.loads((exp.dir / "judging_blind_key.json").read_text())[tests[1]["id"]])}}, "me", blind=True, only_arms=["condensed"])
    assert ("fake:m", "condensed", tests[1]["id"]) in exp.load_judgments() and ("fake:m", "original", tests[1]["id"]) not in exp.load_judgments()


def test_import_external_answers_and_model_scoped_grading(tok, tmp_path, manual_tests):
    from context_bench.evaluation import evaluator as ev
    exp = Experiment(SYNTH, tmp_path, tok)
    exp.index(None)
    (tmp_path / "manual.json").write_text(json.dumps({"tests": manual_tests}))
    exp.make_tests(None, manual_file=str(tmp_path / "manual.json"), log=lambda *_: None)
    t0, t1 = exp.load_tests()[:2]
    fc = FakeCompressor(); fc.name = "llmlingua2"; exp.compress(fc, [0.9], log=lambda *_: None)
    # an existing answer from another model for the same arm must survive the import and grading
    exp.evaluate([ev.Representation("original", "static", text="x")], [t0], [FakeLLM(lambda *a: "qwen answer", model="q")], None, log=lambda *_: None)
    f = tmp_path / "ans.md"
    f.write_text(f"### [{t0['id']}]\nfirst answer\nsecond line\n\n### [{t1['id']}]\nanother answer\n\n### [ZZ-99]\nnot a test\n")
    r = exp.import_answers(f, "original", "ext:m")
    assert r["imported"] == 2 and r["unknown"] == ["ZZ-99"] and len(r["missing"]) == len(exp.load_tests()) - 2
    got = {(a.model, a.test_id): a for a in exp.load_answers()}
    assert got[("ext:m", t0["id"])].answer == "first answer\nsecond line" and got[("ext:m", t0["id"])].input_tokens == 0
    assert ("fake:q", t0["id"]) in got
    exp.import_answers(f, "original", "ext:m")                                # re-import replaces, does not duplicate
    assert len([a for a in exp.load_answers() if a.model == "ext:m"]) == 2
    with pytest.raises(ValueError):
        exp.import_answers(f, "no-such-arm", "ext:m")
    exp.import_verdicts({t0["id"]: {"fail": ["original"]}}, "me", model="ext:m")
    j = exp.load_judgments()
    assert j[("ext:m", "original", t0["id"])]["verdict"] == "fail" and ("fake:q", "original", t0["id"]) not in j
    sheet = exp.judging_sheet(model="ext:m").read_text()
    assert "first answer" in sheet and "qwen answer" not in sheet
    res = exp.build_results([0.9], {}, {})                                   # unknown tokens/navigation do not break the report
    rows = {(r["model"], r["rep"]): r for r in res["aggregate"]["rows"]}
    assert rows[("ext:m", "original")].get("mean_input_tokens") is None and rows[("fake:q", "original")]["mean_input_tokens"] > 0
