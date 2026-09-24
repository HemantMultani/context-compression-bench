import json
from context_bench.evaluation.test_generator import (CATEGORIES, draft_tests, fact_in_text, generate_tests,
                                                     ground_check, validate_test, validate_tests)
from context_bench.indexing import build_plan
from context_bench.indexing.plan import section_text
from conftest import FakeLLM


def test_manual_tests_match_schema_and_are_grounded(synth_doc, tok, manual_tests):
    assert validate_tests(manual_tests) == []
    assert {t["category"] for t in manual_tests} == set(CATEGORIES)
    plan = build_plan(synth_doc, tok, None)
    texts = {s.id: section_text(synth_doc, s) for s in plan.sections}
    for t in manual_tests:
        assert ground_check(t, texts) == [], t["id"]


def test_schema_validation_errors():
    good = {"id": "A-1", "category": "A", "question": "q?", "expected_answer": "a", "source_sections": ["S1"],
            "critical_facts": ["f"], "difficulty": "easy"}
    assert validate_test(good) == []
    assert validate_test({**good, "category": "Z"}) and validate_test({**good, "difficulty": "x"})
    assert validate_test({**good, "critical_facts": []}) and validate_test({**good, "question": " "})
    assert validate_test({k: v for k, v in good.items() if k != "id"})


def test_fact_matching():
    assert fact_in_text("timeout of 30 seconds", "The query timeout of 30 seconds applies.")
    assert fact_in_text("Idempotency-Key", "an `Idempotency-Key` header")
    assert not fact_in_text("timeout of 45 seconds", "The query timeout of 30 seconds applies.")
    assert fact_in_text("retry at most three times always", "always retry at most three times")   # fuzzy
    assert not fact_in_text("", "anything")


def test_ground_check_rejects_invented_facts_and_bad_categories():
    texts = {"S1": "Timeout is 30 seconds.", "S2": "Batch is 500 rows."}
    base = {"id": "x", "category": "A", "question": "q", "expected_answer": "a", "source_sections": ["S1"],
            "critical_facts": ["Timeout is 30 seconds"], "difficulty": "easy"}
    assert ground_check(base, texts) == []
    assert ground_check({**base, "critical_facts": ["Timeout is 99 seconds"]}, texts)
    assert ground_check({**base, "source_sections": ["S9"]}, texts)
    assert ground_check({**base, "category": "G", "critical_facts": ["Timeout is"]}, texts)   # no digit
    assert ground_check({**base, "category": "D"}, texts)                                       # 1 section only


def test_generator_keeps_grounded_and_rejects_the_rest(synth_doc, tok):
    plan = build_plan(synth_doc, tok, None)
    texts = {s.id: section_text(synth_doc, s) for s in plan.sections}
    def fn(prompt, system, j):
        return json.dumps({"tests": [
            {"question": "What is the query timeout?", "expected_answer": "30 seconds",
             "source_sections": ["S5.2"], "critical_facts": ["timeout of 30 seconds"], "difficulty": "easy"},
            {"question": "Invented?", "expected_answer": "x", "source_sections": ["S5.2"],
             "critical_facts": ["timeout of 999 seconds"], "difficulty": "easy"},
            {"question": "What is the query timeout?", "expected_answer": "30 seconds",
             "source_sections": ["S5.2"], "critical_facts": ["timeout of 30 seconds"], "difficulty": "easy"}]})
    ok, rej = generate_tests(plan, texts, tok, FakeLLM(fn), num_tests=7, log=lambda *_: None)
    assert ok and all(not ground_check(t, texts) for t in ok if t["category"] != "D")
    reasons = " ".join(r for x in rej for r in x["reasons"])
    assert "not found in cited sections" in reasons and "duplicate" in reasons
    assert len({t["id"] for t in ok}) == len(ok)


def test_generator_survives_garbage(synth_doc, tok):
    plan = build_plan(synth_doc, tok, None)
    texts = {s.id: section_text(synth_doc, s) for s in plan.sections}
    ok, rej = generate_tests(plan, texts, tok, FakeLLM(lambda *a: "garbage"), num_tests=7, log=lambda *_: None)
    assert ok == []


def test_draft_tests_without_llm(synth_doc, tok):
    plan = build_plan(synth_doc, tok, None)
    texts = {s.id: section_text(synth_doc, s) for s in plan.sections}
    d = draft_tests(plan, texts)
    assert len(d) > 10 and all(t["origin"] == "draft" and t["question"].startswith("TODO") for t in d)
