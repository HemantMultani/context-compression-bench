import json
import pytest
from context_bench.evaluation import evaluator as ev
from context_bench.indexing import build_plan
from context_bench.indexing.plan import render_index, section_text
from conftest import FakeLLM


def nav_rep(doc, tok, name="indexed", compressed=None):
    plan = build_plan(doc, tok, None, 400)
    secs = compressed or {s.id: section_text(doc, s) for s in plan.sections}
    return plan, ev.Representation(name, "navigate", index_md=render_index(plan, doc, tok), sections=secs,
                                   titles={s.id: s.title for s in plan.sections},
                                   related={s.id: [(r["id"], r["relation"]) for r in s.related] for s in plan.sections})


def guard_for(doc, plan):
    return ev.LeakGuard(doc.text, {s.id: section_text(doc, s) for s in plan.sections})


def test_static_answer_only_contains_the_representation(synth_doc, tok, tmp_path):
    plan = build_plan(synth_doc, tok, None)
    rep = ev.Representation("llmlingua2@0.5", "static", text="COMPRESSED DOC ONLY about timeout 30")
    llm = FakeLLM(lambda p, s, j: "30 seconds")
    tests = [{"id": "T1", "question": "timeout?"}]
    res = ev.run_answers([rep], tests, llm, tok, guard_for(synth_doc, plan), tmp_path / "a.jsonl", log=lambda *_: None)
    assert res[0].answer == "30 seconds" and res[0].turns == 1
    system, prompt = llm.prompts[0]
    assert "COMPRESSED DOC ONLY" in prompt and "timeout?" in prompt
    assert synth_doc.text[:300] not in prompt
    assert res[0].input_tokens == tok.count(system + "\n" + prompt) and res[0].output_tokens == tok.count("30 seconds")


def test_leakage_guard_blocks_original_text(synth_doc, tok, tmp_path):
    plan = build_plan(synth_doc, tok, None)
    g = guard_for(synth_doc, plan)
    leaky = ev.Representation("llmlingua2@0.5", "static", text="short")
    sec = section_text(synth_doc, plan.by_id()["S7.3"])
    with pytest.raises(ev.LeakageError):
        g.check("doc: " + sec, leaky)
    with pytest.raises(ev.LeakageError):
        g.check(synth_doc.text, leaky)
    g.check(synth_doc.text, leaky, is_original=True)               # the original representation may hold it
    legit = ev.Representation("x", "static", text=sec)             # same text legitimately held (e.g. ratio 1.0)
    g.check(sec, legit)


def test_leak_is_raised_through_run_answers(synth_doc, tok, tmp_path):
    plan = build_plan(synth_doc, tok, None)
    rep = ev.Representation("bad", "static", text="tiny")
    # simulate a bug that smuggles a section into the prompt via the question
    sec = section_text(synth_doc, plan.by_id()["S7.3"])
    tests = [{"id": "T1", "question": sec}]
    with pytest.raises(ev.LeakageError):
        ev.run_answers([rep], tests, FakeLLM(lambda *a: "x"), tok, guard_for(synth_doc, plan), tmp_path / "a.jsonl")


def test_navigation_only_shows_requested_sections(synth_doc, tok, tmp_path):
    plan, rep = nav_rep(synth_doc, tok)
    turns = []
    def fn(prompt, system, j):
        turns.append(prompt)
        if len(turns) == 1:
            return json.dumps({"read": ["S7.3", "NOPE"]})
        return json.dumps({"answer": "3 retries"})
    llm = FakeLLM(fn)
    res = ev.run_answers([rep], [{"id": "T1", "question": "retries?"}], llm, tok, guard_for(synth_doc, plan),
                         tmp_path / "a.jsonl", log=lambda *_: None)
    a = res[0]
    assert a.fetched == ["S7.3"] and a.answer == "3 retries" and a.turns == 2
    assert "<index>" in turns[0] and section_text(synth_doc, plan.by_id()["S7.3"]).strip()[:60] not in turns[0]
    assert "[S7.3]" in turns[1] and "[S5.2]" not in turns[1]
    assert "Layering" not in turns[1]                                # unrequested section absent


def test_navigation_second_hop_and_fallbacks(synth_doc, tok, tmp_path):
    plan, rep = nav_rep(synth_doc, tok)
    seq = iter([json.dumps({"read": ["S5.2"]}), json.dumps({"read": ["S7.3"]}), json.dumps({"answer": "ok"})])
    llm = FakeLLM(lambda p, s, j: next(seq))
    a = ev.run_answers([rep], [{"id": "T1", "question": "q"}], llm, tok, guard_for(synth_doc, plan),
                       tmp_path / "a.jsonl", log=lambda *_: None)[0]
    assert a.fetched == ["S5.2", "S7.3"] and a.turns == 3 and a.answer == "ok"
    llm2 = FakeLLM(lambda p, s, j: json.dumps({"read": ["NOPE"]}))                 # nothing valid requested
    b = ev.run_answers([rep], [{"id": "T2", "question": "q"}], llm2, tok, guard_for(synth_doc, plan),
                       tmp_path / "b.jsonl", log=lambda *_: None)[0]
    assert b.answer == "NOT FOUND" and b.fetched == []


def test_combined_serves_compressed_sections_not_originals(synth_doc, tok, tmp_path):
    plan = build_plan(synth_doc, tok, None, 400)
    compressed = {s.id: "compressed " + s.id for s in plan.sections}
    _, rep = nav_rep(synth_doc, tok, "combined@0.8", compressed)
    seen = []
    def fn(prompt, system, j):
        seen.append(prompt)
        return json.dumps({"read": ["S7.3"]}) if len(seen) == 1 else json.dumps({"answer": "x"})
    ev.run_answers([rep], [{"id": "T1", "question": "q"}], FakeLLM(fn), tok, guard_for(synth_doc, plan),
                   tmp_path / "a.jsonl", log=lambda *_: None)
    assert "compressed S7.3" in seen[1] and "exponential backoff" not in seen[1]


def test_resume_skips_finished_and_stops_on_repeated_errors(synth_doc, tok, tmp_path):
    plan = build_plan(synth_doc, tok, None)
    rep = ev.Representation("r", "static", text="doc")
    tests = [{"id": f"T{i}", "question": "q"} for i in range(6)]
    path = tmp_path / "a.jsonl"
    llm = FakeLLM(lambda *a: "ans")
    ev.run_answers([rep], tests[:3], llm, tok, guard_for(synth_doc, plan), path, log=lambda *_: None)
    llm2 = FakeLLM(lambda *a: "ans")
    res = ev.run_answers([rep], tests, llm2, tok, guard_for(synth_doc, plan), path, log=lambda *_: None)
    assert len(llm2.prompts) == 3 and len(res) == 6                  # only the 3 new tests were asked
    def boom(*a): raise RuntimeError("quota")
    logs = []
    res = ev.run_answers([rep], tests, FakeLLM(boom), tok, guard_for(synth_doc, plan), tmp_path / "b.jsonl",
                         log=logs.append, max_consecutive_errors=2)
    assert len(res) == 2 and all(r.error for r in res) and any("consecutive errors" in l for l in logs)
