import json
from context_bench.indexing import build_plan
from context_bench.indexing.llm_splitter import make_blocks
from context_bench.indexing.plan import section_text
from conftest import FakeLLM


def plan_json(secs):
    return json.dumps({"sections": [
        {"id": f"S{i + 1}", "title": f"T{i + 1}", "start_block": a, "end_block": b, "parent": None,
         "summary": f"summary {i + 1}", "use_when": "when", "key_terms": ["k"],
         "related": [{"id": "S2", "relation": "exception_to", "reason": "r"}] if i == 0 else []}
        for i, (a, b) in enumerate(secs)]})


def test_blocks_cover_every_line(synth_doc):
    b = make_blocks(synth_doc)
    assert b[0][0] == 1 and b[-1][1] == len(synth_doc.lines)
    assert all(b[i][1] + 1 == b[i + 1][0] for i in range(len(b) - 1))


def test_valid_plan_is_sliced_mechanically(small_doc, tok):
    n = len(make_blocks(small_doc))
    llm = FakeLLM(lambda p, s, j: plan_json([(1, 3), (4, n)]))
    plan = build_plan(small_doc, tok, llm)
    assert plan.source == "llm" and len(plan.sections) == 2
    assert "".join(section_text(small_doc, s) for s in plan.sections) == small_doc.text
    assert plan.sections[0].related[0]["relation"] == "exception_to"


def test_gap_triggers_one_repair_retry(small_doc, tok):
    n = len(make_blocks(small_doc))
    answers = iter([plan_json([(1, 2), (4, n)]), plan_json([(1, 3), (4, n)])])
    llm = FakeLLM(lambda p, s, j: next(answers))
    plan = build_plan(small_doc, tok, llm)
    assert plan.source == "llm" and len(llm.prompts) == 2
    assert "PREVIOUS ANSWER WAS INVALID" in llm.prompts[1][1] and "gap" in llm.prompts[1][1]


def test_hallucinated_numbers_fall_back_to_heuristic(small_doc, tok, capsys):
    llm = FakeLLM(lambda p, s, j: plan_json([(1, 400), (401, 800)]))
    plan = build_plan(small_doc, tok, llm)
    assert plan.source == "heuristic-fallback" and len(llm.prompts) == 2
    assert any("rejected" in n for n in plan.notes)
    assert "falling back" in capsys.readouterr().err
    assert "".join(section_text(small_doc, s) for s in plan.sections) == small_doc.text


def test_garbage_and_fenced_json(small_doc, tok):
    n = len(make_blocks(small_doc))
    llm = FakeLLM(lambda p, s, j: "not json at all")
    assert build_plan(small_doc, tok, llm).source == "heuristic-fallback"
    llm = FakeLLM(lambda p, s, j: "```json\n" + plan_json([(1, n)]) + "\n```")
    assert build_plan(small_doc, tok, llm).source == "llm"


def test_windows_for_large_documents(synth_doc, tok):
    n = len(make_blocks(synth_doc))
    calls = []
    def fn(prompt, system, j):
        import re
        nums = [int(x) for x in re.findall(r"\[B(\d+)\]", prompt.split("DOCUMENT BLOCKS:")[1])]
        lo, hi = min(nums), max(nums); calls.append((lo, hi))
        return plan_json([(lo, hi)])
    plan = build_plan(synth_doc, tok, FakeLLM(fn), window_tokens=1000)
    assert len(calls) > 1 and plan.source == "llm"
    assert "".join(section_text(synth_doc, s) for s in plan.sections) == synth_doc.text
    assert len({s.id for s in plan.sections}) == len(plan.sections)
