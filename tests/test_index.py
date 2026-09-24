import pytest
from context_bench.indexing.markdown_index import heuristic_plan
from context_bench.indexing.plan import (SectionSpec, SplitPlan, normalize_and_validate, render_index,
                                         section_text, slug, write_index_dir)


def test_plan_is_exact_cover_and_valid(synth_doc, tok):
    plan = heuristic_plan(synth_doc, tok, 300)
    assert normalize_and_validate(plan, len(synth_doc.lines)) == []
    assert "".join(section_text(synth_doc, s) for s in plan.sections) == synth_doc.text


def test_written_sections_reproduce_original_byte_for_byte(synth_doc, tok, tmp_path):
    plan = heuristic_plan(synth_doc, tok, 300)
    info = write_index_dir(tmp_path, synth_doc, plan, tok)
    rebuilt = "".join((tmp_path / "sections" / f"{s.id}-{slug(s.title)}.md").read_text()
                      for s in sorted(plan.sections, key=lambda s: s.start_line))
    assert rebuilt == synth_doc.text and info["sections"] == len(plan.sections)
    for f in ("index.md", "graph.json", "split_plan.json"):
        assert (tmp_path / f).is_file()


def test_oversized_sections_are_chunked(tok):
    from context_bench.analysis.markdown_parser import parse
    doc = parse("# T\n\n## A\n\n" + "para text here.\n\n" * 200)
    plan = heuristic_plan(doc, tok, 200)
    assert len([s for s in plan.sections if s.id.startswith("S1")]) > 1
    assert normalize_and_validate(plan, len(doc.lines)) == []


def test_cross_references_including_lists(synth_doc, tok):
    plan = heuristic_plan(synth_doc, tok)
    by = plan.by_id()
    rel = {r["id"] for r in by["S9.1"].related}
    assert {"S4.1", "S4.3"} <= rel                      # "Sections 4.1 and 4.3"
    assert "S7.3" in {r["id"] for r in by["S5.3"].related}   # "see Section 7.3"-style ref


def test_index_is_a_router_not_a_copy(synth_doc, tok):
    plan = heuristic_plan(synth_doc, tok)
    idx = render_index(plan, synth_doc, tok)
    assert tok.count(idx) < 0.85 * tok.count(synth_doc.text)
    assert "[S7.3]" in idx and "Related:" in idx and " tok)" in idx
    assert "exponential backoff with a base delay of 250 milliseconds, doubling" not in idx


def _spec(i, a, b, **kw): return SectionSpec(i, i, a, b, **kw)


def test_validate_repairs_gaps_and_flags_overlaps():
    p = SplitPlan([_spec("S1", 2, 4), _spec("S2", 7, 10)], "llm")
    assert normalize_and_validate(p, 12) == []
    assert (p.sections[0].start_line, p.sections[0].end_line, p.sections[-1].end_line) == (1, 6, 12)
    assert any("gap" in n for n in p.notes)
    bad = SplitPlan([_spec("S1", 1, 6), _spec("S2", 5, 10)], "llm")
    assert any("overlap" in e for e in normalize_and_validate(bad, 10))


def test_validate_rejects_bad_ids_and_ranges_and_cleans_relations():
    assert normalize_and_validate(SplitPlan([_spec("S1", 1, 3), _spec("S1", 4, 6)], "llm"), 6)
    assert normalize_and_validate(SplitPlan([_spec("S1", 1, 99)], "llm"), 6)
    p = SplitPlan([_spec("S1", 1, 3, related=[{"id": "S9", "relation": "x"}, {"id": "S2", "relation": "bogus"}],
                         parent="NOPE"), _spec("S2", 4, 6)], "llm")
    assert normalize_and_validate(p, 6) == []
    assert p.sections[0].related == [{"id": "S2", "relation": "see_also", "reason": ""}]
    assert p.sections[0].parent is None
