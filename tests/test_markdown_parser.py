from context_bench.analysis.markdown_parser import parse, find_headings, stats
from context_bench.analysis.tokenizer import CharApproxTokenizer, get_tokenizer
from conftest import SMALL


def test_heading_hierarchy_and_title_mode(small_doc):
    secs = {s.id: s for s in small_doc.sections}
    # single H1 with no preamble would be S0; here a preamble exists so H1 is a normal section
    assert secs["S1"].title == "Handbook"
    assert secs["S1.1"].title == "1. Alpha Rules" and secs["S1.1"].parent == "S1"
    assert secs["S1.2.1"].parent == "S1.2"
    assert secs["S1.2"].children == ["S1.2.1"]


def test_lone_h1_is_title_and_ids_follow_headings():
    doc = parse("# Title\n\nintro\n\n## 1. A\n\ntext\n\n### 1.1 A1\n\nx\n\n## 2. B\n\ny\n")
    assert [s.id for s in doc.sections] == ["S0", "S1", "S1.1", "S2"]
    assert doc.sections[1].doc_number == "1" and doc.sections[2].doc_number == "1.1"


def test_fenced_headings_ignored(small_doc):
    titles = [t for _, _, t in find_headings(small_doc.lines)]
    assert "not a heading" not in titles


def test_preamble_and_exact_cover(small_doc):
    assert small_doc.sections[0].id == "S0" and small_doc.sections[0].level == 0
    assert "".join(small_doc.section_text(s) for s in small_doc.sections) == small_doc.text


def test_no_headings_document():
    doc = parse("just text\n\nmore text\n")
    assert [s.id for s in doc.sections] == ["S0"] and doc.heading_count == 0
    assert doc.section_text(doc.sections[0]) == doc.text


def test_empty_document():
    assert parse("").sections == []


def test_stats_distinguish_chars_words_lines_tokens(small_doc):
    s = stats(small_doc, CharApproxTokenizer())
    assert s["characters"] == len(small_doc.text)
    assert s["words_approx"] == len(small_doc.text.split())
    assert s["lines"] == len(small_doc.lines)
    assert s["headings"] == 4 and s["tokens"] == (len(small_doc.text) + 3) // 4
    assert s["characters"] != s["tokens"] != s["words_approx"]


def test_tokenizer_is_replaceable(small_doc):
    class Words:
        name = "words"
        def count(self, t): return len(t.split())
    assert stats(small_doc, Words())["tokens"] == len(small_doc.text.split())


def test_tiktoken_counts():
    try:
        t = get_tokenizer("tiktoken:cl100k_base")
    except Exception:
        import pytest; pytest.skip("tiktoken vocabulary unavailable offline")
    assert t.count("hello world") == 2 and t.count("") == 0
