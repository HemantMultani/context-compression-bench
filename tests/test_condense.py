from context_bench.compression.condense import (condense_document, condense_section, hard_tokens,
                                                missing_facts)
from context_bench.indexing import build_plan
from conftest import FakeLLM, terse_llm

TEXT = """## 7.3 Retries

Retries are allowed only for operations that are idempotent. A retried operation is attempted at most 3 times, with a base delay of 250 milliseconds and TLS 1.3 required. Never retry `capture_payment`; see LedgerlineError and /healthz. Payment capture is never retried, except by manual reconciliation. """ * 1 + "Filler sentence that is not important at all. " * 12


def test_hard_tokens_capture_numbers_identifiers_code_acronyms():
    h = hard_tokens(TEXT)
    for t in ("3", "250", "1.3", "capture_payment", "ledgerlineerror", "/healthz", "tls"):
        assert t in h, t


def test_missing_facts_detects_each_kind_of_loss():
    ok = TEXT.replace("Filler sentence that is not important at all. ", "")
    assert missing_facts(TEXT, ok) == []
    assert any("250" in m for m in missing_facts(TEXT, ok.replace("250", "some")))
    assert any("heading" in m for m in missing_facts(TEXT, ok.replace("## 7.3 Retries", "Retries")))
    assert any("capture_payment" in m for m in missing_facts(TEXT, ok.replace("capture_payment", "capture")))
    no_neg = "## 7.3 Retries\n" + "retry only idempotent ops; 3 times; 250 milliseconds; TLS 1.3; capture_payment; LedgerlineError; /healthz; manual reconciliation"
    assert any("negation" in m for m in missing_facts(TEXT, no_neg.replace("Never", "")))
    assert any("condition" in m for m in missing_facts(TEXT.replace("only", "ONLY").replace("except", "EXCEPT"),
                                                        "## 7.3 Retries\nnever 3 250 TLS 1.3 capture_payment LedgerlineError /healthz not"))


def test_small_sections_are_left_verbatim(tok):
    llm = FakeLLM(lambda *a: "should not be called")
    out, info = condense_section(llm, "short text", tok, min_tokens=80)
    assert out == "short text" and info["status"] == "skipped_small" and llm.prompts == []


def test_good_rewrite_is_accepted(tok):
    llm = terse_llm()
    out, info = condense_section(llm, TEXT, tok, min_tokens=20)
    assert info["status"] == "ok" and info["attempts"] == 1 and tok.count(out) < tok.count(TEXT)
    assert missing_facts(TEXT, out) == []


def test_retry_with_feedback_then_accept(tok):
    good = TEXT.replace("Filler sentence that is not important at all. ", "")
    answers = iter([good.replace("250", "some"), good])
    llm = FakeLLM(lambda *a: next(answers))
    out, info = condense_section(llm, TEXT, tok, min_tokens=20)
    assert info["attempts"] == 2 and info["status"] == "ok" and out == good.strip()
    assert "REJECTED" in llm.prompts[1][1] and "250" in llm.prompts[1][1]


def test_falls_back_to_verbatim_when_facts_keep_getting_lost(tok):
    llm = FakeLLM(lambda *a: "## 7.3 Retries\nretry stuff")
    out, info = condense_section(llm, TEXT, tok, min_tokens=20)
    assert out == TEXT and info["status"] == "fallback_verbatim" and info["missing"]


def test_not_shorter_is_rejected_and_model_errors_fall_back(tok):
    out, info = condense_section(FakeLLM(lambda *a: TEXT + " extra"), TEXT, tok, min_tokens=20)
    assert out == TEXT and info["status"] == "fallback_verbatim"
    def boom(*a): raise RuntimeError("down")
    out, info = condense_section(FakeLLM(boom), TEXT, tok, min_tokens=20)
    assert out == TEXT and "model error" in info["missing"][0]


def test_condense_document_keeps_headings_and_reports_fallbacks(synth_doc, tok):
    plan = build_plan(synth_doc, tok, None, 400)
    text, info = condense_document(synth_doc, plan.sections, terse_llm(), tok, min_tokens=40, log=lambda *_: None)
    for line in synth_doc.text.splitlines():
        if line.startswith("## "):
            assert line in text
    assert tok.count(text) < tok.count(synth_doc.text)
    # a section the model cannot shorten is (correctly) kept verbatim, never silently altered
    for r in info["sections"]:
        if r["status"] == "fallback_verbatim":
            assert r["missing"] == ["output was not shorter than the original"]
    bad, info = condense_document(synth_doc, plan.sections, FakeLLM(lambda *a: "junk"), tok, min_tokens=40, log=lambda *_: None)
    assert bad.strip() == synth_doc.text.strip() and len(info["fallback_sections"]) > 0    # everything kept verbatim, nothing lost


def test_orphaned_numbers_are_caught():
    from context_bench.compression.condense import orphaned_numbers
    orig = "## 5.4 Indexes\n\nIndexes on tables larger than 10 million rows must be created with the CONCURRENTLY option."
    good = "## 5.4 Indexes\n\n- tables > 10 million rows: CONCURRENTLY"
    dumped = "## 5.4 Indexes\n\n- CONCURRENTLY for large tables\n- 10; 7.3"
    assert orphaned_numbers(orig, good) == []
    assert any("number 10" in m for m in orphaned_numbers(orig, dumped))
    assert any("number 10" in m for m in missing_facts(orig, dumped))              # wired into the main check
    assert orphaned_numbers("A timeout of 30 seconds applies.", "- 30-second timeout") == []
    assert orphaned_numbers("At most 3 times.", "- max 3 retries") == []            # synonym swap on a real line: not flagged
    assert orphaned_numbers("At most 3 times.", "- 3") != []                        # a bare number is flagged
