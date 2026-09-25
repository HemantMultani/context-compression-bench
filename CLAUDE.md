# CLAUDE.md

@AGENTS.md

The rules above (privacy, no secrets in git, isolation of test contexts, honest reporting) apply to every task. This file adds notes for working on the code.

## Commands

```bash
source .venv/bin/activate
pytest -q                                   # 69 tests, offline: fake LLMs and a fake compressor, no models or network
python -m context_bench --help              # analyze | index | compress | condense | generate-tests | evaluate | run
                                            # judging-sheet | import-verdicts | import-answers | import-index | import-condensed
                                            # export-questions | import-manual | report
```

## Layout

- `context_bench/cli.py`: argparse commands. `experiment.py`: the `Experiment` class that runs each step inside `experiments/<doc>/` (all pipeline logic lives here).
- `indexing/`: `plan.py` (split-plan schema, validation, index rendering, writing section files), `llm_splitter.py` (LLM plans over numbered blocks), `markdown_index.py` (deterministic splitter).
- `compression/`: `base.py` (structure-aware driver: headings and code stay verbatim), `llmlingua2.py`, `llmlingua.py`, `condense.py` (LLM rewrite plus mechanical fact checks), `_hf.py` (dependency and model-cache checks, never downloads silently).
- `evaluation/`: `test_generator.py` (grounded questions), `evaluator.py` (answering, navigation loop, leakage guard), `judge.py`, `metrics.py`.
- `models/`: providers (`ollama:`, `gemini:`, `openai:` specs) behind one `LLMClient` with disk cache, rate limit and retry.
- `tests/conftest.py`: `FakeLLM`, `FakeCompressor`, `block_plan_llm`, `terse_llm`. Reuse them for new tests.

## Invariants (do not break; each has a test)

- Section files re-concatenate to the original **byte for byte**. The LLM only proposes boundaries and routing notes; code slices the text.
- An LLM index that fails validation never silently becomes the heading-based one: the `indexed` arm just does not exist (`SplitFailed`, `strict=True`).
- The model under test sees only its arm. `LeakGuard` raises if a prompt contains original text the arm does not hold.
- Compression targets are retained ratios for the whole document; headings and code fences are kept, the prose rate is adjusted; the achieved ratio is always measured and reported.
- Navigation is scored by line-range overlap (`source_lines`), never by comparing section ids across different indexes.
- Answers are keyed `(model, arm, test)` in `answers/*.jsonl` and judgments in `judgments.json`. **If you replace an arm's content, archive that arm's answers and judgments first**, or stale results will be reused.
- Manual test questions cite the heading-based section ids printed by `analyze`.
- LLM calls are cached in `experiments/<doc>/.cache` (keyed by provider, model, prompt); a daily-quota error fails fast (`QuotaExhausted`) instead of retrying.

## Conventions

- Python 3.10+, `from __future__ import annotations`, small modules, comments explain why not what. No new dependency without a strong reason (`transformers` is pinned to 4.44.2 for llmlingua).
- Every bug fix gets a test. A metric that depends on string matching must be reported next to a judged score.
- Never add a real document, question set derived from one, result file or key to this repo. Fictional examples only.
