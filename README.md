# context-compression-bench

An **experimental benchmark** that answers one question empirically:

> Can a large documentation/context file be made cheaper for an LLM (fewer tokens) **without losing the
> information the LLM needs**?

It is domain-agnostic (any Markdown/text file). It compares the original document against
(a) a structurally split + indexed version an agent navigates like a graph, (b) LLMLingua / LLMLingua-2
compressed versions, and (c) combinations, using the *same* auto-generated tests for every
representation. **High token reduction does not mean successful knowledge preservation**, and this tool is
built so it can show that compression hurts as easily as that it helps.

## 1. What problem this investigates

A large instruction document (coding rules, review guidelines, runbooks) loaded into an agent's context
on every request burns the token budget, even though most tasks need a small part of it. Options:

1. **Do nothing** (original).
2. **Index + navigate**: split the doc into sections, give the agent a short routing index, let it fetch
   only the sections it needs (fewer tokens per question, but the agent must navigate correctly).
3. **Compress** the text (drop low-information tokens).
4. **Combine** 2 and 3.

## 2. What prompt compression is, and LLMLingua

Prompt compression shortens text while trying to keep what a model needs. **LLMLingua** (Microsoft) drops
tokens a small language model finds predictable. **LLMLingua-2** trains a small encoder classifier
(distilled from GPT-4 judgments) to keep/drop each token; it is faster, better out of domain and is the
default here. Both are *extractive*: they only delete tokens, never rewrite. Deleting tokens can silently
delete a condition ("only when...", "never...") or a value, which is exactly what the tests probe.

Two properties matter for interpretation: compressors often **miss the target ratio** (the report always
shows the achieved ratio), and compression that is **question-agnostic** (as here: the document is
compressed before the task is known) is lossier than question-aware compression.

## 3. Install

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate      # or python -m venv
uv pip install -e ".[compress,gemini,dev]"                    # or: pip install -e ".[compress,gemini,dev]"
cp .env.example .env                                          # then fill only what you use
pytest -q                                                     # 50+ tests, no network/models needed
```

- `[compress]` installs `llmlingua`, `torch`, and pins `transformers==4.44.2` (the version `llmlingua` was
  written against; newer versions break v1's GPT-2 cache handling).
- Models are **not** downloaded silently. If a Hugging Face model is not cached, the tool stops and tells
  you the name and size; re-run with `--allow-download` to permit it (only model files are downloaded, your
  document is not uploaded). LLMLingua-2 default model: `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`
  (~2.2 GB). LLMLingua v1 default scorer: `openai-community/gpt2` (~0.5 GB; the paper used LLaMA-7B, set
  `--v1-model`).
- The default tokenizer is `tiktoken:o200k_base`. Your production model may tokenize differently: use
  `--tokenizer tiktoken:cl100k_base` or `hf:<model>`. All token numbers in a report state the tokenizer.

## 4. Run the synthetic experiment

```bash
python -m context_bench analyze examples/synthetic_engineering_guide.md
python -m context_bench run examples/synthetic_engineering_guide.md \
    --index-model gemini:gemini-3.5-flash \
    --gen-model   gemini:gemini-3.5-flash \
    --tests examples/synthetic_expected_tests.json \
    --answer-models ollama:qwen2.5-coder:7b-instruct-q4_K_M \
    --condense-model ollama:qwen2.5-coder:7b-instruct-q4_K_M \
    --arms original,llmlingua2@0.9,llmlingua2@0.8,llmlingua2@0.7,condensed,indexed,indexed_heuristic,combined@0.8
```

Output goes to `experiments/synthetic_engineering_guide/` (gitignored):

```text
original.md   compressed/{llmlingua,llmlingua2,combined}/   indexed/   indexed_heuristic/
test_cases.json   test_cases.rejected.json   answers/*.jsonl   judgments.json
results.json   report.md   manifest.json   .cache/ (LLM response cache; makes runs resumable)
```

## 5. Run on your own document

```bash
python -m context_bench analyze ./my_doc.md                                   # sizes, structure (no API)
python -m context_bench run ./my_doc.md \
    --index-model  <provider:model> --gen-model <provider:model> \
    --answer-models <provider:model> --judge-model <provider:model>
```

Individual steps: `analyze`, `index`, `compress` (`--method llmlingua|llmlingua2|both`, `--ratios`, `--target-ratio`,
`--target-tokens N`, `--combined`), `condense`, `generate-tests`, `evaluate`, `judging-sheet` + `import-verdicts`
(a person or coding agent grades), `export-questions` (questions + answer key for manual runs in Copilot/Claude/Codex),
`import-manual`, `report`. `--arms a,b,c` restricts which arms are evaluated/reported. **There is no forced ratio:** the
default is a sweep `0.9,0.8,0.7` (LLMLingua needs a rate), and the report shows how much can be cut before questions
start failing.

**Model specs** are `provider:model`: `ollama:<model>` (local), `gemini:<model>`, `openai:<model>` (any
OpenAI-compatible endpoint: set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, optionally `OPENAI_EXTRA_HEADERS` as JSON
for gateways that need extra headers). **No model at all** also works: `analyze`, `index` (heuristic),
`compress`, draft tests, and `evaluate --manual` (exports one prompt file per representation x test plus a
CSV to fill in; `import-manual` loads your verdicts).

Hand-written tests: `--tests file.json` = `{"tests": [{"id","category","question","expected_answer",
"source_sections":["S7.3"],"critical_facts":["exact phrase from the doc"],"difficulty":"easy|medium|hard"}]}`
with categories A-G (see below). `critical_facts` must be verbatim phrases; the tool warns when they are not
found in the cited sections. In manual tests, `source_sections` are the **heading-based ids printed by `analyze`**
(e.g. `S7.3` for the heading "7.3 ..."); the tool converts them to original line ranges (`source_lines`) so
navigation is scored fairly against any index (LLM-planned or heading-based).

## 6. How evaluation works

1. **Index** (`indexed/`): a model reads the document as numbered blocks and proposes sections, `parent`s,
   summaries, "use when" hints and typed links (`depends_on`, `exception_to`, `overrides`, `see_also`). It
   returns only boundaries and metadata; **the tool slices the original text**, so no content can be altered or
   lost (a test checks the sections re-concatenate byte-for-byte). Invalid plans (gaps, overlaps, invented
   block numbers) get one repair retry, then fall back to the deterministic heading-based splitter, loudly.
   `indexed_heuristic/` is that deterministic splitter, kept as a comparison.
2. **Tests** are generated from the *original* only, in 7 categories: A direct recall, B rule application,
   C exceptions, D multi-section, E negative constraints, F terminology, G exact values. Every
   `critical_fact` must appear in the sections the test cites, checked mechanically, not by an LLM; failures
   go to `test_cases.rejected.json`. The same saved tests are used for every representation.
3. **Representations (arms)**: `original`; `llmlingua2@R` (LLMLingua-2 per prose block; headings and code kept
   verbatim; `llmlingua@R` (v1) with `--methods llmlingua`); `condensed` (an LLM rewrites each section into terse
   agent-readable form; a mechanical check requires every number, identifier, code span, acronym, quote, heading and
   negation/condition marker to survive and each number to stay next to what it quantifies, else the section stays
   verbatim); `indexed` (LLM-planned split; agent gets `index.md`, requests sections by id, max 2 fetch rounds);
   `indexed_heuristic` (deterministic heading-based split; a separate arm: if the LLM split fails the `indexed` arm
   simply does not exist, it never silently becomes the heuristic one); `combined@0.8` (same navigation as
   `indexed`, fetched sections LLMLingua-2 compressed).
4. **No leakage**: the model only sees the representation under test. A guard raises if a prompt contains
   original text that the representation does not legitimately hold; navigation representations only ever
   show the sections the model requested.
5. **Judging**: an LLM judge labels each answer pass / partial / fail from the question, expected answer and
   critical facts (it never sees the representation). **These are model-judged, not objective truth.** The
   report puts them next to objective metrics: fact survival in the representation, critical-fact string match
   in the answer, and how often judge and string match agree.

## 7. Interpreting token reduction vs knowledge preservation

- `retained_ratio = compressed_tokens / original_tokens`, `reduction = 1 - retained_ratio`.
- **Fact survival** (objective) is an upper bound: a fact that is gone cannot be answered.
- Read the **per-category** table: compression typically hurts exceptions, exact values and multi-section
  questions before it hurts plain recall, and averages hide that.
- For `indexed`, compare *mean input tokens per question* (a fixed index is paid on every question) and the
  navigation table (did the model fetch the sections that held the answer?).
- The report never declares a "best" method. It lists the Pareto-efficient representations (not beaten on
  both tokens and score) and every failed/partial test with the expected answer, model answer and source
  section, so you can see *what* was destroyed.

## 8. Privacy

- Nothing is uploaded implicitly. With no model specs the tool is fully local. `ollama:` runs locally.
- Non-local providers (`gemini:`, `openai:`) print `[EXTERNAL API] ... will receive document content` and
  send section text and questions to that provider only when *you* pass such a spec. Use models your
  organisation has approved for the document's classification.
- Artifacts (`experiments/`, LLM cache, `.env`, `inputs/`, `*.private.md`) are gitignored. The repository
  contains only the fictional synthetic document. Never commit proprietary documents or experiment output.
- `.env` holds API keys; keep it out of git (it is in `.gitignore`).

## 9. Limitations

- Generated tests and judged verdicts inherit the weaknesses of the LLMs that produced them; small samples
  (tens of tests) make differences of a few points noise. Prefer the objective columns and read the failures.
- Token counts are tokenizer-specific; results depend on the answering model's size (small models are hurt
  more by compression than large ones).
- LLMLingua-2 sees at most 512 tokens at a time and forces every chunk to drop about the same fraction whatever its
  information density; it cannot see cross-section relationships. The `condensed` and `indexed` arms exist because of that.
- The mechanical fact check for `condensed` cannot see lost nuance in prose (e.g. "including administrators"); the
  questions and grading are what measure that.
- LLMLingua-2 is trained on meeting transcripts; on code, tables and identifier-heavy text it may drop
  exactly the tokens that matter. `--preserve-digits` protects numbers, not names or conditions.
- The index costs tokens on every question; for short documents it may cost more than it saves.
- Free-tier APIs are rate limited (Gemini free tier: about 20 requests/day *per model*, and retried 503s seem to
  count). Responses are cached and judged batches are saved as they finish, so interrupted runs resume; a daily
  quota error stops immediately instead of retrying. Large test sets need a paid/organisation endpoint or a local model.
- The objective answer match is a lenient number/keyword proxy for free-form answers: it can over-credit a right
  number in the wrong context and under-credit a correct paraphrase. Judged scores have their own biases. Look at both.
- Free local models (7B) cannot reliably do the LLM index split; use a stronger model, or the heuristic
  splitter.
- This is an experiment harness, not production infrastructure.

## 10. Further reading

- `TOMORROW.md`: step-by-step run for a real document.
- `docs/RESEARCH_FINDINGS.md`: what we measured so far (results, conclusions, mistakes, limitations).
- `AGENTS.md`: instructions for a coding agent working in this repo (privacy rules, commands, grading rubric).

## 11. License

MIT, see `LICENSE`. Copyright (c) 2026 Hemant Multani. Models you download (e.g. LLMLingua-2 from Hugging Face) have their own licenses; this repo does not include them.
