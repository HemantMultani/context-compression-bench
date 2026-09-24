# Quick start for a real document

Goal: produce candidate copies of the document (indexed / compressed / condensed), pick the ones that cut tokens with
no measured loss, then confirm with the real agent (Copilot / Claude / Codex) using the SAME questions.

Assumes the repo is cloned on the work laptop and the document is at `./inputs/my_doc.md`
(`inputs/` and `experiments/` are gitignored; never commit them).

## 0. One-time setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[compress,dev]"          # add ,gemini only if you use Gemini
pytest -q                                    # all tests should pass (no network/models needed)
cp .env.example .env                         # set OPENAI_BASE_URL / OPENAI_API_KEY for the approved company model (optional)
```

LLMLingua-2 needs `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (~2.2 GB). The tool never downloads it
without `--allow-download`; if huggingface.co is blocked, copy the Hugging Face cache from another machine.

## 1. Is it worth it? (2 minutes, no API)

```bash
python -m context_bench analyze inputs/my_doc.md
```

If the document is only a few thousand tokens, there is little to save (an index costs ~25% of a 4k-token doc; the
ratio improves as documents grow). Nothing forces you to use every arm.

## 2. Build the candidate copies

Any of these can be run separately; each writes into `experiments/my_doc/`.

```bash
# Indexed: a model plans the split (needs a capable model; a small local one fails and the tool says so)
python -m context_bench index inputs/my_doc.md --index-model openai:<model>
#   -> indexed/index.md + indexed/sections/*.md       (the "index + section files" copy)
#   -> indexed_heuristic/                              (deterministic heading-based copy, for comparison)
#   If it prints "LLM split FAILED", the LLM arm does not exist: use a stronger model.

# Compressed (LLMLingua-2): sweep, no forced ratio
python -m context_bench compress inputs/my_doc.md --method llmlingua2 --ratios 0.9,0.8,0.7
#   -> compressed/llmlingua2/r0.90.md, r0.80.md, r0.70.md

# Condensed: a model rewrites into terse agent-readable form, sections that lose a number/identifier/condition stay verbatim
python -m context_bench condense inputs/my_doc.md --condense-model openai:<model>
#   -> compressed/condensed/condensed.md   (use a strong model: a 7B local model saved only 10% and confused the reader)
```

No approved model API for the split or the rewrite? Have your coding agent write them and import them (the tool
validates both): `import-index plan.json --label <agent>` (the plan must cover every line of the document exactly; schema
in `README.md`/`AGENTS.md`) and `import-condensed condensed.md --label <agent>` (runs the same mechanical fact check).
Findings so far suggest the split and the rewrite need a capable model; small local ones did poorly.

## 3. Screen the arms with a local model (optional, unlimited)

```bash
python -m context_bench run inputs/my_doc.md \
    --index-model openai:<model> --gen-model openai:<model> --condense-model openai:<model> \
    --answer-models ollama:qwen2.5-coder:7b-instruct-q4_K_M \
    --arms original,llmlingua2@0.9,llmlingua2@0.8,llmlingua2@0.7,condensed,indexed,indexed_heuristic,combined@0.8
python -m context_bench judging-sheet inputs/my_doc.md      # every arm's answers per question
# grade it (you, or paste it to a coding agent), save {"<test id>": {"pass": [arms], "partial": [...], "fail": [...]}}
python -m context_bench import-verdicts inputs/my_doc.md verdicts.json --judge-label me
python -m context_bench report inputs/my_doc.md --arms <same list>
```

Read `experiments/my_doc/report.md`: the **"Loss versus the original"** table lists, per arm, the questions the
original answered correctly that the arm gets wrong. Zero regressions on ~40 questions is a good sign, not proof.
A small local model is a stress test and can mislead in either direction (e.g. it answers NOT FOUND on questions that
need inference); use it to drop clearly bad arms, and decide with step 4.

Without any API: `generate-tests` (draft candidates), edit `test_cases.json` or pass `--tests my_questions.json`
(see README for the format), and `evaluate --manual` + `import-manual`.

## 4. Confirm with the real agent (Copilot A/B)

```bash
python -m context_bench export-questions inputs/my_doc.md
#   -> experiments/my_doc/questions.md  (ask these)     answer_key.md  (grade against)
```

For each of 2-3 finalist arms **and the original**, open a NEW chat with a clean context:
- single-file arms: put only that file in context (or in the repo as the instructions file);
- `indexed`: copy `indexed/` (index.md + sections/) into the repo and tell the agent "read index.md first and open
  only the section files you need".
Ask every question, keep the answers, grade against `answer_key.md` (pass / partial / fail). Better still, add 10-15
questions from real PR reviews (a diff or code snippet plus "what review comments does the guide require?").
Choose the arm with the largest token reduction whose grades match the original's.

To grade answers an agent wrote into a file (`### [TEST-ID]` blocks): `import-answers answers.md --arm <arm> --model <label>`,
then `judging-sheet --blind --model <label>` and `import-verdicts verdicts.json --blind --model <label>`, then `report`.
Also ask the agent to list which section files it opened (indexed arm) and note the real token usage from the tool you
test in: token usage was not measured in the first trial, so we could not compare arms on cost.

## 5. If something breaks

- `ModelNotCached`: add `--allow-download` or copy the Hugging Face cache.
- `LLM split FAILED`: use a stronger `--index-model` (the split needs a capable model).
- 429 / daily quota: cached calls are not repeated; judged batches are saved; re-run later or use another model.
- The condensed arm keeps many sections verbatim: the model dropped a number/identifier/condition (listed in the log).
