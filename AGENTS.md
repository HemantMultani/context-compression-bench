# Instructions for a coding agent working in this repo

This repo is `context-compression-bench`: a command-line benchmark that tests whether a long Markdown document can be made cheaper for an LLM (fewer tokens) without losing information the LLM needs. Read `README.md` first, `TOMORROW.md` for the step-by-step run, and `docs/RESEARCH_FINDINGS.md` for what has already been learned.

## Ground rules

- **Privacy first.** The document being tested may be proprietary. Never send it to any external API unless the user has explicitly configured that model (`--index-model`, `--gen-model`, `--answer-models`, `--condense-model`, `--judge-model`). The tool prints an `[EXTERNAL API]` notice when it does. Use only models the organisation has approved for that document.
- **Never commit** the document, `inputs/`, `experiments/`, `.env` or any API key. They are git-ignored; keep it that way. Do not paste the document into issues, PRs or chat logs.
- **Do not put the answer key or other arms in the test context.** When the user tests a version of the document with an agent, that agent must see only that version (its own empty folder, fresh chat).
- Report results faithfully: token reduction alone is not success. Say when a test was too easy, when a grader saw the answers, or when token usage was not measured.

## Setup and checks

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[compress,dev]"      # pins transformers==4.44.2 (needed by llmlingua)
pytest -q                                # no network or model downloads needed
```

LLMLingua models are downloaded only with `--allow-download`.

## Typical tasks

- **Run the experiment on a document**: follow `TOMORROW.md` (`analyze`, `index`, `compress`, `condense`, `run`, `judging-sheet`, `import-verdicts`, `report`).
- **Author an arm yourself** (when no approved model API is available): write a split plan JSON and run `import-index PLAN.json --label <you>`; write a condensed copy and run `import-condensed FILE.md --label <you>`. The tool validates the plan as an exact cover of the document and runs its mechanical fact check on the rewrite. Keep every heading, number, identifier, condition and exception.
- **Grade answers**: `judging-sheet --blind` then `import-verdicts --blind`. Pass = all critical facts stated correctly and nothing contradicted; partial = a required fact or condition is missing; fail = wrong or missing.
- **Load answers produced elsewhere**: `import-answers FILE.md --arm <arm> --model <label>`; the file must use `### [TEST-ID]` blocks.

## When changing code

Keep the tool domain-agnostic, keep tests green, add a test for every bug fixed, and never make a metric depend on the answer being an exact string match without also reporting a judged score.
