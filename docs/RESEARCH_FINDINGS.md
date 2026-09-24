# Research findings: shrinking long documents for coding agents

Session date: 2026-09-24. Status: finished, results are **preliminary** (one small fictional document, one strong model, see Limitations).

## 1. Question and short answer

**Question.** Can a long instruction document that a coding agent reads on every request be made cheaper in tokens without losing information the agent needs?

**Short answer.**
- **Nothing we tried is provably lossless**, and none of it could be ranked properly by the end: a strong model (Gemini 3.8 Flash, high reasoning) answered all 39 test questions correctly from *every* version, so the questions could not tell the versions apart.
- With a **weak reader** (Qwen 7B), the differences were large: the sensible arms cost 0 to 14 points of accuracy (the 14 at 70% kept), and a badly made one (the heading-based index) cost 28.
- **LLMLingua-2** is the best of the automatic compressors we tested, but it forces roughly the same drop on every chunk of text whatever it says.
- **Splitting the document into sections behind a short index** saves the most tokens per question (about 58%) and is lossless by construction (code does the cutting). What the agent does with it is the risk.
- **Rewriting sections tersely with a strong LLM** saved about 20% with no measured loss; the same rewrite by a weak LLM saved 10% and hurt.

## 2. The versions of the document we compared ("arms")

| Arm | What it is |
|---|---|
| `original` | The document as written. |
| `llmlingua2@R` | LLMLingua-2 token deletion, keeping fraction R of tokens (0.9, 0.8, 0.7). Headings and code blocks kept verbatim. |
| `llmlingua@R` | Original LLMLingua (v1) with a GPT-2 scorer. Poor; dropped from the default run. |
| `condensed` | An LLM rewrites each section tersely for an agent (`key: value`, `condition -> rule`, no filler). A mechanical check requires every number, identifier, code span, acronym, quote, heading and negation/condition marker to survive, and each number to stay next to what it quantifies. A section failing the check stays verbatim. |
| `indexed` | An LLM plans the split (only boundaries and routing notes); code cuts the text, so nothing can be altered. The agent reads a short `index.md` and opens only the section files it needs. |
| `indexed_heuristic` | Same idea, but split deterministically at headings (no LLM). |
| `combined` | `indexed` with each section also LLMLingua-2 compressed. |

## 3. LLMLingua family: what each one is

- **LLMLingua (v1)**: drops tokens a small language model finds predictable (low perplexity). Claims up to 20x on some tasks.
- **LongLLMLingua**: v1 plus awareness of the *question*; reorders and prunes context for long-context QA. Needs the question at compression time.
- **LLMLingua-2**: a token classifier (`xlm-roberta-large`, 24 layers, 1024 hidden, 16 heads, about 560M parameters, 2.2 GB) trained on GPT-4-made "keep/drop" labels from meeting transcripts. It reads the whole chunk both ways and outputs a keep probability per token. It never generates text. About 0.9 s per 2k-token document on an Apple M4.
- **SecurityLingua**: not a cost saver; extracts a prompt's "true intention" to help the target model resist jailbreaks.
- **Internals we verified in the library code**: LLMLingua-2 has a hard 512-token window and splits any text into chunks of at most about 510 tokens at `.` or newline. The keep threshold is a percentile computed **per chunk**, so every chunk loses about the same fraction whether it is dense rules or filler. It cannot see relations between sections.
- **Alternatives** (Selective-Context, PCRL, TACO-RL, AdaComp, lexical pipelines, commercial services): we found no independent benchmark showing any clearly beats LLMLingua-2. Vendor comparison guides are self-interested. One independent study reports compressors often miss their target ratio and that on short prompts the overhead can cancel the speed gain. That matches what we saw for v1.

## 4. Part A: early experiments (Wikipedia paragraphs, 40 documents of about 2k tokens)

Setup: each document is 12 SQuAD paragraphs, one holds the answer. Reader 1: an extractive QA model. Uncompressed F1 0.910. "kept" is the actual fraction of tokens kept.

| Method | ~82% kept | ~60% kept | ~40% kept | ~26% kept |
|---|---|---|---|---|
| LLMLingua-2 (no question) | 0.886 | 0.811 | 0.613 | 0.452 |
| LongLLMLingua (sees the question) | 0.856 | 0.814 | 0.855 | 0.804 |
| Truncate | 0.785 | 0.608 | 0.379 | 0.338 |
| Random token drop | 0.595 | 0.401 | 0.272 | 0.202 |
| LLMLingua v1 (GPT-2) | 0.525 (at 65% kept) | 0.285 (43%) | 0.126 (28%) | 0.080 (18%) |

Reader 2: a generative local model (Qwen2.5-coder 7B via Ollama), uncompressed F1 0.778:
LLMLingua-2 kept 82% -> 0.765, 60% -> 0.680, 41% -> 0.560 (answers "NOT FOUND" 17% of the time), 27% -> 0.340 (38% NOT FOUND). Latency fell from 18.1 s to 5.6 s per question as the prompt shrank.

Takeaways: LLMLingua-2 beats every simple baseline at matched size; a question-aware method is far better at low ratios but is not usable when the document is compressed before the task is known; generative readers are hurt more than extractive ones.

## 5. Part B: the benchmark tool and a fictional handbook (4,104 tokens)

Tool: `context-compression-bench` (this repo). It builds the arms above from any Markdown file, generates and grounds test questions from the original, runs every arm against the same questions, and reports paired losses.

Test set: 39 questions in 7 categories (direct recall, rule application, exceptions, multi-section, negative constraints, terminology, exact values): 27 hand-written (incl. 6 "review scenario" questions) and 12 generated by an LLM; 3 generated ones were rejected by an automatic check that each cited fact exists in the source.

Sizes: LLMLingua-2 at targets 0.9 / 0.8 / 0.7 achieved 90.4% / 80.6% / 70.8% kept; v1 at 0.8 achieved 93.6% (missed). Index sizes: heading-based 47 sections, 2,188 tokens (53% of the document); LLM-planned 11 to 13 sections, about 1,000 tokens (24%).

**Reader: Qwen 7B, one question per call, 39 questions.** Score = (pass + half of partial) / n, graded by an LLM (Claude). "Regressions" = questions the original answered correctly that this arm gets wrong (of 33).

| Arm | Input tokens per question | Saved | Score | Regressions |
|---|---|---|---|---|
| original | 4,179 | 0% | 86% | – |
| llmlingua2 @0.9 | 3,784 | 9% | 86% | 0 |
| llmlingua2 @0.8 | 3,382 | 19% | 83% | 2 |
| llmlingua2 @0.7 | 2,979 | 29% | 72% | 9 |
| indexed (plan by Gemini) | 1,774 | 58% | 86% | 2 |
| indexed (plan by Claude Sonnet 5) | 1,715 | 59% | 85% | 2 |
| condensed (rewrite by Qwen 7B) | 3,780 | 10% | 74% | 6 |
| condensed (rewrite by Claude Sonnet 5) | 3,340 | 20% | 85% | 1 |
| combined (index + LLMLingua-2 @0.8) | 1,667 | 60% | 76% | 6 |
| indexed_heuristic | 2,575 | 38% | 58% | 14 |

Navigation: with the LLM-planned index the model fetched the section holding the answer 95-97% of the time; with the heading-based index 74%.
Three "review scenario" questions were failed by **every** arm including the original, and a fourth by most: Qwen answers "NOT FOUND" when the answer needs inference. They cannot discriminate with a weak reader.

## 6. Part C: strong reader (Gemini 3.8 Flash, high, run in Google Antigravity)

Protocol: each arm in its own empty folder, fresh chat, all 39 questions in one prompt, answers written to a file. No `llmlingua2@0.7` run. Graded blind (arm names hidden and shuffled per question).

Result: **39/39 correct on every arm** (original, llmlingua2 @0.9 and @0.8, condensed, indexed). Objective keyword-match scores 96-99%. The strict "does the exact critical phrase still appear in the document" measure was 100% original, 97% @0.9, 96% @0.8, 99% condensed, 100% indexed, yet answers were still correct: a strong model repairs damaged text (for example "attempted most 3 times").
Token usage was **not measured** in this run. Only context sizes are known: 4,104 original, 3,709, 3,307, 3,265, and 1,029 for the index alone (plus whichever section files the agent opened).

## 7. Findings

1. No question-agnostic compressor is lossless. LLMLingua-2 is the best automatic one; loss is small at 80-90% kept and steep below about 60% for weak readers.
2. LLMLingua-2 beats truncation, random deletion, stopword removal and LLMLingua v1 at matched size. v1 with a small scorer is not competitive and misses its target ratio.
3. LLMLingua-2 cannot adapt to information density (uniform per-chunk drop, 512-token window) and can delete a condition ("only when") while leaving the sentence grammatical.
4. **Reader strength dominates the outcome.** The same 80%-kept LLMLingua-2 text cost Qwen 7B a few points and cost Gemini 3.8 Flash nothing measurable.
5. LLM-planned splitting is lossless by construction and cut about 58% of tokens per question. The planner's quality barely mattered (Gemini vs Sonnet: one question apart). The losses came from the small reader failing to use fetched sections.
6. A capable LLM found the same 11 sections in the document with all headings removed; the heading-based splitter collapsed it into 4 arbitrary chunks. A 7B model could not produce a valid split (it invented block numbers past the end); the validator rejected it.
7. The condense arm depends on the rewriter: 7B saved 10% and hurt (74%); a strong model saved 20% with no measured loss on either reader.
8. Rewriters game checks: told which numbers it dropped, the 7B model appended them on a stray line ("10; 7.3") to pass. A context check (number must sit next to what it quantifies) fixed that. Mechanical checks cannot see lost nuance in prose ("including administrators").
9. Compressing the index sections on top of indexing made things worse for the weak reader (86% -> 76%) for 2 more points of savings.
10. The index costs tokens on every question (about 24% of this 4k document), so small documents gain little from indexing.

## 8. Conclusions and recommendations

- For a **strong agent and modest cuts** (10-20%), LLMLingua-2 at 0.8-0.9 or a condensed rewrite by a strong model both looked safe on these questions. Indexing gives the largest saving; its real cost depends on how many sections the agent opens (not measured with the strong model).
- **This does not prove no loss.** The questions were too easy for a strong model, on a short fictional document.
- Suggested process for a real document: check size first; build the arms; screen with a local model only to drop clearly bad ones; decide with the real agent, using harder questions (realistic review scenarios, conditions/exceptions, details buried mid-paragraph); pick the largest reduction with **no regressions**; record real token usage.

## 9. Mistakes and bugs found along the way (worth knowing)

- A compression target of 1.0 still compressed (segment token counts do not sum to the whole).
- Navigation was scored against mismatched section ids (showed 25% instead of the true 93%); now scored by line-range overlap.
- The keyword scorer marked correct answers wrong when they ended in "number + period".
- Retrying a *daily* quota error for minutes wasted time; now fails fast.
- An early LLM split silently fell back to the heading-based one; now the LLM arm simply does not exist if the plan is invalid.
- Old grades and answers keyed by arm name would have been reused after replacing an arm; they must be archived first.
- `transformers` 5.x breaks LLMLingua v1 (GPT-2 cache format) and removed the QA pipeline; pin `transformers==4.44.2`.

## 10. Limitations and biases

- One fictional 4k-token document (plus Wikipedia text early on). Not code, not the real target document.
- The arms were written by the same model that graded them and that had seen the questions (best case). Grading was blind between arms, and a blind re-grade of the original matched the earlier grades on all 39, but that is not fully independent.
- One run per arm; Antigravity gave no control of sampling. All 39 questions were in one prompt, so answers may influence each other. Not comparable with the one-question-per-call Qwen numbers.
- Differences of one question (2.5 points) are noise. Objective and judged scores agreed 67-87% of the time with the weak reader.
- The Qwen model used is the coder variant, the only one available locally.

## 11. Practical notes

- **Gemini free tier**: about 20 requests per day **per model** for the flash models (plus 5 per minute), separate quota per model, and 503 "high demand" spikes; retried calls seem to count. Keys shared across projects share the quota. On the dashboard, "x / y" is used / allowed, red is at or over the limit.
- Environment: Python 3.12, `transformers==4.44.2`, LLMLingua models cached from Hugging Face (2.2 GB for LLMLingua-2); the tool never downloads without `--allow-download`.
- Privacy: nothing leaves the machine unless a non-local model is configured; the tool prints an `[EXTERNAL API]` notice when it does.
