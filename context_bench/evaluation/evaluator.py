"""Run the same tests against each representation; the model only ever sees that representation.

Static representations (original / compressed docs) put the whole text in one prompt. Navigation
representations (indexed / combined) give the model only index.md first; it must request sections by
id (agentic graph traversal, max 2 fetch rounds) and only the requested sections are ever shown.
A leakage guard rejects any prompt that contains original text the representation doesn't legitimately hold."""
from __future__ import annotations
import json, re, time
from dataclasses import dataclass, field
from pathlib import Path

ANSWER_SYS = ("Answer the question using only the provided document. Be concise, but include every specific "
              "value, condition and exception from the document that is relevant to the question. "
              "If the document does not contain the answer, reply exactly: NOT FOUND")
NAV_SYS = ("You answer questions about a large document that you can only access through a navigation index. "
           "Use only text you have been given; if it does not contain the answer, the answer is: NOT FOUND. "
           "Reply with JSON only.")


class LeakageError(RuntimeError):
    pass


def _ws(s: str) -> str:
    return " ".join(s.split())


@dataclass
class Representation:
    name: str
    kind: str                                   # "static" | "navigate"
    text: str = ""                              # static: full context shown to the model
    index_md: str = ""                          # navigate
    sections: dict[str, str] = field(default_factory=dict)   # navigate: id -> text that will be served
    titles: dict[str, str] = field(default_factory=dict)
    related: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    ranges: dict[str, tuple[int, int]] = field(default_factory=dict)   # section id -> (start_line, end_line) in the ORIGINAL
    meta: dict = field(default_factory=dict)

    def all_text(self) -> str:
        """Everything a model could ever see from this representation (for fact-survival metrics)."""
        return self.text if self.kind == "static" else self.index_md + "\n" + "\n".join(self.sections.values())

    def context_tokens(self, tok) -> int:
        return tok.count(self.text) if self.kind == "static" else tok.count(self.index_md)


class LeakGuard:
    def __init__(self, original_text: str, original_sections: dict[str, str], min_chars: int = 200):
        self.orig = _ws(original_text)
        self.secs = {k: _ws(v) for k, v in original_sections.items() if len(_ws(v)) >= min_chars}

    def check(self, prompt: str, rep: Representation, is_original: bool = False) -> None:
        if is_original:
            return
        p, legit = _ws(prompt), _ws(rep.all_text())
        if self.orig in p and self.orig not in legit:
            raise LeakageError(f"[{rep.name}] prompt contains the full original document")
        for sid, s in self.secs.items():
            if s in p and s not in legit:
                raise LeakageError(f"[{rep.name}] prompt contains original text of section {sid} "
                                   "that is not part of this representation")


def _json(text: str):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        return json.loads(m.group(0)) if m else None


@dataclass
class Answer:
    model: str
    rep: str
    test_id: str
    answer: str
    input_tokens: int
    output_tokens: int
    provider_input_tokens: int | None
    provider_output_tokens: int | None
    latency_sec: float
    turns: int
    fetched: list[str] = field(default_factory=list)
    error: str | None = None
    cached: bool = False


class _Meter:
    def __init__(self, client, tok, guard, rep, is_original):
        self.c, self.tok, self.guard, self.rep, self.orig = client, tok, guard, rep, is_original
        self.i = self.o = 0; self.pi = self.po = 0; self.lat = 0.0; self.turns = 0; self.cached = True

    def ask(self, prompt, system, max_tokens=700, json_mode=False):
        self.guard.check(system + "\n" + prompt, self.rep, self.orig)
        r = self.c.generate(prompt, system=system, max_tokens=max_tokens, json_mode=json_mode)
        self.i += self.tok.count(system + "\n" + prompt); self.o += self.tok.count(r.text)
        self.pi += r.input_tokens or 0; self.po += r.output_tokens or 0
        self.lat += r.latency_sec; self.turns += 1; self.cached &= r.cached
        return r.text


def _fmt_sections(rep, ids):
    return "\n\n".join(f"[{i}] {rep.titles.get(i, '')}\n{rep.sections[i]}" for i in ids)


def _valid_ids(rep, obj, limit):
    ids = obj.get("read") if isinstance(obj, dict) else None
    out = []
    for i in ids or []:
        i = str(i).strip().strip("[]")
        if i in rep.sections and i not in out:
            out.append(i)
    return out[:limit]


def answer_static(m: _Meter, rep: Representation, q: str) -> tuple[str, list[str]]:
    return m.ask(f"<document>\n{rep.text}\n</document>\n\nQuestion: {q}", ANSWER_SYS).strip(), []


def answer_navigate(m: _Meter, rep: Representation, q: str) -> tuple[str, list[str]]:
    p1 = (f"<index>\n{rep.index_md}\n</index>\n\nQuestion: {q}\n\nWhich sections must you read to answer? "
          'Reply JSON: {"read": ["<id>", ...]} with at most 4 ids, most relevant first.')
    fetched = _valid_ids(rep, _json(m.ask(p1, NAV_SYS, 200, True)), 4)
    if not fetched:
        return "NOT FOUND", []
    for hop in range(2):
        rel = [f"{i} ({r})" for f in fetched for i, r in rep.related.get(f, []) if i not in fetched]
        last = hop == 1
        p = (f"Sections you requested:\n\n{_fmt_sections(rep, fetched)}\n\n"
             + (f"Other sections related to these (not yet read): {', '.join(dict.fromkeys(rel))}\n\n" if rel and not last else "")
             + f"Question: {q}\n\n"
             + ('Answer now, concisely, including every specific value, condition and exception the text gives. '
                'Reply JSON: {"answer": "..."} (use "NOT FOUND" if the text lacks the answer).' if last else
                'If you can answer fully, reply JSON: {"answer": "..."} (concise, with every relevant value, '
                'condition and exception). If you need more sections, reply {"read": ["<id>", ...]} (max 3).'))
        raw = m.ask(p, NAV_SYS, 700, True)
        obj = _json(raw)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"]).strip(), fetched
        more = [i for i in _valid_ids(rep, obj, 3) if i not in fetched] if isinstance(obj, dict) else []
        if isinstance(obj, dict) and "read" in obj and more and not last:
            fetched += more
            continue
        return (raw.strip() if not isinstance(obj, dict) else "NOT FOUND"), fetched
    return "NOT FOUND", fetched


def run_answers(reps: list[Representation], tests: list[dict], client, tok, guard: LeakGuard,
                out_path: Path, log=print, max_consecutive_errors: int = 4) -> list[Answer]:
    """Answer every test against every representation. Resumable: finished (model,rep,test) rows in
    out_path are skipped. Stops early for a client after repeated errors (e.g. quota) and reports it."""
    out_path = Path(out_path)
    done: dict[tuple, Answer] = {}
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            a = Answer(**json.loads(line))
            if a.error is None:
                done[(a.model, a.rep, a.test_id)] = a
    model = client.label
    results: list[Answer] = list(done.values())
    consec = 0
    with out_path.open("a") as fh:
        for rep in reps:
            for t in tests:
                key = (model, rep.name, t["id"])
                if key in done:
                    continue
                m = _Meter(client, tok, guard, rep, rep.name == "original")
                try:
                    ans, fetched = (answer_static if rep.kind == "static" else answer_navigate)(m, rep, t["question"])
                    a = Answer(model, rep.name, t["id"], ans, m.i, m.o, m.pi or None, m.po or None,
                               m.lat, m.turns, fetched, None, m.cached)
                    consec = 0
                except LeakageError:
                    raise
                except Exception as e:                        # noqa: BLE001
                    a = Answer(model, rep.name, t["id"], "", m.i, m.o, None, None, m.lat, m.turns, [],
                               f"{type(e).__name__}: {str(e)[:200]}")
                    consec += 1
                fh.write(json.dumps(a.__dict__) + "\n"); fh.flush()
                results.append(a)
                if consec >= max_consecutive_errors:
                    log(f"[evaluate] {model}: {consec} consecutive errors (last: {a.error}); "
                        "stopping this model. Re-run later to resume from cache.")
                    return results
            log(f"[evaluate] {model} / {rep.name}: done")
    return results
