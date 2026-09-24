import json, re
from pathlib import Path
import pytest

from context_bench.analysis.markdown_parser import parse
from context_bench.analysis.tokenizer import CharApproxTokenizer
from context_bench.models.base import LLMClient, LLMResponse

ROOT = Path(__file__).resolve().parent.parent
SYNTH = ROOT / "examples" / "synthetic_engineering_guide.md"
MANUAL_TESTS = ROOT / "examples" / "synthetic_expected_tests.json"

SMALL = """Intro paragraph before any heading.

# Handbook

Welcome text.

## 1. Alpha Rules

Always use `alpha_call` when the timeout is 30 seconds. Never exceed 500 rows.

```python
# not a heading
x = 1
```

See Section 2 and section 2.1 for exceptions.

## 2. Beta Rules

Beta text with a value of 42.

### 2.1 Beta Exceptions

The exception applies only on the reporting replica.
"""


class FakeCompressor:
    """Deterministic: keeps ~rate of the words, spread evenly."""
    name, model, params = "fake", "fake-model", {"k": 1}

    def compress(self, text, rate):
        words = text.split(" ")
        keep = max(1, round(len(words) * rate))
        idx = {int(i * len(words) / keep) for i in range(keep)}
        return " ".join(w for i, w in enumerate(words) if i in idx)


class FakeLLM(LLMClient):
    provider, is_external = "fake", False

    def __init__(self, fn, model="fake", **kw):
        super().__init__(**kw)
        self.model, self.fn, self.prompts = model, fn, []

    def _call(self, prompt, system, max_tokens, json_mode):
        self.prompts.append((system, prompt))
        return LLMResponse(self.fn(prompt, system, json_mode), 10, 5, 0.01, self.model)


@pytest.fixture
def tok():
    return CharApproxTokenizer()


@pytest.fixture
def small_doc():
    return parse(SMALL)


@pytest.fixture
def synth_text():
    return SYNTH.read_text()


@pytest.fixture
def synth_doc(synth_text):
    return parse(synth_text)


@pytest.fixture
def manual_tests():
    return json.loads(MANUAL_TESTS.read_text())["tests"]


def block_plan_llm(per=10):
    """FakeLLM that answers the split prompt with valid sections of `per` blocks each."""
    def fn(prompt, system, j):
        nums = [int(x) for x in re.findall(r"\[B(\d+)\]", prompt.split("DOCUMENT BLOCKS:")[1])]
        lo, hi = min(nums), max(nums)
        secs, i = [], lo
        while i <= hi:
            e = min(hi, i + per - 1)
            secs.append({"id": f"S{len(secs) + 1}", "title": f"Part {len(secs) + 1}", "start_block": i, "end_block": e,
                         "parent": None, "summary": "s", "use_when": "u", "key_terms": [], "related": []})
            i = e + 1
        return json.dumps({"sections": secs})
    return FakeLLM(fn, model="splitter")


def terse_llm():
    """FakeLLM condenser: drops filler words but keeps every hard fact."""
    def fn(prompt, system, j):
        text = prompt.split("SECTION:\n", 1)[1]
        return "\n".join(l if l.startswith("#") else re.sub(r"\b(the|that|which|very)\s+", "", l) for l in text.splitlines())
    return FakeLLM(fn, model="condenser")
