# Both-Intent Query Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For `both` questions, retrieve with an English query that contains only the 10-K part of the question, so price wording stops pulling the vector search toward valuation chunks.

**Architecture:** The classifier already returns `language` and `search_query`. A pure function `retrieval_query(question, intent, result)` decides which text goes to retrieval: the rewrite for `both` and for non-English questions, the original question otherwise. `generate_answer` keeps receiving the original question. The retrieval benchmark gains `--classify` / `--save-rewrites` / `--rewrites` so the effect is measured with real rewrites and can be replayed without the router.

**Tech Stack:** Python 3.12 (uv), LangGraph, pytest. No new dependencies.

Spec: `docs/superpowers/specs/2026-10-29-both-query-rewrite-design.md`. Branch: `feat/both-query-rewrite` (already created from `main`).

## Global Constraints

- Python 3.12 via uv. Run everything with `uv run`.
- Unit tests never call the network, the LLM or download a model. Tests that need the router sit behind `RUN_INTEGRATION=1` and the `integration` marker.
- `.env` and `data/` are never committed. Never print API keys.
- Commit messages end with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (second `-m`).
- Retrieval rule (spec 4.1): the text sent to retrieval is the stripped `search_query` when it is non-empty and (the effective intent is `both` or the language is `other`); otherwise the original question. English `filing` and `price` questions never use a rewrite. The effective intent is the one after ticker filtering.
- `generate_answer` and `structure` keep receiving the original question. `Deps` signatures do not change.
- The classifier prompt change must not move any intent: 30/30 on the golden set, the 14 advice phrasings refused and the 9 controls kept (`tests/test_advice_refusal.py`), and the older classifier tests in `tests/test_llm_smoke.py` still pass.
- Do not change the reranker, `postprocess.py`, the Chroma store or `data/`.
- Documentation prose is Vietnamese, plain, humanizer-compliant: no em or en dashes, no decorative bold labels, no not-X-but-Y contrasts, no one-line closers.
- The suite (`248 passed, 38 skipped` now) must stay green after every task.

## File Structure

```
src/finassist/graph.py            + resolve_intent, retrieval_query; classify_intent uses them
src/finassist/chains.py           CLASSIFY_SYSTEM: search_query rules for both
src/finassist/schemas.py          IntentResult.search_query description
eval/retrieval_eval.py            + rewrite_table, query_for, evaluate(query_of=), --classify/--save-rewrites/--rewrites
tests/test_retrieval_query.py     NEW  rules and graph wiring
tests/test_both_rewrite.py        NEW  integration: real classifier rewrites
tests/test_retrieval_eval.py      + rewrite helpers and CLI argument checks
reports/query_rewrites.json       NEW  rewrite table of the official run (replayable)
reports/retrieval_experiments.md  + experiment section and decision
reports/ragas_both_rewrite_{1,2}.{md,json}  NEW
```

---

### Task 1: `resolve_intent`, `retrieval_query` and graph wiring

**Files:**
- Modify: `src/finassist/graph.py`, `src/finassist/schemas.py`
- Test: `tests/test_retrieval_query.py`

**Interfaces:**
- Produces `resolve_intent(res: IntentResult) -> tuple[Intent, list[str]]` (covered tickers, upper-cased and deduplicated; an intent that needs data but has no covered ticker becomes `out_of_scope`) and `retrieval_query(question: str, intent: Intent, result: IntentResult) -> str`.
- `classify_intent` sets `state["search_query"] = retrieval_query(question, intent, res)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval_query.py`:
```python
import pytest

from finassist.graph import Deps, build_graph, resolve_intent, retrieval_query
from finassist.schemas import Chunk, FinanceAnswer, IntentResult, PriceInfo

Q = "What is NVIDIA's market capitalization now, and what was its total revenue last fiscal year?"
RW = "What was NVIDIA's total revenue in its most recent fiscal year?"


def res(intent="both", language="en", search_query=RW, tickers=("NVDA",)):
    return IntentResult(intent=intent, tickers=list(tickers), language=language, search_query=search_query)


@pytest.mark.parametrize("intent,language,sq,expected", [
    ("both", "en", RW, RW),
    ("both", "en", None, Q),
    ("both", "en", "   ", Q),
    ("both", "en", f"  {RW}  ", RW),
    ("filing", "en", RW, Q),
    ("price", "en", RW, Q),
    ("filing", "other", RW, RW),
    ("both", "other", RW, RW),
    ("filing", "other", "", Q),
])
def test_retrieval_query_rules(intent, language, sq, expected):
    assert retrieval_query(Q, intent, res(intent, language, sq)) == expected


def test_resolve_intent_filters_tickers_and_downgrades():
    assert resolve_intent(IntentResult(intent="filing", tickers=[" aapl ", "TSLA", "AAPL"])) == ("filing", ["AAPL"])
    assert resolve_intent(IntentResult(intent="both", tickers=["TSLA"])) == ("out_of_scope", [])
    assert resolve_intent(IntentResult(intent="out_of_scope", tickers=[])) == ("out_of_scope", [])


def run(question, result):
    seen = {}
    chunk = Chunk(chunk_id="NVDA-2026-7-000", text="Revenue 215,938", ticker="NVDA", fiscal_year=2026, item="7")

    def retrieve(query, tickers):
        seen["retrieve"] = (query, tickers)
        return [chunk]

    def generate(q, chunks, prices):
        seen["generate"] = q
        return "Revenue was 215,938 [NVDA-2026-7-000]."

    deps = Deps(
        classify=lambda q: result,
        retrieve=retrieve,
        price=lambda t: PriceInfo(ticker=t, price=200.0, as_of="2026-10-29T00:00:00+00:00"),
        generate=generate,
        structure=lambda q, a, cs, ps, it: FinanceAnswer(intent=it, answer=a),
    )
    return seen, build_graph(deps).invoke({"question": question})


def test_both_question_retrieves_with_the_rewrite_but_answers_the_original_question():
    seen, state = run(Q, res("both"))
    assert seen["retrieve"] == (RW, ["NVDA"])
    assert seen["generate"] == Q
    assert state["search_query"] == RW


def test_english_filing_question_ignores_a_rewrite():
    seen, _ = run(Q, res("filing"))
    assert seen["retrieve"][0] == Q


def test_both_without_a_rewrite_falls_back_to_the_question():
    seen, _ = run(Q, res("both", search_query=None))
    assert seen["retrieve"][0] == Q


def test_both_with_only_uncovered_tickers_is_refused_and_never_retrieves():
    seen, state = run(Q, res("both", tickers=("TSLA",)))
    assert "retrieve" not in seen
    assert state["output"].intent == "out_of_scope"
```

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run pytest tests/test_retrieval_query.py -q`
Expected: collection error `ImportError: cannot import name 'resolve_intent'`.

- [ ] **Step 3: Implement**

In `src/finassist/graph.py`, add above `build_graph` (after `_join`):
```python
def resolve_intent(res: IntentResult) -> tuple[Intent, list[str]]:
    """Keep only covered tickers; an intent that needs data but has no covered ticker becomes out_of_scope."""
    tickers = [t for t in dict.fromkeys(x.strip().upper() for x in res.tickers) if t in TICKERS]
    intent = res.intent if (tickers or res.intent == "out_of_scope") else "out_of_scope"
    return intent, tickers


def retrieval_query(question: str, intent: Intent, result: IntentResult) -> str:
    """Text sent to retrieval. The classifier's English rewrite is used for `both` questions (it drops the
    price wording that pulls the search toward valuation chunks) and for non-English questions. Every other
    question, including English `filing` and `price` ones, is searched as asked. `intent` is the effective
    intent after ticker filtering."""
    rewritten = (result.search_query or "").strip()
    if rewritten and (intent == "both" or result.language == "other"):
        return rewritten
    return question
```
and replace the first four lines of `classify_intent` so it reads:
```python
    def classify_intent(state: GraphState) -> GraphState:
        res = deps.classify(state["question"])
        intent, tickers = resolve_intent(res)
        return {"intent": intent, "tickers": tickers,
                "search_query": retrieval_query(state["question"], intent, res),
                "retrieved_chunks": [], "price_data": [], "error": None, "short_circuit": False}
```
(delete the old `tickers = ...`, `intent = ...` and `sq = ...` lines and the old return). In `src/finassist/schemas.py` change the `search_query` description to:
`description="English question used for retrieval: the 10-K-only part of the question when intent is 'both', an English rewrite when language is 'other'"`.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/test_retrieval_query.py -q` then `uv run pytest -q`
Expected: all pass; the suite is green. Behaviour of the live system is unchanged at this point, because the current prompt returns `search_query` only for non-English questions.
```bash
git add src/finassist/graph.py src/finassist/schemas.py tests/test_retrieval_query.py
git commit -m "feat: retrieval_query and resolve_intent, rewrite used for both-intent questions" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Classifier prompt and live rewrite tests

**Files:**
- Modify: `src/finassist/chains.py`
- Test: `tests/test_both_rewrite.py`

**Interfaces:**
- Consumes `retrieval_query` rules (Task 1).
- Produces a `CLASSIFY_SYSTEM` whose `search_query` section asks for a 10-K-only English question when the intent is `both`.

- [ ] **Step 1: Write the integration test (fails against the old prompt)**

Create `tests/test_both_rewrite.py`:
```python
"""The real classifier must give `both` questions a search_query that holds only the 10-K part.

Needs the router: RUN_INTEGRATION=1.
"""
import os
import re

import pytest

from finassist.config import get_settings

pytestmark = pytest.mark.integration

PRICE_WORDS = re.compile(r"\b(price|prices|market cap\w*|p/e|pe ratio|stock price|share price)\b|giá|vốn hóa", re.I)
VIETNAMESE = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)

BOTH_CASES = [
    ("What is NVIDIA's current stock price and how do export controls threaten it?", ["NVIDIA"]),
    ("What is Apple's current stock price, and what were its total net sales last fiscal year?", ["Apple"]),
    ("What is NVIDIA's market capitalization now, and what was its total revenue last fiscal year?", ["NVIDIA"]),
    ("Compare Apple's and Microsoft's latest annual revenue and tell me both current stock prices.", ["Apple", "Microsoft"]),
    ("What is Microsoft's current P/E ratio, and what is its biggest competitive risk?", ["Microsoft"]),
    ("Giá cổ phiếu NVIDIA hiện tại là bao nhiêu và rủi ro kiểm soát xuất khẩu của họ ra sao?", ["NVIDIA"]),
    ("Vốn hóa của Microsoft hiện nay và doanh thu năm tài chính gần nhất là bao nhiêu?", ["Microsoft"]),
    ("Is Apple's stock up today, and what risks does its 10-K describe for the supply chain?", ["Apple"]),
]


@pytest.fixture(scope="module")
def classify():
    from finassist.chains import build_llm, make_deps

    s = get_settings()
    return make_deps(s, store=object(), llm=build_llm(s)).classify


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
@pytest.mark.parametrize("question,companies", BOTH_CASES)
def test_both_questions_get_an_english_filing_only_query(classify, question, companies):
    r = classify(question)
    assert r.intent == "both"
    sq = (r.search_query or "").strip()
    assert sq, "a both question must carry a search_query"
    assert not PRICE_WORDS.search(sq), sq
    assert not VIETNAMESE.search(sq), sq
    for name in companies:
        assert name.lower() in sq.lower(), (name, sq)
```
Run (router must be up): `RUN_INTEGRATION=1 uv run pytest tests/test_both_rewrite.py -q`
Expected: FAIL for the English cases (`a both question must carry a search_query`), because the current prompt returns `null` for English questions. Record which cases fail (RED evidence). Without `RUN_INTEGRATION`, all 8 are skipped.

- [ ] **Step 2: Change the prompt**

In `src/finassist/chains.py`, replace the last two lines of `CLASSIFY_SYSTEM` (the `language:` line and the `search_query:` line) with:
```
language: "en" if the question is written in English, "other" for any other language (e.g. Vietnamese).
search_query: the English question used to find passages in the 10-K text.
- When intent is "both": ALWAYS give a self-contained English question about the 10-K part only (revenue, EPS, net income, risks, MD&A, business). Remove every word about price, market capitalization, P/E, shares or the market. Keep the name of every company that is asked about. Examples: "What is NVIDIA's market capitalization now, and what was its total revenue last fiscal year?" -> "What was NVIDIA's total revenue in its most recent fiscal year?"; "Giá cổ phiếu Apple hiện tại và rủi ro chuỗi cung ứng của họ là gì?" -> "What supply chain risks does Apple disclose in its 10-K?".
- When language is "other" and intent is not "both": an English rewrite of the question that keeps company names and financial terms, so it can be matched against English 10-K text (example: "Rủi ro kiểm soát xuất khẩu đối với NVIDIA là gì?" -> "What are the export control risks for NVIDIA?").
- Otherwise (an English question that is not "both"): null."""
```
(the closing `"""` stays on the last line, as in the current file).

- [ ] **Step 3: Verify the rewrite tests**

Run: `RUN_INTEGRATION=1 uv run pytest tests/test_both_rewrite.py -q`
Expected: 8 passed.
The router may replay a cached answer keyed on the user message alone (an earlier task saw this). If cases still return `null` right after the prompt change although the prompt is correct, change the human message in `make_deps.classify` from `f"QUESTION\n{question}"` to `f"QUESTION (v2)\n{question}"` (a one-time cache buster), re-run, and say so in the report. If a case fails for a real reason (price word left, a company dropped), report the exact `search_query`; adjust the prompt examples once, and if it still fails mark that single case `xfail(strict=True)` with the observed rewrite as the reason and report it as a finding.

- [ ] **Step 4: Regression checks: no intent may move**

Run the classifier regression suite and the golden-set intent check:
```bash
RUN_INTEGRATION=1 uv run pytest tests/test_advice_refusal.py tests/test_llm_smoke.py -q
uv run python - <<'EOF'
import json
from finassist.chains import build_llm, make_deps
from finassist.config import get_settings

s = get_settings()
classify = make_deps(s, store=object(), llm=build_llm(s)).classify
golden = json.load(open("eval/golden_set.json", encoding="utf-8"))
bad = [(q["id"], q["expected_intent"], classify(q["question"]).intent)
       for q in golden if classify(q["question"]).intent != q["expected_intent"]]
print("intent mismatches:", bad)
print("OK 30/30" if not bad else "FAIL")
EOF
```
Expected: the advice and smoke tests pass (the same counts as before, with one xfail allowed only for the known q05 test elsewhere) and `OK 30/30`. If any intent moved, revert the prompt wording that caused it, or move the `both` instruction into a shorter sentence, and repeat until all pass; report what you changed.

- [ ] **Step 5: Unit suite and commit**

Run: `uv run pytest -q` (green; the 8 new tests are skipped without the router).
```bash
git add src/finassist/chains.py tests/test_both_rewrite.py
git commit -m "feat: classifier returns a 10-K-only English search_query for both-intent questions" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Benchmark options `--classify`, `--save-rewrites`, `--rewrites`

**Files:**
- Modify: `eval/retrieval_eval.py`
- Test: `tests/test_retrieval_eval.py`

**Interfaces:**
- Consumes `resolve_intent`, `retrieval_query` (Task 1), `IntentResult`.
- Produces `rewrite_table(golden, classify) -> dict[str, dict]` (id to `IntentResult.model_dump()` for every non-trap question), `query_for(q: dict, table: dict | None) -> str`, `evaluate(golden, retrieve, query_of=None)` where `query_of(q) -> str` supplies the query (default: the question), and the CLI options. `--classify` and `--rewrites` are mutually exclusive; `--save-rewrites PATH` needs `--classify`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieval_eval.py`:
```python
def test_query_for_applies_the_graph_rules():
    from eval.retrieval_eval import query_for

    table = {
        "q21": {"intent": "both", "tickers": ["AAPL"], "language": "en", "search_query": "What were Apple's net sales?"},
        "q01": {"intent": "filing", "tickers": ["AAPL"], "language": "en", "search_query": "ignored"},
        "q90": {"intent": "both", "tickers": ["TSLA"], "language": "en", "search_query": "rewrite"},
    }
    assert query_for({"id": "q21", "question": "price and revenue?"}, table) == "What were Apple's net sales?"
    assert query_for({"id": "q01", "question": "net sales?"}, table) == "net sales?"
    assert query_for({"id": "q90", "question": "tesla?"}, table) == "tesla?"      # uncovered ticker: out_of_scope
    assert query_for({"id": "q21", "question": "price and revenue?"}, None) == "price and revenue?"
    assert query_for({"id": "q77", "question": "x"}, table) == "x"                # not in the table


def test_rewrite_table_classifies_each_scored_question_once():
    from eval.retrieval_eval import rewrite_table
    from finassist.schemas import IntentResult

    golden = [{"id": "q1", "category": "numeric", "question": "a"}, {"id": "q2", "category": "trap", "question": "b"}]
    calls = []

    def classify(q):
        calls.append(q)
        return IntentResult(intent="filing", tickers=["AAPL"])

    table = rewrite_table(golden, classify)
    assert calls == ["a"]
    assert table == {"q1": IntentResult(intent="filing", tickers=["AAPL"]).model_dump()}


def test_evaluate_uses_query_of_for_retrieval_and_cache_key():
    golden = [{"id": "q1", "category": "combined", "question": "price and revenue?",
               "evidence": [{"ticker": "AAPL", "contains": ["416,161"]}]}]
    seen = []

    def retrieve(query, ticker):
        seen.append(query)
        return [ch("Total net sales 416,161")]

    rows, _ = evaluate(golden, retrieve, query_of=lambda q: "net sales?")
    assert seen == ["net sales?"]
    assert rows[0]["ranks"] == [1]


def test_main_rejects_conflicting_rewrite_options():
    import pytest as _pytest

    from eval.retrieval_eval import main

    with _pytest.raises(SystemExit):
        main(["--label", "x", "--classify", "--rewrites", "r.json"])
    with _pytest.raises(SystemExit):
        main(["--label", "x", "--save-rewrites", "r.json"])
```

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run pytest tests/test_retrieval_eval.py -q`
Expected: new tests fail with `ImportError` / `unexpected keyword argument 'query_of'`.

- [ ] **Step 3: Implement**

In `eval/retrieval_eval.py` add imports `from finassist.graph import resolve_intent, retrieval_query` and `from finassist.schemas import IntentResult`. Add after `evaluate`'s helper section (before `_pct`):
```python
def rewrite_table(golden: list[dict], classify) -> dict[str, dict]:
    """Classify every scored question once: {id: IntentResult as a dict}. Traps are skipped."""
    return {q["id"]: classify(q["question"]).model_dump() for q in golden if q["category"] != "trap"}


def query_for(q: dict, table: dict[str, dict] | None) -> str:
    """The query the app would send to retrieval for this question, given a rewrite table."""
    if table is None or q["id"] not in table:
        return q["question"]
    res = IntentResult.model_validate(table[q["id"]])
    intent, _ = resolve_intent(res)
    return retrieval_query(q["question"], intent, res)
```
Change `evaluate` to
```python
def evaluate(golden: list[dict], retrieve, query_of=None) -> tuple[list[dict], list[float]]:
    """retrieve(query, ticker) -> ranked list[Chunk]. query_of(q) -> the query text (default: the question)."""
    cache: dict[tuple[str, str], list[str]] = {}
    rows: list[dict] = []
    times: list[float] = []
    for q in golden:
        if q["category"] == "trap":
            continue
        text = query_of(q) if query_of else q["question"]
        ranks = []
        for g in q["evidence"]:
            key = (text, g["ticker"])
            if key not in cache:
                t0 = time.perf_counter()
                chunks = retrieve(text, g["ticker"])
                times.append((time.perf_counter() - t0) * 1000)
                cache[key] = [c.text for c in chunks]
            ranks.append(first_rank(cache[key], g["contains"]))
        rows.append({"id": q["id"], "category": q["category"], "ranks": ranks, "metrics": question_metrics(ranks)})
    return rows, times
```
(keep whatever import of `Chunk` or typing the file already has; do not re-add unused imports). In `main`, add the arguments
```python
    p.add_argument("--classify", action="store_true", help="rewrite queries with the real classifier (needs the router)")
    p.add_argument("--save-rewrites", help="with --classify: write the rewrite table to this JSON file")
    p.add_argument("--rewrites", help="replay a saved rewrite table instead of calling the router")
```
right after `--append`, validate right after `a = p.parse_args(argv)`:
```python
    if a.classify and a.rewrites:
        p.error("use --classify or --rewrites, not both")
    if a.save_rewrites and not a.classify:
        p.error("--save-rewrites needs --classify")
```
and replace the lines from `golden = json.loads(...)` through the `config = {...}` assignment with:
```python
    golden = json.loads((ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))
    table = None
    queries = "original questions"
    if a.classify:
        from finassist.chains import build_llm, make_deps

        table = rewrite_table(golden, make_deps(s, store, build_llm(s)).classify)
        queries = "classifier rewrite (live)"
        if a.save_rewrites:
            Path(a.save_rewrites).parent.mkdir(parents=True, exist_ok=True)
            Path(a.save_rewrites).write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    elif a.rewrites:
        table = json.loads(Path(a.rewrites).read_text(encoding="utf-8"))
        queries = f"classifier rewrite (replay of {Path(a.rewrites).name})"
    rows, times = evaluate(golden, lambda q, t: retriever.retrieve(q, [t]), query_of=lambda q: query_for(q, table))
    config = {"mode": s.retrieval_mode, "reranker": s.reranker, "candidates": s.candidates,
              "chroma_dir": s.chroma_dir.name, "queries": queries}
```
Update the module docstring usage line to list `[--classify [--save-rewrites PATH] | --rewrites PATH]`.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/test_retrieval_eval.py -q` then `uv run pytest -q`
Expected: green. Then confirm the default path is unchanged (needs the reranker model, cached): `uv run python -m eval.retrieval_eval --label "check"` (no `--append`) prints overall any@5 64%, all@5 36%.
```bash
git add eval/retrieval_eval.py tests/test_retrieval_eval.py
git commit -m "feat: benchmark can use the classifier's retrieval query (live or replayed)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Experiment and gate

**Files:**
- Create: `reports/query_rewrites.json`
- Modify: `reports/retrieval_experiments.md`

**Interfaces:**
- Consumes the benchmark options (Task 3) and the new prompt (Task 2). The router must be up.

- [ ] **Step 1: Run the official measurement and save the rewrites**

```bash
uv run python -m eval.retrieval_eval --label "both-rewrite: control, câu hỏi gốc (cấu hình cuối)" --show-misses --append reports/retrieval_experiments.md
uv run python -m eval.retrieval_eval --label "both-rewrite: truy vấn do bộ phân loại viết lại (chạy thật)" --classify --save-rewrites reports/query_rewrites.json --show-misses --append reports/retrieval_experiments.md
uv run python -m eval.retrieval_eval --label "both-rewrite: phát lại bảng đã lưu" --rewrites reports/query_rewrites.json --show-misses --append reports/retrieval_experiments.md
```
Expected: the control run reproduces any@5 64%, all@5 36%; the live run writes `reports/query_rewrites.json`; the replay run prints the same table as the live run (same hits; latency may differ). If the replay differs in any hit number, stop and report it.

- [ ] **Step 2: Show the rewrites that matter**

Print the table entries for q21 to q25 and, for every question where `search_query` is non-null and the effective intent is not `both` or the language is English (should be unused), list them:
```bash
uv run python - <<'EOF'
import json
t = json.load(open("reports/query_rewrites.json", encoding="utf-8"))
for k in ("q21", "q22", "q23", "q24", "q25"):
    print(k, t[k]["intent"], t[k]["tickers"], "|", t[k]["search_query"])
print("non-null search_query on other questions:",
      {k: v["search_query"] for k, v in t.items() if v["search_query"] and k not in ("q21", "q22", "q23", "q24", "q25")})
EOF
```

- [ ] **Step 3: Apply the gate (spec 4.4) and write the decision**

Keep the change when, in the live run, the `combined` row has any@5 of at least 60% (was 40%) and the overall any@5 is at least 64% with `numeric` and `risk` any@5 not lower than the control (60% and 60%). Append to `reports/retrieval_experiments.md`:
```markdown
### Quyết định cho viết lại truy vấn câu both

<Kept or dropped>: <real numbers: combined any@5 before and after, overall any@5 and all@5 before and after, MRR, per category, which of q21 to q25 changed rank and from what to what>. <The rewrites of q21 to q25 quoted from the table, one line each>. <Distance to the 70% and 40% spec targets after the change.> <Any rewrite that kept a price word, dropped a company or looked wrong.>
```
Write the filled version with real numbers (no angle brackets). Note that each combined question is 20 points inside its group and 4 points overall, and that the 20 other questions are unchanged by construction (state whether the numbers confirm that).

- [ ] **Step 4: If the gate FAILS, disable the both rule**

Change `retrieval_query` in `src/finassist/graph.py` so the condition is `if rewritten and result.language == "other":`, update the docstring, remove the `("both", "en", RW, RW)` and `("both", "en", f"  {RW}  ", RW)` cases from `tests/test_retrieval_query.py` (replace them with `("both", "en", RW, Q)`), change `test_both_question_retrieves_with_the_rewrite...` to expect the original question, run `uv run pytest -q`, and revert the prompt wording that asks for a `both` rewrite only if it moved an intent. Record the failure and the reason in the decision section, skip Tasks 5, and go to Task 6 documenting that the rewrite was measured and not adopted. (If the gate passes, skip this step.)

- [ ] **Step 5: Commit**

```bash
git add reports/query_rewrites.json reports/retrieval_experiments.md src tests
git commit -m "docs: both-intent query rewrite experiment and decision" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Full RAGAS comparison (only if the gate passed)

**Files:**
- Create: `reports/ragas_both_rewrite_1.{md,json}`, `reports/ragas_both_rewrite_2.{md,json}`
- Modify: `reports/retrieval_experiments.md`

**Interfaces:**
- Consumes the adopted rule (Task 4). Each run takes about 15 to 20 minutes; run them one after the other in the background and wait.

- [ ] **Step 1: Run RAGAS twice**

```bash
uv run python -m eval.run_ragas --tag both_rewrite_1
uv run python -m eval.run_ragas --tag both_rewrite_2
```
Expected: four report files. In each `.md`, replace every `TO ANALYZE` in the worst-case section with a short root cause based on the stored per-question data (which chunks were retrieved, whether the evidence was among them, what the answer said). Do not edit scores.

- [ ] **Step 2: Compare and judge honestly**

Append a table to `reports/retrieval_experiments.md` comparing `ragas_retrieval_v2`, `ragas_demo_check` (the two earlier runs of the previous configuration) and the two new runs: faithfulness, context precision, context recall, answer relevancy (overall and per category), intent accuracy, numeric sanity, runtime. Judge the spec 4.5 criteria and state each as met or not met: combined context recall above the best earlier run (0.267), faithfulness overall not down by more than 0.03, and none of q21 to q25 losing evidence it had. State plainly which differences are inside the run-to-run noise (the two runs of the same configuration differ by under 0.02).

- [ ] **Step 3: Commit**

```bash
git add reports
git commit -m "docs: RAGAS comparison for the both-intent query rewrite" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Documentation and final verification

**Files:**
- Modify: `docs/ARCHITECTURE.md`, `docs/NEXT_STEP.md`, `README.md`, `docs/KNOWLEDGE_CHECK.md`, `docs/DEMO_SCRIPT.md`, `docs/TEAM_LOG.md`, `docs/README.md` (only what is stale)

**Interfaces:**
- Consumes the decision of Task 4 and, if run, Task 5. Write the documents to match what was adopted; if the rewrite was not adopted, describe it as measured and not adopted.

- [ ] **Step 1: Update the documents (Vietnamese, humanizer rules from Global Constraints)**

- `docs/ARCHITECTURE.md`: node table row for `classify_intent` and `retrieve` (the query sent to retrieval); section 4.4 ("Câu hỏi tiếng Việt") and its diagram so they describe the general rule (`search_query` for `both` and for non-English questions; original question otherwise; `generate_answer` always gets the original), with a diagram check; section 4.3 (what the classifier prompt asks for); section 7.6 decisions table (one row for the rewrite with the measured numbers); section 8 limitations (rewrite quality depends on the classifier; measured on 5 questions); section 9 (the new benchmark options and `reports/query_rewrites.json`).
- `docs/NEXT_STEP.md`: mark the `both` direction done with the numbers and the verdict; keep the remaining directions; update the status table of the benchmark (`any@5`, `all@5`) with the new values.
- `README.md`: the benchmark command with `--classify` and `--rewrites`, and the updated test counts.
- `docs/KNOWLEDGE_CHECK.md` and `docs/DEMO_SCRIPT.md`: update numbers and the retrieval explanation; re-verify every `file:line` reference by opening the file; keep the Debug cases true to the new results (read `reports/ragas_both_rewrite_*.json` before writing about any question).
- `docs/TEAM_LOG.md`: add the new commits from `git log`.
Validate every mermaid block you touch by rendering it in a browser (a temporary HTML page with `mermaid@10.9.1` from `cdn.jsdelivr.net`, served from the scratchpad directory) and take a screenshot of the changed diagram.

- [ ] **Step 2: Final verification**

```bash
uv run pytest -q                                                              # green
RUN_INTEGRATION=1 uv run pytest tests/test_both_rewrite.py tests/test_advice_refusal.py tests/test_llm_smoke.py tests/test_retrieval_integration.py -q
git ls-files | grep -E '^\.env$' || echo ".env not tracked"
git grep -nE 'sk-[A-Za-z0-9-]{12,}|lsv2_pt_' -- . ':!docs/DEMO_SCRIPT.md' ':!docs/superpowers' || echo "no keys"
uv run finassist ask x --replay runs/demo.json
```
Also check that every relative Markdown link in `README.md` and `docs/*.md` resolves, and that the edited docs contain no em or en dashes outside code and mermaid.

- [ ] **Step 3: Commit**

```bash
git add README.md docs
git commit -m "docs: describe the both-intent retrieval query and its measured effect" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage (self-review)

| Spec section | Task |
|---|---|
| 4.1 retrieval query rule, `retrieval_query` | 1 |
| 4.2 classifier prompt | 2 |
| 4.3 benchmark with live and replayed rewrites | 3, 4 |
| 4.4 gate | 4 (with the disable path in step 4) |
| 4.5 RAGAS verification | 5 |
| 5 repo changes, docs | 6 |
| 6 tests (unit rule tests, graph wiring, benchmark helpers, integration rewrite quality, classifier regressions) | 1, 2, 3 |
| 7 risks (intent drift, lost company names, rewrite quality vs hand-written, small sample) | 2 (regression checks, company-name assertions), 4 (real rewrites, honest decision), 5 |

Execution dependencies: Tasks 1 and 3 need no router. Tasks 2, 4 and 5 need the router. Task 5 needs about 40 minutes of RAGAS time and is skipped if the gate in Task 4 fails.
