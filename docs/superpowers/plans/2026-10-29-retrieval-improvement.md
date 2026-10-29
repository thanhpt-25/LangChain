# Retrieval Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the chunk that holds the answer into the top-5 chunks sent to the LLM, and prove it with a retrieval benchmark that runs in seconds without any LLM call.

**Architecture:** A new `Retriever` composes three optional layers behind one `retrieve(query, tickers)` function: dense vector search (current behaviour), BM25 merged by Reciprocal Rank Fusion, and a local cross-encoder reranker. `graph.py`, `Deps` and `postprocess.py` stay unchanged. Each layer is switched by an environment variable and is kept only if the benchmark shows it helps (experiment ladder, spec §3.4).

**Tech Stack:** Python 3.12 (uv), Chroma, `sentence-transformers` (`CrossEncoder`, already installed), `rank_bm25` (new), pytest.

Spec: `docs/superpowers/specs/2026-10-29-retrieval-improvement-design.md`. Branch: `feat/retrieval-improvement` (already created).

## Global Constraints

- Python 3.12 via uv. Run everything with `uv run`.
- Unit tests never call the network, the LLM or download a model. Use `DeterministicFakeEmbedding` and injected scorers. Tests that need the router or real models sit behind `RUN_INTEGRATION=1`.
- `.env` and `data/` are never committed. Never print API keys.
- Commit messages end with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (pass it as a second `-m`).
- Do not change `src/finassist/graph.py`, the `Deps` dataclass, or `src/finassist/postprocess.py`. `schemas.py` gets exactly one addition (`Chunk.context_prefix`).
- `top_k` stays 5 chunks per ticker sent to the LLM. Dense mode with no reranker must return exactly what `FilingStore.search(query, ticker, top_k)` returns today.
- `Chunk.score` keeps its meaning "Chroma distance, lower is closer". Chunks that only BM25 found have `score=None`; the reranker does not overwrite it.
- Retrieval benchmark ground truth is a list of evidence groups per question, never chunk ids: `{"ticker": "MSFT", "contains": ["133,749", "Net income"]}`. A group is satisfied by a retrieved chunk that contains every string (case-insensitive, whitespace-collapsed).
- Documentation prose is Vietnamese, plain, humanizer-compliant: no em or en dashes, no decorative bold labels, no not-X-but-Y contrasts, no one-line closers.
- The existing suite (`145 passed, 12 skipped`) must stay green after every task.

## File Structure

```
pyproject.toml                       + rank-bm25
src/finassist/config.py              + retrieval settings, FINASSIST_CHROMA_DIR
src/finassist/schemas.py             + Chunk.context_prefix
src/finassist/vectorstore.py         + embed_prefix, chunks_for, reset
src/finassist/retrieval.py           NEW  tokenize, rrf, make_scorer, Retriever, make_retriever
src/finassist/chains.py              make_deps uses Retriever
src/finassist/ingest.py              + table rows (opt-in)
src/finassist/cli.py                 + ingest --rebuild
eval/golden_set.json                 + "evidence" for 25 questions
eval/retrieval_eval.py               NEW  benchmark
eval/run_ragas.py                    + --tag
tests/test_retrieval.py              NEW
tests/test_vectorstore.py            + new store behaviour
tests/test_golden_evidence.py        NEW
tests/test_retrieval_eval.py         NEW
tests/test_ingest.py                 + table rows
tests/test_cli_replay.py             + ingest --rebuild parsing
reports/retrieval_experiments.md     NEW  one section per experiment
```

---

### Task 1: Settings, dependency and `Chunk.context_prefix`

**Files:**
- Modify: `pyproject.toml`, `src/finassist/config.py`, `src/finassist/schemas.py`
- Test: `tests/test_config.py`, `tests/test_schema.py`

**Interfaces:**
- Produces `Settings` fields (all with defaults, appended last): `retrieval_mode: str = "dense"`, `reranker: str = "none"`, `candidates: int = 30`, `embed_prefix: bool = False`, `table_rows: bool = False`.
- `Settings.chroma_dir` now honours env `FINASSIST_CHROMA_DIR` (default `<root>/data/chroma`).
- Produces `Chunk.context_prefix -> str`, for example `"AAPL FY2025 Item 8"`.

- [ ] **Step 1: Add the dependency**

Run: `uv add rank-bm25`
Expected: `pyproject.toml` and `uv.lock` updated, `rank-bm25` resolved. Then `uv run python -c "from rank_bm25 import BM25Okapi; print(BM25Okapi)"` prints the class.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_config.py`:
```python
def test_retrieval_settings_defaults(monkeypatch):
    for name in ("FINASSIST_RETRIEVAL", "FINASSIST_RERANKER", "FINASSIST_CANDIDATES",
                 "FINASSIST_EMBED_PREFIX", "FINASSIST_TABLE_ROWS", "FINASSIST_CHROMA_DIR"):
        monkeypatch.delenv(name, raising=False)
    s = get_settings()
    assert s.retrieval_mode == "dense"
    assert s.reranker == "none"
    assert s.candidates == 30
    assert s.embed_prefix is False
    assert s.table_rows is False
    assert s.chroma_dir.name == "chroma"


def test_retrieval_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FINASSIST_RETRIEVAL", "hybrid")
    monkeypatch.setenv("FINASSIST_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    monkeypatch.setenv("FINASSIST_CANDIDATES", "40")
    monkeypatch.setenv("FINASSIST_EMBED_PREFIX", "1")
    monkeypatch.setenv("FINASSIST_TABLE_ROWS", "1")
    monkeypatch.setenv("FINASSIST_CHROMA_DIR", str(tmp_path / "alt"))
    s = get_settings()
    assert (s.retrieval_mode, s.candidates) == ("hybrid", 40)
    assert s.reranker == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.embed_prefix is True and s.table_rows is True
    assert s.chroma_dir == tmp_path / "alt"
```
Append to `tests/test_schema.py`:
```python
def test_chunk_context_prefix():
    from finassist.schemas import Chunk
    c = Chunk(chunk_id="AAPL-2025-8-001", text="x", ticker="AAPL", fiscal_year=2025, item="8")
    assert c.context_prefix == "AAPL FY2025 Item 8"
```

- [ ] **Step 3: Run, expect FAIL**

Run: `uv run pytest tests/test_config.py tests/test_schema.py -q`
Expected: 3 failures (`AttributeError` on the new fields and `context_prefix`).

- [ ] **Step 4: Implement**

In `src/finassist/config.py` add the fields at the end of the `Settings` dataclass:
```python
    retrieval_mode: str = "dense"
    reranker: str = "none"
    candidates: int = 30
    embed_prefix: bool = False
    table_rows: bool = False
```
and in `get_settings()` change `chroma_dir=data / "chroma",` to `chroma_dir=Path(os.getenv("FINASSIST_CHROMA_DIR", str(data / "chroma"))),` and append to the `Settings(...)` call:
```python
        retrieval_mode=os.getenv("FINASSIST_RETRIEVAL", "dense"),
        reranker=os.getenv("FINASSIST_RERANKER", "none"),
        candidates=int(os.getenv("FINASSIST_CANDIDATES", "30")),
        embed_prefix=os.getenv("FINASSIST_EMBED_PREFIX", "0") == "1",
        table_rows=os.getenv("FINASSIST_TABLE_ROWS", "0") == "1",
```
In `src/finassist/schemas.py` add to class `Chunk`:
```python
    @property
    def context_prefix(self) -> str:
        return f"{self.ticker} FY{self.fiscal_year} Item {self.item}"
```

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest -q`
Expected: green, 148 passed and 12 skipped.
```bash
git add pyproject.toml uv.lock src/finassist/config.py src/finassist/schemas.py tests/test_config.py tests/test_schema.py
git commit -m "feat: retrieval settings, rank-bm25 and Chunk.context_prefix" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Store additions and the `Retriever`

**Files:**
- Modify: `src/finassist/vectorstore.py`
- Create: `src/finassist/retrieval.py`
- Test: `tests/test_vectorstore.py`, `tests/test_retrieval.py`

**Interfaces:**
- Consumes `Chunk` (Task 1), `Settings` (Task 1).
- Produces `FilingStore(persist_dir, embeddings, collection="filings", embed_prefix=False)` with the existing `upsert/search/count` plus `chunks_for(ticker) -> list[Chunk]` (sorted by `chunk_id`) and `reset() -> None` (drops the collection and recreates it empty). With `embed_prefix=True`, `upsert` embeds `f"{c.context_prefix}\n{c.text}"` but stores `c.text` as the document.
- Produces in `retrieval.py`: `tokenize(text) -> list[str]`, `rrf(rankings: list[list[str]], k: int = 60) -> list[str]`, `Scorer = Callable[[str, list[str]], list[float]]`, `make_scorer(model_name: str) -> Scorer | None`, `class Retriever(store, *, mode="dense", scorer=None, candidates=30, top_k=5)` with `retrieve(query: str, tickers: list[str]) -> list[Chunk]`, and `make_retriever(settings, store) -> Retriever`.

> Deviation from spec 3.3: the embed prefix is applied only to the vectors, not to the BM25 index. BM25 indexes are built per ticker, so the ticker and fiscal year in the prefix are identical in every document and carry no signal; only the Item would differ. The spec is updated accordingly in this task's commit.

- [ ] **Step 1: Write the failing store tests**

Append to `tests/test_vectorstore.py`:
```python
def test_chunks_for_returns_stored_chunks_sorted(tmp_path):
    store = FilingStore(tmp_path, DeterministicFakeEmbedding(size=32))
    store.upsert([mk("AAPL", 1, "Services"), mk("AAPL", 0, "iPhone sales"), mk("MSFT", 0, "Azure growth")])
    got = store.chunks_for("AAPL")
    assert [c.chunk_id for c in got] == ["AAPL-2025-7-000", "AAPL-2025-7-001"]
    assert got[0].text == "iPhone sales" and got[0].ticker == "AAPL" and got[0].fiscal_year == 2025 and got[0].item == "7"


def test_reset_empties_the_collection_and_it_stays_usable(tmp_path):
    store = FilingStore(tmp_path, DeterministicFakeEmbedding(size=32))
    chunks = [mk("AAPL", 0, "iPhone sales"), mk("MSFT", 0, "Azure growth")]
    store.upsert(chunks)
    store.reset()
    assert store.count() == 0
    store.upsert(chunks)
    assert store.count() == 2


def test_embed_prefix_changes_the_vector_but_not_the_stored_text(tmp_path):
    store = FilingStore(tmp_path, DeterministicFakeEmbedding(size=32), embed_prefix=True)
    store.upsert([mk("AAPL", 0, "iPhone sales"), mk("AAPL", 1, "Services")])
    # DeterministicFakeEmbedding is a hash of the text: the exact prefixed text matches with distance ~0.
    hits = store.search("AAPL FY2025 Item 7\niPhone sales", "AAPL", k=2)
    assert hits[0].chunk_id == "AAPL-2025-7-000"
    assert hits[0].score is not None and hits[0].score < 1e-6
    assert hits[0].text == "iPhone sales"
```

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run pytest tests/test_vectorstore.py -q`
Expected: 3 new failures (`chunks_for`, `reset`, unexpected keyword `embed_prefix`).

- [ ] **Step 3: Implement the store changes**

Replace the class header and `upsert` in `src/finassist/vectorstore.py` (keep `search` and `count` as they are) so the class reads:
```python
class FilingStore:
    def __init__(self, persist_dir: Path, embeddings: Embeddings, collection: str = "filings",
                 embed_prefix: bool = False):
        self._persist_dir = persist_dir
        self._embeddings = embeddings
        self._collection_name = collection
        self._embed_prefix = embed_prefix
        self._db = self._open()

    def _open(self) -> Chroma:
        return Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
        )

    def reset(self) -> None:
        """Drop every stored chunk. Needed when the chunking changes, because chunk ids change with it."""
        self._db.delete_collection()
        self._db = self._open()

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {"ticker": c.ticker, "fiscal_year": c.fiscal_year, "item": c.item, "source_url": c.source_url}
            for c in chunks
        ]
        if self._embed_prefix:
            # Embed "AAPL FY2025 Item 8\n<text>" but store (and later show the LLM) only <text>.
            vectors = self._embeddings.embed_documents([f"{c.context_prefix}\n{c.text}" for c in chunks])
            self._db._collection.upsert(ids=ids, embeddings=vectors,
                                        documents=[c.text for c in chunks], metadatas=metadatas)
        else:
            self._db.add_texts(texts=[c.text for c in chunks], metadatas=metadatas, ids=ids)

    def chunks_for(self, ticker: str) -> list[Chunk]:
        res = self._db.get(where={"ticker": ticker}, include=["documents", "metadatas"])
        chunks = [
            Chunk(chunk_id=i, text=d, ticker=m["ticker"], fiscal_year=int(m["fiscal_year"]),
                  item=m["item"], source_url=m.get("source_url", ""))
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]
        return sorted(chunks, key=lambda c: c.chunk_id)
```
If `delete_collection()` followed by reopening the same persist directory raises in the installed chromadb, recreate through `self._db._client.delete_collection(name)` and `self._db = self._open()`, keep the test as the contract, and say so in the report.

- [ ] **Step 4: Verify the store tests pass**

Run: `uv run pytest tests/test_vectorstore.py -q`
Expected: 4 passed.

- [ ] **Step 5: Write the failing retriever tests**

Create `tests/test_retrieval.py`:
```python
from dataclasses import replace

import pytest

from finassist.config import get_settings
from finassist.retrieval import Retriever, make_retriever, make_scorer, rrf, tokenize
from finassist.schemas import Chunk


def ck(i, text, ticker="AAPL"):
    return Chunk(chunk_id=f"{ticker}-2025-8-{i:03d}", text=text, ticker=ticker, fiscal_year=2025,
                 item="8", score=float(i))


class FakeStore:
    """Dense order is insertion order. chunks_for returns every chunk of the ticker."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.search_calls = []

    def search(self, query, ticker, k=5):
        self.search_calls.append((ticker, k))
        return [c for c in self.chunks if c.ticker == ticker][:k]

    def chunks_for(self, ticker):
        return [c for c in self.chunks if c.ticker == ticker]


def test_tokenize_keeps_numbers_with_separators_whole():
    assert tokenize("Net income: $133,749 and EPS 4.90.") == ["net", "income", "133,749", "and", "eps", "4.90"]


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([["a", "b", "c"], ["c", "a"]]) == ["a", "c", "b"]


def test_make_scorer_none_values():
    assert make_scorer("none") is None
    assert make_scorer("") is None


def test_dense_mode_without_scorer_matches_store_search():
    store = FakeStore([ck(0, "a"), ck(1, "b"), ck(2, "c"), ck(0, "m", "MSFT"), ck(1, "n", "MSFT")])
    r = Retriever(store, mode="dense", top_k=2)
    got = r.retrieve("q", ["AAPL", "MSFT"])
    assert [c.chunk_id for c in got] == ["AAPL-2025-8-000", "AAPL-2025-8-001", "MSFT-2025-8-000", "MSFT-2025-8-001"]
    assert store.search_calls == [("AAPL", 2), ("MSFT", 2)]


def test_reranker_reorders_a_wider_candidate_pool():
    store = FakeStore([ck(0, "iPhone"), ck(1, "Services"), ck(2, "Mac"), ck(3, "Net income 133,749"), ck(4, "Other")])
    scorer = lambda q, passages: [10.0 if "income" in p else 0.0 for p in passages]
    r = Retriever(store, mode="dense", scorer=scorer, candidates=5, top_k=2)
    got = r.retrieve("net income", ["AAPL"])
    assert got[0].chunk_id == "AAPL-2025-8-003"
    assert len(got) == 2
    assert store.search_calls == [("AAPL", 5)]


def test_hybrid_surfaces_a_chunk_the_dense_top_n_missed():
    chunks = [ck(0, "iPhone sales"), ck(1, "Services revenue"), ck(2, "Mac and iPad"), ck(9, "Net income 133,749 for the year")]
    store = FakeStore(chunks)
    r = Retriever(store, mode="hybrid", candidates=3, top_k=2)
    got = r.retrieve("net income 133,749", ["AAPL"])
    ids = [c.chunk_id for c in got]
    assert "AAPL-2025-8-009" in ids           # dense top-3 never contained it; BM25 rank 1 did
    assert store.search_calls == [("AAPL", 3)]


def test_hybrid_result_keeps_dense_scores_and_none_for_bm25_only_chunks():
    store = FakeStore([ck(0, "iPhone sales"), ck(1, "Services revenue"), ck(9, "Net income 133,749")])
    r = Retriever(store, mode="hybrid", candidates=2, top_k=3)
    by_id = {c.chunk_id: c for c in r.retrieve("net income 133,749", ["AAPL"])}
    assert by_id["AAPL-2025-8-000"].score == 0.0
    assert by_id["AAPL-2025-8-009"].score == 9.0   # FakeStore chunks carry their own score


def test_make_retriever_rejects_unknown_mode():
    with pytest.raises(ValueError):
        make_retriever(replace(get_settings(), retrieval_mode="sparse"), FakeStore([]))
```
- [ ] **Step 6: Run, expect FAIL**

Run: `uv run pytest tests/test_retrieval.py -q`
Expected: collection error `ModuleNotFoundError: finassist.retrieval`.

- [ ] **Step 7: Implement `src/finassist/retrieval.py`**

```python
from __future__ import annotations

import re
from typing import Callable, Protocol

from finassist.config import Settings
from finassist.schemas import Chunk

Scorer = Callable[[str, list[str]], list[float]]

# A number with thousands separators or decimals stays one token ("133,749", "4.90"); words are lowercase runs.
_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?|[a-z]+")


def tokenize(text: str) -> list[str]:
    return [t.rstrip(",") for t in _TOKEN_RE.findall(text.lower())]


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank). Ties break on id for determinism."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


def make_scorer(model_name: str) -> Scorer | None:
    if model_name in ("", "none"):
        return None
    from sentence_transformers import CrossEncoder  # heavy import, only when a reranker is requested

    model = CrossEncoder(model_name)

    def score(query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in model.predict([(query, p) for p in passages])]

    return score


class SearchStore(Protocol):
    def search(self, query: str, ticker: str, k: int = 5) -> list[Chunk]: ...
    def chunks_for(self, ticker: str) -> list[Chunk]: ...


class Retriever:
    def __init__(self, store: SearchStore, *, mode: str = "dense", scorer: Scorer | None = None,
                 candidates: int = 30, top_k: int = 5):
        self.store = store
        self.mode = mode
        self.scorer = scorer
        self.candidates = candidates
        self.top_k = top_k
        self._bm25: dict[str, tuple[object | None, list[Chunk]]] = {}

    def retrieve(self, query: str, tickers: list[str]) -> list[Chunk]:
        return [c for t in tickers for c in self._for_ticker(query, t)]

    def _for_ticker(self, query: str, ticker: str) -> list[Chunk]:
        if self.mode == "dense" and self.scorer is None:
            return self.store.search(query, ticker, self.top_k)
        dense = self.store.search(query, ticker, self.candidates)
        pool = {c.chunk_id: c for c in dense}
        order = [c.chunk_id for c in dense]
        if self.mode == "hybrid":
            bm25 = self._bm25_top(query, ticker, self.candidates)
            for c in bm25:
                pool.setdefault(c.chunk_id, c)
            order = rrf([order, [c.chunk_id for c in bm25]])
        candidates = [pool[cid] for cid in order]
        if self.scorer is not None and candidates:
            scores = self.scorer(query, [c.text for c in candidates])
            ranked = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
            candidates = [candidates[i] for i in ranked]
        return candidates[: self.top_k]

    def _bm25_top(self, query: str, ticker: str, n: int) -> list[Chunk]:
        if ticker not in self._bm25:
            from rank_bm25 import BM25Okapi

            chunks = self.store.chunks_for(ticker)
            docs = [tokenize(c.text) for c in chunks]
            self._bm25[ticker] = (BM25Okapi(docs) if docs else None, chunks)
        index, chunks = self._bm25[ticker]
        if index is None:
            return []
        scores = index.get_scores(tokenize(query))
        order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
        return [chunks[i] for i in order[:n]]


def make_retriever(settings: Settings, store: SearchStore) -> Retriever:
    if settings.retrieval_mode not in ("dense", "hybrid"):
        raise ValueError(f"FINASSIST_RETRIEVAL must be 'dense' or 'hybrid', got {settings.retrieval_mode!r}")
    return Retriever(store, mode=settings.retrieval_mode, scorer=make_scorer(settings.reranker),
                     candidates=settings.candidates, top_k=settings.top_k)
```

- [ ] **Step 8: Verify and commit**

Run: `uv run pytest tests/test_retrieval.py tests/test_vectorstore.py -q` then `uv run pytest -q`
Expected: all new tests pass and the full suite is green (145 existing tests plus the new ones) with no warnings.
If `test_hybrid_surfaces_a_chunk_the_dense_top_n_missed` fails because BM25 scores the target at zero (tiny-corpus IDF edge), keep the intent, adjust the fake chunks (add more filler chunks so the target term is rare), and say so in the report.
```bash
git add src/finassist/vectorstore.py src/finassist/retrieval.py tests/test_vectorstore.py tests/test_retrieval.py
git commit -m "feat: Retriever with BM25 fusion and cross-encoder rerank; store reset, chunks_for, embed prefix" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Wire the `Retriever` into `make_deps`

**Files:**
- Modify: `src/finassist/chains.py`
- Test: `tests/test_chains.py`

**Interfaces:**
- Consumes `make_retriever`, `Retriever` (Task 2).
- Produces `make_deps(settings, store, llm, retriever=None) -> Deps`. When `retriever` is `None` it is built with `make_retriever(settings, store)`. `Deps.retrieve(question, tickers)` delegates to `retriever.retrieve(question, tickers)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chains.py`:
```python
def test_make_deps_retrieve_delegates_to_the_retriever():
    from finassist.chains import make_deps
    from finassist.config import get_settings
    from finassist.schemas import Chunk

    class FakeLLM:
        def with_structured_output(self, *a, **k):
            return object()

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, tickers):
            self.calls.append((query, tickers))
            return [Chunk(chunk_id="AAPL-2025-8-001", text="t", ticker="AAPL", fiscal_year=2025, item="8")]

    r = FakeRetriever()
    deps = make_deps(get_settings(), store=object(), llm=FakeLLM(), retriever=r)
    got = deps.retrieve("english query", ["AAPL"])
    assert [c.chunk_id for c in got] == ["AAPL-2025-8-001"]
    assert r.calls == [("english query", ["AAPL"])]
```

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run pytest tests/test_chains.py -q`
Expected: `TypeError: make_deps() got an unexpected keyword argument 'retriever'`.

- [ ] **Step 3: Implement**

In `src/finassist/chains.py` add `from finassist.retrieval import make_retriever`, change the signature to `def make_deps(settings: Settings, store: FilingStore, llm: ChatAnthropic, retriever=None) -> Deps:`, add `retriever = retriever or make_retriever(settings, store)` as the first statement, and replace the body of `retrieve` with:
```python
    def retrieve(question: str, tickers: list[str]) -> list[Chunk]:
        return retriever.retrieve(question, tickers)
```

- [ ] **Step 4: Verify the default path is unchanged and commit**

Run: `uv run pytest -q`
Expected: green. With the defaults (`dense`, no reranker) `Retriever` returns exactly `store.search(query, ticker, top_k)` per ticker (covered by `test_dense_mode_without_scorer_matches_store_search`).
```bash
git add src/finassist/chains.py tests/test_chains.py
git commit -m "feat: make_deps retrieves through the Retriever" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Evidence groups in the golden set

**Files:**
- Modify: `eval/golden_set.json`
- Test: `tests/test_golden_evidence.py`

**Interfaces:**
- Produces on each of the 25 non-trap entries a field `"evidence": [{"ticker": "MSFT", "contains": ["133,749", "Net income"]}, ...]`. Trap entries (q26 to q30) get no `evidence` field.

- [ ] **Step 1: Write the failing test**

Create `tests/test_golden_evidence.py`:
```python
import json

import pytest

from finassist.config import ROOT, TICKERS, get_settings

GOLD = json.loads((ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))


def test_evidence_shape():
    for q in GOLD:
        ev = q.get("evidence")
        if q["category"] == "trap":
            assert not ev, q["id"]
            continue
        assert ev, q["id"]
        for g in ev:
            assert set(g) == {"ticker", "contains"}, q["id"]
            assert g["ticker"] in TICKERS, q["id"]
            assert g["contains"] and all(isinstance(s, str) and s.strip() for s in g["contains"]), q["id"]


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "AAPL.html").exists(), reason="needs data/raw")
def test_every_evidence_group_is_satisfiable_by_some_chunk():
    from finassist.ingest import load_filing

    s = get_settings()
    corpus = {t: [" ".join(c.text.lower().split()) for c in load_filing(t, s.raw_dir)] for t in TICKERS}
    for q in GOLD:
        for g in q.get("evidence", []):
            needles = [" ".join(n.lower().split()) for n in g["contains"]]
            assert any(all(n in text for n in needles) for text in corpus[g["ticker"]]), (q["id"], g)
```

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run pytest tests/test_golden_evidence.py -q`
Expected: `test_evidence_shape` fails (`AssertionError: q01`).

- [ ] **Step 3: Author the evidence by hand from the filings**

Never generate evidence with an LLM and never from memory. For each question read the chunks named in its `ground_truth` (`uv run python scripts/find_in_filing.py <TICKER> "<term>" [--item N]`), then choose short, distinctive strings that really occur in one chunk. Keep strings short (a figure plus its row label, or a 2 to 6 word phrase). Add the field with a small Python script that rewrites `eval/golden_set.json` (UTF-8, `ensure_ascii=False`, `indent=2`).

Numeric questions (figures come from the verified ground truth):

| id | evidence groups |
|---|---|
| q01 | AAPL: `["416,161", "Total net sales"]` |
| q02 | AAPL: `["7.46", "Diluted"]` |
| q03 | AAPL: `["209,586", "iPhone"]` |
| q04 | MSFT: `["331,839", "Total revenue"]` |
| q05 | MSFT: `["133,749", "Net income"]` |
| q06 | MSFT: `["17.95", "Diluted"]` |
| q07 | NVDA: `["215,938", "Revenue"]` |
| q08 | NVDA: `["120,067", "Net income"]` |
| q09 | NVDA: `["4.90", "diluted"]` |
| q10 | NVDA: `["193,737", "Data Center"]` |

Risk questions q11 to q20: for each cited chunk id in the ground truth that carries a distinct fact, add one group `{"ticker": <T>, "contains": [<phrase>]}` where `<phrase>` is 2 to 6 words copied from that chunk (for example for q17 a phrase about export controls on GPUs). Use at most 3 groups per question and prefer the chunks the ground truth names first.

Combined questions q21 to q25: take only the filing part. q21 AAPL `["416,161", "Total net sales"]`; q23 NVDA `["215,938", "Revenue"]`; q24 AAPL `["416,161", "Total net sales"]` and MSFT `["331,839", "Total revenue"]`; q22 and q25 use phrase groups from their cited competition and export-control chunks.

If a string does not appear in any chunk or differs in case or spacing, fix the string, not the test.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest tests/test_golden_evidence.py tests/test_golden_set.py -q`
Expected: all pass (the satisfiability test runs because `data/raw` exists locally).
```bash
git add eval/golden_set.json tests/test_golden_evidence.py
git commit -m "feat: hand-written evidence groups for retrieval benchmarking" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Retrieval benchmark and the stage 0 baseline

**Files:**
- Create: `eval/retrieval_eval.py`, `reports/retrieval_experiments.md`
- Test: `tests/test_retrieval_eval.py`

**Interfaces:**
- Consumes `Retriever`, `make_retriever` (Task 2), evidence groups (Task 4).
- Produces: `first_rank(texts: list[str], needles: list[str]) -> int | None` (1-based), `question_metrics(ranks: list[int | None], ks=KS) -> dict` with keys `any@K`, `all@K`, `rr`, `summarize(rows, ks=KS) -> dict` (`overall`, `by_category`), `evaluate(golden, retrieve) -> (rows, times_ms)` where `retrieve(query: str, ticker: str) -> list[Chunk]` returns the ranked list, `render_markdown(label, config, summary, times_ms) -> str`, and the CLI `uv run python -m eval.retrieval_eval --label L [--mode M] [--reranker R] [--candidates N] [--show-misses] [--append PATH]`. `KS = (5, 10, 20)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrieval_eval.py`:
```python
from eval.retrieval_eval import evaluate, first_rank, question_metrics, render_markdown, summarize
from finassist.schemas import Chunk


def ch(text):
    return Chunk(chunk_id="X-2025-8-000", text=text, ticker="AAPL", fiscal_year=2025, item="8")


def test_first_rank_needs_every_string_in_the_same_chunk_case_insensitive():
    texts = ["Net  income\nis shown", "133,749 only", "NET INCOME 133,749 total"]
    assert first_rank(texts, ["net income", "133,749"]) == 3
    assert first_rank(texts, ["missing"]) is None


def test_question_metrics_any_all_and_rr():
    m = question_metrics([2, 12], ks=(5, 10, 20))
    assert (m["any@5"], m["all@5"]) == (True, False)
    assert (m["any@10"], m["all@10"]) == (True, False)
    assert (m["any@20"], m["all@20"]) == (True, True)
    assert m["rr"] == 0.5
    miss = question_metrics([None], ks=(5,))
    assert miss["any@5"] is False and miss["rr"] == 0.0


def test_summarize_overall_and_by_category():
    rows = [
        {"id": "a", "category": "numeric", "metrics": question_metrics([1], ks=(5,))},
        {"id": "b", "category": "numeric", "metrics": question_metrics([None], ks=(5,))},
        {"id": "c", "category": "risk", "metrics": question_metrics([3], ks=(5,))},
    ]
    s = summarize(rows, ks=(5,))
    assert s["overall"]["n"] == 3
    assert round(s["overall"]["any@5"], 3) == 0.667
    assert s["by_category"]["numeric"]["any@5"] == 0.5
    assert s["by_category"]["risk"]["any@5"] == 1.0


def test_evaluate_skips_traps_caches_queries_and_times_each_retrieval():
    golden = [
        {"id": "q1", "category": "numeric", "question": "net income?",
         "evidence": [{"ticker": "MSFT", "contains": ["133,749"]}, {"ticker": "MSFT", "contains": ["net income"]}]},
        {"id": "q2", "category": "trap", "question": "should I buy?"},
    ]
    calls = []

    def retrieve(query, ticker):
        calls.append((query, ticker))
        return [ch("irrelevant"), ch("Net income 133,749")]

    rows, times = evaluate(golden, retrieve)
    assert calls == [("net income?", "MSFT")]          # one retrieval for the two groups, trap skipped
    assert len(times) == 1
    assert rows[0]["id"] == "q1" and rows[0]["ranks"] == [2, 2]


def test_render_markdown_contains_label_numbers_and_latency():
    rows = [{"id": "a", "category": "numeric", "metrics": question_metrics([1], ks=(5, 10, 20))}]
    md = render_markdown("stage 0", {"mode": "dense", "reranker": "none"}, summarize(rows), [10.0, 20.0, 30.0])
    for needle in ("stage 0", "dense", "any@5", "100%", "p50"):
        assert needle in md
```

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run pytest tests/test_retrieval_eval.py -q`
Expected: `ModuleNotFoundError: eval.retrieval_eval`.

- [ ] **Step 3: Implement `eval/retrieval_eval.py`**

```python
"""Retrieval-only benchmark: for each golden question, does a retrieved chunk contain the evidence?

Usage: uv run python -m eval.retrieval_eval --label "stage 0: dense" [--mode hybrid] [--reranker NAME]
       [--candidates 30] [--show-misses] [--append reports/retrieval_experiments.md]
The store is opened as it is on disk (set FINASSIST_CHROMA_DIR to benchmark another store).
No LLM is called. Runs in seconds once the embedding (and reranker) models are loaded.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from finassist.config import ROOT, TICKERS, get_settings
from finassist.retrieval import make_retriever
from finassist.schemas import Chunk
from finassist.vectorstore import FilingStore, get_embeddings

KS = (5, 10, 20)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def first_rank(texts: list[str], needles: list[str]) -> int | None:
    want = [_norm(n) for n in needles]
    for i, text in enumerate(texts, 1):
        t = _norm(text)
        if all(w in t for w in want):
            return i
    return None


def question_metrics(ranks: list[int | None], ks: tuple[int, ...] = KS) -> dict:
    out: dict = {}
    for k in ks:
        hits = [r is not None and r <= k for r in ranks]
        out[f"any@{k}"] = any(hits)
        out[f"all@{k}"] = all(hits)
    found = [r for r in ranks if r is not None]
    out["rr"] = 1.0 / min(found) if found else 0.0
    return out


def summarize(rows: list[dict], ks: tuple[int, ...] = KS) -> dict:
    def agg(sub: list[dict]) -> dict:
        n = len(sub)
        d: dict = {"n": n}
        for k in ks:
            d[f"any@{k}"] = sum(r["metrics"][f"any@{k}"] for r in sub) / n
            d[f"all@{k}"] = sum(r["metrics"][f"all@{k}"] for r in sub) / n
        d["mrr"] = sum(r["metrics"]["rr"] for r in sub) / n
        return d

    cats = sorted({r["category"] for r in rows})
    return {"overall": agg(rows), "by_category": {c: agg([r for r in rows if r["category"] == c]) for c in cats}}


def evaluate(golden: list[dict], retrieve) -> tuple[list[dict], list[float]]:
    """retrieve(query, ticker) -> ranked list[Chunk]. Returns per-question rows and latencies (ms)."""
    cache: dict[tuple[str, str], list[str]] = {}
    rows: list[dict] = []
    times: list[float] = []
    for q in golden:
        if q["category"] == "trap":
            continue
        ranks = []
        for g in q["evidence"]:
            key = (q["question"], g["ticker"])
            if key not in cache:
                t0 = time.perf_counter()
                chunks = retrieve(q["question"], g["ticker"])
                times.append((time.perf_counter() - t0) * 1000)
                cache[key] = [c.text for c in chunks]
            ranks.append(first_rank(cache[key], g["contains"]))
        rows.append({"id": q["id"], "category": q["category"], "ranks": ranks, "metrics": question_metrics(ranks)})
    return rows, times


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def render_markdown(label: str, config: dict, summary: dict, times_ms: list[float]) -> str:
    cols = [f"any@{k}" for k in KS] + [f"all@{k}" for k in KS] + ["mrr"]
    cfg = ", ".join(f"{k}={v}" for k, v in config.items())
    lines = [f"## {label}", "", f"Cấu hình: {cfg}", "",
             "| nhóm | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name, d in [("overall", summary["overall"]), *summary["by_category"].items()]:
        vals = [f"{d[c]:.3f}" if c == "mrr" else _pct(d[c]) for c in cols]
        lines.append(f"| {name} | {d['n']} | " + " | ".join(vals) + " |")
    lines += ["", f"Độ trễ mỗi lần truy hồi (ms): p50 {_quantile(times_ms, 0.5):.0f}, p95 {_quantile(times_ms, 0.95):.0f}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--mode")
    p.add_argument("--reranker")
    p.add_argument("--candidates", type=int)
    p.add_argument("--show-misses", action="store_true")
    p.add_argument("--append")
    a = p.parse_args(argv)

    s = get_settings()
    s = replace(
        s,
        retrieval_mode=a.mode or s.retrieval_mode,
        reranker=a.reranker or s.reranker,
        candidates=a.candidates or s.candidates,
        top_k=max(KS),
    )
    store = FilingStore(s.chroma_dir, get_embeddings(s))
    retriever = make_retriever(s, store)
    for t in TICKERS:  # warm-up: load models and build BM25 indexes before timing
        retriever.retrieve("warm up", [t])
    golden = json.loads((ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))
    rows, times = evaluate(golden, lambda q, t: retriever.retrieve(q, [t]))
    config = {"mode": s.retrieval_mode, "reranker": s.reranker, "candidates": s.candidates,
              "chroma_dir": s.chroma_dir.name}
    md = render_markdown(a.label, config, summarize(rows), times)
    print(md)
    if a.show_misses:
        for r in rows:
            if not r["metrics"]["any@5"]:
                print(f"miss@5 {r['id']} ({r['category']}): ranks {r['ranks']}")
    if a.append:
        path = Path(a.append)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify the unit tests**

Run: `uv run pytest tests/test_retrieval_eval.py -q` then `uv run pytest -q`
Expected: pass; the suite stays green.

- [ ] **Step 5: Run the stage 0 baseline on the real store**

First create `reports/retrieval_experiments.md` with this header (Vietnamese, plain):
```markdown
# Thí nghiệm truy hồi

Mỗi mục dưới đây là một lần chạy `eval.retrieval_eval` trên 25 câu có `evidence` trong `eval/golden_set.json`. Không có lệnh gọi LLM. any@K là tỷ lệ câu có ít nhất một nhóm bằng chứng nằm trong top-K của đúng ticker; all@K là tỷ lệ câu có mọi nhóm bằng chứng nằm trong top-K.

Quy tắc giữ hay bỏ một tầng nằm trong `docs/superpowers/specs/2026-10-29-retrieval-improvement-design.md` mục 3.4.

```
Run: `uv run python -m eval.retrieval_eval --label "giai đoạn 0: vector thuần (baseline)" --show-misses --append reports/retrieval_experiments.md`
Expected: a table with `overall`, `numeric`, `risk`, `combined` rows, latency p50/p95, and the list of questions that miss at top-5. The numbers should be close to the earlier rank measurement (any@5 near 24%, all@5 near 8%) but are not identical, because the evidence rules differ from the chunk-id rule.

- [ ] **Step 6: Confirm the targets and commit**

Read the baseline row. If `any@5` is already above 50%, or the baseline is far from the spec's 24% and 8% for a reason you can name, edit `docs/superpowers/specs/2026-10-29-retrieval-improvement-design.md` section 6 to state the measured baseline and the confirmed targets (keep the 3 second latency budget unless p50 baseline suggests otherwise). Add one sentence under the baseline section in `reports/retrieval_experiments.md` saying which targets were confirmed.
```bash
git add eval/retrieval_eval.py tests/test_retrieval_eval.py reports/retrieval_experiments.md docs/superpowers/specs/2026-10-29-retrieval-improvement-design.md
git commit -m "feat: retrieval benchmark and stage 0 baseline" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Stage 1 experiment, the reranker

**Files:**
- Modify: `reports/retrieval_experiments.md`

**Interfaces:**
- Consumes the benchmark CLI (Task 5) and `make_scorer` (Task 2). No code change unless a model fails to load.

- [ ] **Step 1: Run both reranker candidates on the unchanged vector store**

Run (the first call downloads each model once; expect a few minutes):
```bash
uv run python -m eval.retrieval_eval --label "giai đoạn 1a: vector top-30 + MiniLM" --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --show-misses --append reports/retrieval_experiments.md
uv run python -m eval.retrieval_eval --label "giai đoạn 1b: vector top-30 + bge-reranker-base" --reranker BAAI/bge-reranker-base --show-misses --append reports/retrieval_experiments.md
```
Expected: two more sections in `reports/retrieval_experiments.md`, each with latency p50 and p95. If a model name cannot be loaded, report the exact error, try the nearest valid name on the Hugging Face hub (for example `cross-encoder/ms-marco-MiniLM-L-12-v2`), and record what was actually used.

- [ ] **Step 2: Apply the gate and write the decision**

Gate (spec 3.4): keep a reranker when overall `any@5` rises by at least 15 percentage points over stage 0. Choose the better of 1a and 1b by `any@5`, breaking ties by lower p50 latency. Also compare p50 latency against the 3 second budget. Append to `reports/retrieval_experiments.md`:
```markdown
### Quyết định giai đoạn 1

<kept or dropped>: <which model>, any@5 từ <X>% lên <Y>%, all@5 từ <X>% lên <Y>%, p50 <N> ms. <One or two sentences on which questions improved or still miss, from the --show-misses output.>
```
Write the filled version (real numbers, no angle brackets). If the gate fails, state that the reranker is dropped and why, and continue the ladder with `none` as the reranker.

- [ ] **Step 3: Commit**

```bash
git add reports/retrieval_experiments.md
git commit -m "docs: stage 1 reranker experiment results" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Stage 2 experiment, BM25 hybrid

**Files:**
- Modify: `reports/retrieval_experiments.md`

**Interfaces:**
- Consumes the benchmark CLI and the stage 1 decision (best reranker or `none`).

- [ ] **Step 1: Run hybrid with and without the stage 1 reranker**

```bash
uv run python -m eval.retrieval_eval --label "giai đoạn 2a: hybrid BM25 + vector, không reranker" --mode hybrid --show-misses --append reports/retrieval_experiments.md
uv run python -m eval.retrieval_eval --label "giai đoạn 2b: hybrid + reranker đã chọn" --mode hybrid --reranker <chosen model from stage 1> --show-misses --append reports/retrieval_experiments.md
```
Use the real model name from the stage 1 decision; if stage 1 dropped the reranker, run only 2a.

- [ ] **Step 2: Apply the gate and write the decision**

Gate (spec 3.4): keep hybrid when overall `any@5` or `all@5` rises by at least 5 percentage points over the best stage 1 result (over stage 0 if stage 1 was dropped). Compare against the better of 2a and 2b. Append a `### Quyết định giai đoạn 2` section in the same form as in Task 6, with real numbers.

- [ ] **Step 3: Commit**

```bash
git add reports/retrieval_experiments.md
git commit -m "docs: stage 2 hybrid experiment results" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Stage 3, table rows and embed prefix

**Files:**
- Modify: `src/finassist/ingest.py`, `src/finassist/cli.py`, `tests/test_golden_evidence.py`
- Test: `tests/test_ingest.py`, `tests/test_cli_replay.py`
- Modify: `reports/retrieval_experiments.md`

**Interfaces:**
- Produces `html_to_text(html: str, table_rows: bool = False) -> str` and `load_filing(ticker, raw_dir, table_rows=False) -> list[Chunk]`. With `table_rows=True` every table row becomes one line with cells joined by `" | "`, lone `$` cells dropped, and orphan `)`, `%`, `%)` cells appended to the previous cell.
- Produces CLI `finassist ingest [--rebuild]`. `--rebuild` calls `FilingStore.reset()` first. Ingestion honours `Settings.table_rows` and `Settings.embed_prefix`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingest.py`:
```python
HTML_TABLE = (
    "<html><body><table>"
    "<tr><td>Net income</td><td>$</td><td>133,749</td><td>$</td><td>101,832</td></tr>"
    "<tr><td>Margin</td><td>45</td><td>%</td></tr>"
    "<tr><td>Loss</td><td>(1,234</td><td>)</td></tr>"
    "</table></body></html>"
)


def test_table_rows_become_single_lines():
    text = html_to_text(HTML_TABLE, table_rows=True)
    lines = [ln for ln in text.splitlines() if ln]
    assert lines == ["Net income | 133,749 | 101,832", "Margin | 45%", "Loss | (1,234)"]


def test_default_html_to_text_still_splits_cells_onto_lines():
    text = html_to_text(HTML_TABLE)
    assert "Net income | 133,749" not in text
    assert "Net income" in text.splitlines() and "133,749" in text.splitlines()
```
(`html_to_text` is already imported at the top of that file.) Append to `tests/test_cli_replay.py`:
```python
def test_ingest_accepts_rebuild_flag(monkeypatch):
    import finassist.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "_cmd_ingest", lambda rebuild=False: seen.update(rebuild=rebuild))
    monkeypatch.setattr("sys.argv", ["finassist", "ingest", "--rebuild"])
    cli.main()
    assert seen == {"rebuild": True}
    monkeypatch.setattr("sys.argv", ["finassist", "ingest"])
    cli.main()
    assert seen == {"rebuild": False}
```

- [ ] **Step 2: Run, expect FAIL**

Run: `uv run pytest tests/test_ingest.py tests/test_cli_replay.py -q`
Expected: the three new tests fail (`unexpected keyword argument 'table_rows'`, unrecognized `--rebuild`).

- [ ] **Step 3: Implement the ingest change**

In `src/finassist/ingest.py` change the import to `from bs4 import BeautifulSoup, NavigableString`, and add above `html_to_text`:
```python
def _table_rows_to_lines(soup: BeautifulSoup) -> None:
    """Replace each <tr> by one text line: cells joined with ' | '. SEC tables put '$', '%' and ')' in cells of their own."""
    for tr in soup.find_all("tr"):
        cells: list[str] = []
        for td in tr.find_all(["td", "th"]):
            text = re.sub(r"\s+", " ", td.get_text(" ", strip=True)).strip()
            if not text or text == "$":
                continue
            if text in {")", "%", "%)"} and cells:
                cells[-1] += text
            else:
                cells.append(text)
        tr.replace_with(NavigableString("\n" + " | ".join(cells) + "\n"))
```
Change `def html_to_text(html: str) -> str:` to `def html_to_text(html: str, table_rows: bool = False) -> str:` and, right after the `for tag in soup.find_all(["script", "style", "ix:header"]): tag.decompose()` loop, add `if table_rows: _table_rows_to_lines(soup)`. Change `load_filing` to `def load_filing(ticker: str, raw_dir: Path, table_rows: bool = False) -> list[Chunk]:` and pass `table_rows=table_rows` into its `html_to_text(...)` call.
In `src/finassist/cli.py` replace `_cmd_ingest` with:
```python
def _cmd_ingest(rebuild: bool = False) -> None:
    from finassist.config import TICKERS, get_settings
    from finassist.ingest import load_filing
    from finassist.vectorstore import FilingStore, get_embeddings

    s = get_settings()
    store = FilingStore(s.chroma_dir, get_embeddings(s), embed_prefix=s.embed_prefix)
    if rebuild:
        store.reset()
    for t in TICKERS:
        chunks = load_filing(t, s.raw_dir, table_rows=s.table_rows)
        store.upsert(chunks)
        print(f"{t}: {len(chunks)} chunks")
    print(f"total in store: {store.count()}")
```
and in `main()` change `sub.add_parser("ingest")` to `ing = sub.add_parser("ingest")` followed by `ing.add_argument("--rebuild", action="store_true")`, and the dispatch line to `_cmd_ingest(args.rebuild)`.
In `tests/test_golden_evidence.py` change the corpus line to `load_filing(t, s.raw_dir, table_rows=s.table_rows)`.

- [ ] **Step 4: Verify the code and commit it**

Run: `uv run pytest -q`
Expected: green.
```bash
git add src/finassist/ingest.py src/finassist/cli.py tests/test_ingest.py tests/test_cli_replay.py tests/test_golden_evidence.py
git commit -m "feat: opt-in table rows in html_to_text and ingest --rebuild" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Build a separate store and check the evidence is still satisfiable**

The baseline store must stay available for comparison, so the new store lives in its own directory:
```bash
export ALT=$(pwd)/data/chroma_tables
FINASSIST_CHROMA_DIR=$ALT FINASSIST_TABLE_ROWS=1 FINASSIST_EMBED_PREFIX=1 uv run finassist ingest --rebuild
FINASSIST_TABLE_ROWS=1 uv run pytest tests/test_golden_evidence.py -q
```
Expected: three `<TICKER>: N chunks` lines and a total; the evidence satisfiability test passes with table rows on. If a group is no longer satisfiable (for example the label sits in a different row than the figure), report it and adjust the string only if the original string was too tight.

- [ ] **Step 6: Run the benchmark on the new store with the best configuration so far**

Use the stage 1 and stage 2 winners (mode and reranker; `dense` and `none` if both were dropped):
```bash
FINASSIST_CHROMA_DIR=$ALT uv run python -m eval.retrieval_eval --label "giai đoạn 3: chunk bảng + tiền tố embed, cấu hình tốt nhất" --mode <winner mode> --reranker <winner reranker> --show-misses --append reports/retrieval_experiments.md
```
(Omit `--mode` or `--reranker` when the winner is the default; use the real values, not placeholders.) Also run the same command with `--mode dense` and no reranker to show the effect of the new chunks alone. The printed config shows the store directory (`chroma_tables`), which identifies the table-row store; state in each label that the store was built with table rows and the embed prefix.

- [ ] **Step 7: Apply the gate and write the decision**

Gate (spec 3.4): keep stage 3 when overall `any@5` rises by at least 5 percentage points over the best earlier result, or when it fixes at least one of q05, q04, q21, q24, q03 without breaking another question (compare the `--show-misses` lists). Append `### Quyết định giai đoạn 3` with real numbers, including whether the chunk count changed and whether q10 and q20 now retrieve a chunk where the figures carry their row labels.
If the gate fails, delete `data/chroma_tables` (`rm -rf data/chroma_tables`) and say so; the code stays because it is opt-in and tested.

- [ ] **Step 8: Commit**

```bash
git add reports/retrieval_experiments.md
git commit -m "docs: stage 3 table rows experiment results" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Defaults, full RAGAS rerun, documentation

**Files:**
- Modify: `src/finassist/config.py`, `eval/run_ragas.py`, `tests/test_config.py`, `README.md`, `docs/ARCHITECTURE.md`, `docs/NEXT_STEP.md`, `docs/DEMO_SCRIPT.md`, `docs/KNOWLEDGE_CHECK.md`, `docs/TEAM_LOG.md`, `runs/demo.json`, `reports/retrieval_experiments.md`
- Create: `tests/test_retrieval_integration.py`, `reports/ragas_baseline_dense.md`, `reports/ragas_baseline_dense.json`, `reports/ragas_retrieval_v2.md`, `reports/ragas_retrieval_v2.json`

**Interfaces:**
- Consumes the decisions of Tasks 6 to 8.
- Produces new defaults in `get_settings()` equal to the winning configuration, a `--tag` option on `eval.run_ragas`, and updated documents.

- [ ] **Step 1: Preserve the hand-annotated baseline before anything can overwrite it**

```bash
cp reports/ragas_baseline.md reports/ragas_baseline_dense.md
cp reports/ragas_baseline.json reports/ragas_baseline_dense.json
```

- [ ] **Step 2: Add `--tag` to the RAGAS runner**

In `eval/run_ragas.py` change `def main() -> None:` to `def main(tag: str = "baseline") -> None:`, replace each of the three occurrences of `"ragas_baseline.json"` / `"ragas_baseline.md"` with `f"ragas_{tag}.json"` / `f"ragas_{tag}.md"` (the JSON write, the Markdown write, and the final `print(... .read_text())`), and replace the bottom of the file with:
```python
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline", help="output files are reports/ragas_<tag>.{json,md}")
    main(ap.parse_args().tag)
```
Run `uv run python -c "import eval.run_ragas"` (must import without error) and `uv run pytest -q` (green).

- [ ] **Step 3: Set the defaults to the winning configuration**

Edit `get_settings()` so the defaults equal the experiment decisions from `reports/retrieval_experiments.md`:

| Decision | Default to set |
|---|---|
| Stage 1 kept model M | `FINASSIST_RERANKER` default `M`; otherwise `"none"` |
| Stage 2 kept | `FINASSIST_RETRIEVAL` default `"hybrid"`; otherwise `"dense"` |
| Stage 3 kept | `FINASSIST_EMBED_PREFIX` and `FINASSIST_TABLE_ROWS` defaults `"1"`; otherwise `"0"` |

Update `test_retrieval_settings_defaults` in `tests/test_config.py` to the new defaults. If stage 3 was kept, rebuild the standard store so the defaults and the data agree, then verify:
```bash
uv run finassist ingest --rebuild
uv run python -m eval.retrieval_eval --label "cấu hình mặc định cuối cùng" --append reports/retrieval_experiments.md
```
Expected: the default-configuration row matches the stage decision it came from (same numbers within rounding). Add a closing `## Kết luận` section to `reports/retrieval_experiments.md` that lists the final configuration, the final numbers against the spec targets, and the layers that were dropped with their numbers.

- [ ] **Step 4: Integration test for the default path**

Create `tests/test_retrieval_integration.py`:
```python
import json
import os

import pytest

from finassist.config import ROOT, get_settings

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
def test_default_retriever_finds_the_msft_net_income_evidence_in_top5():
    from finassist.retrieval import make_retriever
    from finassist.vectorstore import FilingStore, get_embeddings

    s = get_settings()
    retriever = make_retriever(s, FilingStore(s.chroma_dir, get_embeddings(s)))
    q = next(x for x in json.loads((ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8")) if x["id"] == "q05")
    chunks = retriever.retrieve(q["question"], ["MSFT"])
    needles = [n.lower() for n in q["evidence"][0]["contains"]]
    assert any(all(n in " ".join(c.text.lower().split()) for n in needles) for c in chunks)
```
Run: `RUN_INTEGRATION=1 uv run pytest tests/test_retrieval_integration.py -q`. If the final configuration does not retrieve q05's evidence in the top 5, report that honestly instead of weakening the test, and mark the test `xfail` with a reason that states the measured rank.

- [ ] **Step 5: Re-record the demo run if the chunk ids changed**

If stage 3 was kept, the citations in `runs/demo.json` refer to old chunk ids. Re-record it (needs the router):
```bash
uv run finassist ask "What were Apple's total net sales in its most recent fiscal year?" --record runs/demo.json
uv run finassist ask x --replay runs/demo.json
```
Expected: the replay prints the Apple FY2025 net sales (416,161) with citations whose chunk ids exist in the new store. If stage 3 was dropped, leave `runs/demo.json` as it is.

- [ ] **Step 6: Run the full RAGAS comparison**

Run (about 15 minutes; run it in the background and wait for completion):
```bash
uv run python -m eval.run_ragas --tag retrieval_v2
```
Expected: `reports/ragas_retrieval_v2.md` and `.json`. Replace each `TO ANALYZE` in the worst-case section with a short root cause based on the stored per-question data (which chunks were retrieved, whether the evidence was among them), as was done for the first baseline. Do not edit scores. Then append to `reports/retrieval_experiments.md` a table comparing `ragas_baseline_dense` and `ragas_retrieval_v2` (the four metrics, overall and per category) and state whether the spec 6 criteria are met: context precision and recall up in numeric and combined, faithfulness not down by more than 0.03, and q05, q04, q21, q24, q03 retrieving evidence. Report failures plainly.

- [ ] **Step 7: Update the documentation (Vietnamese, humanizer rules from Global Constraints)**

- `docs/ARCHITECTURE.md`: describe the final retrieval in sections 3.2 (`retrieve` row), 4.1 (ingestion diagram and chunk description if table rows were kept), 5 (a short retrieval subsection or sequence-diagram step: candidates, fusion, rerank), 7.6 (decisions table: add the rows for hybrid, reranker and table rows with the measured numbers, or the reason they were dropped), 8 (rewrite limitation 1 with the new numbers), and 9 (new environment variables). Keep all diagrams renderable: validate every mermaid block by rendering it in a browser before committing.
- `docs/NEXT_STEP.md`: mark the retrieval work done with the final numbers and link to `reports/retrieval_experiments.md`; move what remains (grounding semantics, backlog items) to the top as the next priority.
- `README.md`: add the new environment variables (`FINASSIST_RETRIEVAL`, `FINASSIST_RERANKER`, `FINASSIST_CANDIDATES`, `FINASSIST_EMBED_PREFIX`, `FINASSIST_TABLE_ROWS`, `FINASSIST_CHROMA_DIR`), the benchmark command, the `ingest --rebuild` note, and a first-run note that the reranker model downloads once.
- `docs/DEMO_SCRIPT.md` and `docs/KNOWLEDGE_CHECK.md`: update numbers, the retrieval explanation and every `file:line` reference that moved (open each reference and check it); keep the 'Debug' cases true to the new results (q05 may no longer be a miss; if so, replace it with a case that still misses, taken from `reports/ragas_retrieval_v2.json`).
- `docs/TEAM_LOG.md`: add the new commits.
- `reports/ragas_baseline.md` and `.json`: leave as they are, they describe the vector-only system. Add one line at the top of `reports/ragas_baseline.md` pointing to `reports/ragas_retrieval_v2.md` for the current system.

- [ ] **Step 8: Final verification**

Run each and confirm:
```bash
uv run pytest -q                                                    # all green
RUN_INTEGRATION=1 uv run pytest tests/test_llm_smoke.py tests/test_retrieval_integration.py -q
git ls-files | grep -E '^\.env$' || echo ".env not tracked"
git grep -nE 'sk-[A-Za-z0-9-]{12,}|lsv2_pt_' -- . ':!docs/DEMO_SCRIPT.md' ':!docs/superpowers' || echo "no keys"
uv run finassist ask x --replay runs/demo.json
```
Also check that every relative Markdown link in `README.md` and `docs/*.md` resolves, and that the edited docs contain no em or en dashes outside code.

- [ ] **Step 9: Commit**

```bash
git add src/finassist/config.py eval/run_ragas.py tests README.md docs reports runs
git commit -m "feat: final retrieval defaults, RAGAS comparison and documentation" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage (self-review)

| Spec section | Task |
|---|---|
| 3.1 benchmark with evidence groups, hit@k any/all, MRR, latency | 4, 5 |
| 3.2 `Retriever` (vector, BM25, RRF, cross-encoder, env switches) | 1, 2, 3 |
| 3.3 table rows, embed prefix, `ingest --rebuild` | 2 (prefix, reset), 8 |
| 3.4 experiment ladder with gates | 5 (stage 0), 6, 7, 8 |
| 4 changes in the repo | all tasks (file structure above) |
| 5 tests (fakes, benchmark maths, evidence data test, integration) | 2, 4, 5, 8, 9 |
| 6 success criteria, confirmed after the baseline | 5 (confirm), 9 (final check) |
| 7 risks (evidence quality, latency, model download, chunk id churn) | 4, 6, 8 (separate store), 9 (demo re-record, docs) |

Execution dependencies: Tasks 1 to 5 are code and tests and run without the router. Tasks 6 to 8 are experiments that download models and decide the defaults; they need only local embeddings. Task 9 needs the router (RAGAS and the demo re-record) and about 15 minutes of RAGAS time.
