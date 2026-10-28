# Finance AI Chat Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Finance chatbot (LangGraph + RAG over SEC 10-K + deterministic price tool + structured output) with a RAGAS baseline report, LangSmith tracing and a demo-ready doc package.

**Architecture:** A LangGraph `StateGraph` routes by intent (`price | filing | both | out_of_scope`). All side effects (LLM, vector store, price fetch) are injected through a `Deps` dataclass so graph logic is unit-testable with fakes. Citations and stock price in the final `FinanceAnswer` are attached deterministically (from `[chunk_id]` markers and the price tool), never invented by the LLM.

**Tech Stack:** Python 3.12 (uv), langchain, langchain-anthropic, langgraph, langchain-chroma + chromadb, langchain-huggingface (bge-small, local), yfinance, beautifulsoup4, ragas, langsmith, streamlit, pytest.

Spec: `docs/superpowers/specs/2026-10-28-finance-rag-assistant-design.md` (requirement ids R1–R12 are referenced below).

## Global Constraints

- Python 3.12 via uv; never use the system 3.14.
- LLM and judge: `kr/claude-sonnet-5.5` through the local router; `auto/*` models are broken (rate limit), do not use them.
- `ANTHROPIC_BASE_URL` in `.env` ends with `/v1` (curl style). The Anthropic SDK appends `/v1/messages` itself, so `config.normalize_base_url` must strip a trailing `/v1`.
- Embedding: `BAAI/bge-small-en-v1.5`, local, no extra API key.
- Tickers in scope: `AAPL`, `MSFT`, `NVDA` only.
- Chunking: ~1000 chars, overlap 150, split by Item (1, 1A, 7, 7A, 8) first. `chunk_id` format `{TICKER}-{FY}-{ITEM}-{NNN}`, e.g. `AAPL-2025-1A-012`.
- `.env` is never committed; only `.env.example`. Never print API keys.
- Commit messages end with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (pass as a second `-m`).
- Revenue values are USD **millions** exactly as printed in the filing.
- Tests must not call the network or the LLM, except tests marked `integration` (skipped unless `RUN_INTEGRATION=1`).

## File Structure

```
pyproject.toml
src/finassist/__init__.py        version
src/finassist/config.py          Settings, normalize_base_url, TICKERS, ROOT
src/finassist/schemas.py         pydantic models + Intent
src/finassist/ingest.py          html_to_text, split_items, chunk_filing, load_filing
src/finassist/vectorstore.py     get_embeddings, FilingStore
src/finassist/tools.py           PriceUnavailable, get_stock_price
src/finassist/postprocess.py     MARKER_RE, extract_citations, ground_numbers
src/finassist/graph.py           Deps, build_graph, fixed texts
src/finassist/chains.py          build_llm, make_deps (real LLM implementations)
src/finassist/runtime.py         build_app, ask
src/finassist/cli.py             finassist ingest | ask
app/streamlit_app.py
scripts/fetch_filings.py  scripts/find_in_filing.py  scripts/trace_link.py
eval/golden_set.json  eval/report.py  eval/run_ragas.py
tests/...  docs/...  reports/  runs/
```

---

### Task 1: Project scaffold, config, schemas, router smoke test

**Files:**
- Create: `pyproject.toml`, `src/finassist/__init__.py`, `src/finassist/config.py`, `src/finassist/schemas.py`
- Test: `tests/test_config.py`, `tests/test_schema.py`, `tests/test_llm_smoke.py`

**Interfaces:**
- Produces: `Settings` fields `base_url, api_key, llm_model, judge_model, embedding_model, chroma_dir, raw_dir, cache_dir, sec_user_agent, top_k, price_offline`; `get_settings() -> Settings`; `normalize_base_url(url) -> str`; `TICKERS`; `ROOT`.
- Produces schemas: `Intent = Literal["price","filing","both","out_of_scope"]`, `Chunk`, `Citation`, `Money`, `PriceInfo`, `FinanceAnswer`, `IntentResult`, `LLMExtraction`.

- [ ] **Step 1: Create pyproject and install**

`pyproject.toml`:
```toml
[project]
name = "finassist"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "langchain", "langchain-anthropic", "langchain-huggingface", "langchain-chroma",
  "langchain-text-splitters", "langgraph", "chromadb", "sentence-transformers",
  "pydantic>=2", "python-dotenv", "yfinance", "beautifulsoup4", "lxml", "requests",
  "ragas", "langsmith", "streamlit", "pandas",
]

[dependency-groups]
dev = ["pytest"]

[project.scripts]
finassist = "finassist.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/finassist"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["integration: needs router/network, run with RUN_INTEGRATION=1"]
```
Run: `cd /Users/phamtienthanh/Projects/LangChain && uv python pin 3.12 && uv sync`
Expected: resolves and installs (first run downloads torch, several minutes). Record the resolved ragas version: `uv pip list | grep -i ragas`.

- [ ] **Step 2: Write failing tests**

`tests/test_config.py`:
```python
from finassist.config import normalize_base_url, get_settings


def test_strips_trailing_v1():
    assert normalize_base_url("http://localhost:20128/v1") == "http://localhost:20128"
    assert normalize_base_url("http://localhost:20128/v1/") == "http://localhost:20128"
    assert normalize_base_url("http://localhost:20128") == "http://localhost:20128"


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("FINASSIST_LLM_MODEL", "kr/claude-sonnet-5.5")
    s = get_settings()
    assert s.base_url == "http://localhost:20128"
    assert s.llm_model == "kr/claude-sonnet-5.5"
    assert s.top_k == 5
```
`tests/test_schema.py`:
```python
import pytest
from pydantic import ValidationError
from finassist.schemas import FinanceAnswer, Money, Citation


def test_valid_answer_roundtrip():
    a = FinanceAnswer(
        intent="filing", ticker="AAPL", fiscal_year=2025,
        revenue=Money(value=416161.0, period="FY2025"), eps=7.46,
        risk_factors=["supply chain"], answer="ok",
        citations=[Citation(source="AAPL 10-K FY2025", item="7", chunk_id="AAPL-2025-7-001", quote="x")],
    )
    assert FinanceAnswer.model_validate_json(a.model_dump_json()) == a


def test_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        FinanceAnswer(intent="weather", answer="x")


def test_defaults_are_null_safe():
    a = FinanceAnswer(intent="out_of_scope", answer="no")
    assert a.revenue is None and a.risk_factors == [] and a.citations == []
```
`tests/test_llm_smoke.py`:
```python
import os
import pytest
from finassist.config import get_settings

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
def test_router_answers():
    from langchain_anthropic import ChatAnthropic
    s = get_settings()
    llm = ChatAnthropic(model=s.llm_model, base_url=s.base_url, api_key=s.api_key, max_tokens=20)
    assert "ok" in llm.invoke("Reply with: ok").text.lower()
```

- [ ] **Step 3: Run and verify failure**

Run: `uv run pytest tests/test_config.py tests/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: finassist.config`.

- [ ] **Step 4: Implement**

`src/finassist/__init__.py`: `__version__ = "0.1.0"`

`src/finassist/config.py`:
```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
TICKERS = ("AAPL", "MSFT", "NVDA")


def normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    llm_model: str
    judge_model: str
    embedding_model: str
    chroma_dir: Path
    raw_dir: Path
    cache_dir: Path
    sec_user_agent: str
    top_k: int
    price_offline: bool


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    data = ROOT / "data"
    return Settings(
        base_url=normalize_base_url(os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")),
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        llm_model=os.getenv("FINASSIST_LLM_MODEL", "kr/claude-sonnet-5.5"),
        judge_model=os.getenv("FINASSIST_JUDGE_MODEL", "kr/claude-sonnet-5.5"),
        embedding_model=os.getenv("FINASSIST_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        chroma_dir=data / "chroma",
        raw_dir=data / "raw",
        cache_dir=data / "cache",
        sec_user_agent=os.getenv("SEC_USER_AGENT", "finassist-demo demo@example.com"),
        top_k=int(os.getenv("FINASSIST_TOP_K", "5")),
        price_offline=os.getenv("FINASSIST_PRICE_OFFLINE", "0") == "1",
    )
```
`src/finassist/schemas.py`:
```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["price", "filing", "both", "out_of_scope"]


class Chunk(BaseModel):
    chunk_id: str
    text: str
    ticker: str
    fiscal_year: int
    item: str
    source_url: str = ""
    score: float | None = None


class Citation(BaseModel):
    source: str
    item: str
    chunk_id: str
    quote: str


class Money(BaseModel):
    value: float = Field(description="USD millions, exactly as printed in the filing")
    unit: str = "USD millions"
    period: str


class PriceInfo(BaseModel):
    ticker: str
    price: float
    market_cap: float | None = None
    pe_ratio: float | None = None
    as_of: str
    stale: bool = False


class FinanceAnswer(BaseModel):
    ticker: str | None = None
    fiscal_year: int | None = None
    revenue: Money | None = None
    eps: float | None = None
    risk_factors: list[str] = Field(default_factory=list)
    stock_price: PriceInfo | None = None
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    intent: Intent


class IntentResult(BaseModel):
    intent: Intent
    tickers: list[str] = Field(default_factory=list)


class LLMExtraction(BaseModel):
    """Fields the LLM may extract from retrieved chunks. Everything else is deterministic."""
    ticker: str | None = None
    fiscal_year: int | None = None
    revenue: Money | None = None
    eps: float | None = None
    risk_factors: list[str] = Field(default_factory=list)
```

- [ ] **Step 5: Verify**

Run: `uv run pytest tests/test_config.py tests/test_schema.py -v`
Expected: 5 passed.
Run (real router): `RUN_INTEGRATION=1 uv run pytest tests/test_llm_smoke.py -v`
Expected: 1 passed. If it fails with 404, the base URL still has `/v1`; fix `normalize_base_url`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version src tests
git commit -m "feat: scaffold project, config and schemas" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Filing parser and chunker (R3)

**Files:**
- Create: `src/finassist/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Chunk` from Task 1.
- Produces: `html_to_text(html: str) -> str`; `split_items(text: str) -> dict[str, str]` (keys among `"1","1A","7","7A","8"`); `chunk_filing(ticker: str, fiscal_year: int, text: str, source_url: str = "") -> list[Chunk]`.

- [ ] **Step 1: Write failing tests**

`tests/test_ingest.py`:
```python
from finassist.ingest import html_to_text, split_items, chunk_filing

TOC = "Table of Contents\nItem 1.\nBusiness\nItem 1A.\nRisk Factors\nItem 7.\nMD&A\nItem 8.\nFinancial Statements\n"
BODY = (
    "Item 1.  Business\n" + "We make devices. " * 40 + "\n"
    "Item 1A.  Risk Factors\n" + "Supply chain risk matters. " * 80 + "\n"
    "Item 1B.  Unresolved Staff Comments\nNone\n"
    "Item 7.  Management's Discussion\n" + "Revenue grew. " * 50 + "\n"
    "Item 8.  Financial Statements\n" + "Total net sales 416,161. " * 30 + "\n"
    "Item 10.  Directors\n" + "Board stuff. " * 20 + "\n"
)


def test_split_items_uses_body_not_toc():
    items = split_items(TOC + BODY)
    assert set(items) == {"1", "1A", "7", "8"}
    assert "Supply chain risk" in items["1A"]
    assert "Board stuff" not in items["8"]      # Item 10 must not be read as Item 1


def test_chunks_have_metadata_and_stable_ids():
    chunks = chunk_filing("AAPL", 2025, TOC + BODY, "http://x")
    assert chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    first_1a = next(c for c in chunks if c.item == "1A")
    assert first_1a.chunk_id == "AAPL-2025-1A-000"
    assert first_1a.ticker == "AAPL" and first_1a.fiscal_year == 2025
    assert all(len(c.text) <= 1100 for c in chunks)
    assert chunk_filing("AAPL", 2025, TOC + BODY, "http://x") == chunks   # deterministic


def test_html_to_text_drops_script_and_hidden_xbrl():
    html = "<html><body><ix:header>HIDDEN</ix:header><script>x=1</script><p>Item 1.</p><p>Hello&nbsp;world</p></body></html>"
    text = html_to_text(html)
    assert "HIDDEN" not in text and "x=1" not in text
    assert "Hello world" in text
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: finassist.ingest`)

Run: `uv run pytest tests/test_ingest.py -v`

- [ ] **Step 3: Implement**

`src/finassist/ingest.py`:
```python
from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finassist.schemas import Chunk

KEEP_ITEMS = {"1", "1A", "7", "7A", "8"}
# \b after the group stops "Item 10" from matching as Item "1".
ITEM_RE = re.compile(
    r"^[ \t]*item[ \t]+(1a|1b|1c|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16|1)\b[ \t]*[\.\:\-–—]?",
    re.IGNORECASE | re.MULTILINE,
)
_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    text = soup.get_text("\n").replace("\xa0", " ")
    lines = [ln.strip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_items(text: str) -> dict[str, str]:
    """Return {item: body}. 10-Ks list every Item twice (table of contents + body);
    keeping the longest section per item selects the body."""
    matches = list(ITEM_RE.finditer(text))
    best: dict[str, str] = {}
    for i, m in enumerate(matches):
        item = m.group(1).upper()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if item in KEEP_ITEMS and len(body) > len(best.get(item, "")):
            best[item] = body
    return best


def chunk_filing(ticker: str, fiscal_year: int, text: str, source_url: str = "") -> list[Chunk]:
    chunks: list[Chunk] = []
    for item, body in sorted(split_items(text).items()):
        for i, piece in enumerate(_splitter.split_text(body)):
            chunks.append(Chunk(
                chunk_id=f"{ticker}-{fiscal_year}-{item}-{i:03d}",
                text=piece, ticker=ticker, fiscal_year=fiscal_year,
                item=item, source_url=source_url,
            ))
    return chunks


def load_filing(ticker: str, raw_dir: Path) -> list[Chunk]:
    meta = json.loads((raw_dir / f"{ticker}.json").read_text())
    text = html_to_text((raw_dir / f"{ticker}.html").read_text(errors="ignore"))
    return chunk_filing(ticker, meta["fiscal_year"], text, meta["source_url"])
```

- [ ] **Step 4: Verify** — `uv run pytest tests/test_ingest.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/finassist/ingest.py tests/test_ingest.py
git commit -m "feat: parse 10-K html into Item-based chunks" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Fetch real filings and sanity-check parsing

**Files:**
- Create: `scripts/fetch_filings.py`, `scripts/find_in_filing.py`
- Test: manual verification (network)

**Interfaces:**
- Consumes: `get_settings`, `TICKERS`, `load_filing` (Tasks 1–2).
- Produces: `data/raw/{TICKER}.html` and `data/raw/{TICKER}.json` with keys `ticker, fiscal_year, source_url, filing_date`.

- [ ] **Step 1: Write the fetch script**

`scripts/fetch_filings.py`:
```python
"""Download the latest 10-K of each ticker from SEC EDGAR into data/raw/."""
import json
import time

import requests

from finassist.config import TICKERS, get_settings

CIKS = {"AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045459"}


def main() -> None:
    s = get_settings()
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": s.sec_user_agent}
    for t in TICKERS:
        sub = requests.get(f"https://data.sec.gov/submissions/CIK{CIKS[t]}.json", headers=headers, timeout=30)
        sub.raise_for_status()
        recent = sub.json()["filings"]["recent"]
        idx = next(i for i, f in enumerate(recent["form"]) if f == "10-K")
        acc = recent["accessionNumber"][idx].replace("-", "")
        doc = recent["primaryDocument"][idx]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(CIKS[t])}/{acc}/{doc}"
        time.sleep(0.3)
        html = requests.get(url, headers=headers, timeout=60)
        html.raise_for_status()
        (s.raw_dir / f"{t}.html").write_text(html.text)
        meta = {
            "ticker": t,
            "fiscal_year": int(recent["reportDate"][idx][:4]),
            "source_url": url,
            "filing_date": recent["filingDate"][idx],
        }
        (s.raw_dir / f"{t}.json").write_text(json.dumps(meta, indent=2))
        print(t, meta)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
```
`scripts/find_in_filing.py` (used later to write ground truth):
```python
"""Usage: uv run python scripts/find_in_filing.py AAPL "total net sales" [--item 8]"""
import argparse

from finassist.config import get_settings
from finassist.ingest import load_filing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ticker")
    p.add_argument("term")
    p.add_argument("--item")
    a = p.parse_args()
    for c in load_filing(a.ticker, get_settings().raw_dir):
        if a.item and c.item != a.item:
            continue
        if a.term.lower() in c.text.lower():
            print(f"--- {c.chunk_id}\n{c.text}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the fetch**

Run: `export SEC_USER_AGENT="<your name> <your email>"` (SEC requires a real contact; also put it in `.env`), then `uv run python scripts/fetch_filings.py`
Expected: three lines `AAPL {...}`, `MSFT {...}`, `NVDA {...}`; files in `data/raw/`.

- [ ] **Step 3: Sanity-check chunk counts per Item**

Run:
```bash
uv run python - <<'EOF'
from collections import Counter
from finassist.config import TICKERS, get_settings
from finassist.ingest import load_filing
s = get_settings()
for t in TICKERS:
    c = load_filing(t, s.raw_dir)
    print(t, len(c), dict(Counter(x.item for x in c)))
EOF
```
Expected: for each ticker items `1, 1A, 7, 8` all present, and Item `1A` and `7` each have well over 10 chunks. If an Item is missing or has < 5 chunks, open `data/raw/{T}.html`, find how that heading is written, and widen `ITEM_RE` in `src/finassist/ingest.py` (add a test case to `tests/test_ingest.py` reproducing the heading format first).

- [ ] **Step 4: Commit** (data/ is gitignored)

```bash
git add scripts
git commit -m "feat: add SEC fetch and filing search scripts" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Vector store (R3 idempotency)

**Files:**
- Create: `src/finassist/vectorstore.py`
- Test: `tests/test_vectorstore.py`

**Interfaces:**
- Consumes: `Chunk`, `Settings`.
- Produces: `get_embeddings(settings) -> Embeddings`; `FilingStore(persist_dir: Path, embeddings)` with `upsert(chunks: list[Chunk]) -> None`, `search(query: str, ticker: str, k: int = 5) -> list[Chunk]` (score = distance, lower is closer), `count() -> int`.

- [ ] **Step 1: Failing test**

`tests/test_vectorstore.py`:
```python
from langchain_core.embeddings import DeterministicFakeEmbedding

from finassist.schemas import Chunk
from finassist.vectorstore import FilingStore


def mk(ticker, i, text):
    return Chunk(chunk_id=f"{ticker}-2025-7-{i:03d}", text=text, ticker=ticker, fiscal_year=2025, item="7", source_url="u")


def test_upsert_is_idempotent_and_filters_by_ticker(tmp_path):
    store = FilingStore(tmp_path, DeterministicFakeEmbedding(size=32))
    chunks = [mk("AAPL", 0, "iPhone sales"), mk("AAPL", 1, "Services"), mk("MSFT", 0, "Azure growth")]
    store.upsert(chunks)
    store.upsert(chunks)
    assert store.count() == 3
    hits = store.search("anything", "MSFT", k=5)
    assert [h.chunk_id for h in hits] == ["MSFT-2025-7-000"]
    assert hits[0].score is not None and hits[0].item == "7"
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_vectorstore.py -v` (ModuleNotFoundError)

- [ ] **Step 3: Implement**

`src/finassist/vectorstore.py`:
```python
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from finassist.config import Settings
from finassist.schemas import Chunk


def get_embeddings(settings: Settings) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


class FilingStore:
    def __init__(self, persist_dir: Path, embeddings: Embeddings, collection: str = "filings"):
        self._db = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._db.add_texts(
            texts=[c.text for c in chunks],
            metadatas=[
                {"ticker": c.ticker, "fiscal_year": c.fiscal_year, "item": c.item, "source_url": c.source_url}
                for c in chunks
            ],
            ids=[c.chunk_id for c in chunks],
        )

    def search(self, query: str, ticker: str, k: int = 5) -> list[Chunk]:
        results = self._db.similarity_search_with_score(query, k=k, filter={"ticker": ticker})
        return [
            Chunk(
                chunk_id=doc.id or "", text=doc.page_content, ticker=doc.metadata["ticker"],
                fiscal_year=int(doc.metadata["fiscal_year"]), item=doc.metadata["item"],
                source_url=doc.metadata.get("source_url", ""), score=float(score),
            )
            for doc, score in results
        ]

    def count(self) -> int:
        return self._db._collection.count()
```

- [ ] **Step 4: Verify** — `uv run pytest tests/test_vectorstore.py -v` → 1 passed. If `doc.id` is `None` in the installed version, read the id from `doc.metadata.get("id")` or switch to `self._db._collection.get(...)`; keep the test as the contract.

- [ ] **Step 5: Commit**

```bash
git add src/finassist/vectorstore.py tests/test_vectorstore.py
git commit -m "feat: chroma-backed filing store with ticker filter" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Deterministic price tool (R2)

**Files:**
- Create: `src/finassist/tools.py`
- Test: `tests/test_price_tool.py`

**Interfaces:**
- Consumes: `PriceInfo`.
- Produces: `PriceUnavailable(Exception)`; `get_stock_price(ticker: str, cache_path: Path, fetcher: Callable[[str], dict] | None = None, offline: bool = False) -> PriceInfo`. Fetcher returns `{"price": float, "market_cap": float|None, "pe_ratio": float|None}`.

- [ ] **Step 1: Failing tests**

`tests/test_price_tool.py`:
```python
import pytest

from finassist.tools import PriceUnavailable, get_stock_price


def ok(_t):
    return {"price": 200.5, "market_cap": 3.0e12, "pe_ratio": 31.2}


def boom(_t):
    raise ConnectionError("no network")


def test_fetch_writes_cache(tmp_path):
    p = get_stock_price("AAPL", tmp_path / "p.json", fetcher=ok)
    assert p.price == 200.5 and p.stale is False and p.ticker == "AAPL"
    assert (tmp_path / "p.json").exists()


def test_offline_is_deterministic_from_cache(tmp_path):
    cache = tmp_path / "p.json"
    get_stock_price("AAPL", cache, fetcher=ok)
    a = get_stock_price("AAPL", cache, fetcher=boom)
    b = get_stock_price("AAPL", cache, fetcher=boom)
    assert a == b and a.stale is True and a.price == 200.5


def test_forced_offline_never_calls_fetcher(tmp_path):
    cache = tmp_path / "p.json"
    get_stock_price("AAPL", cache, fetcher=ok)
    p = get_stock_price("AAPL", cache, fetcher=boom, offline=True)
    assert p.stale is True


def test_no_cache_and_no_network_raises(tmp_path):
    with pytest.raises(PriceUnavailable):
        get_stock_price("NVDA", tmp_path / "p.json", fetcher=boom)
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_price_tool.py -v`

- [ ] **Step 3: Implement**

`src/finassist/tools.py`:
```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from finassist.schemas import PriceInfo


class PriceUnavailable(Exception):
    pass


def _yf_fetch(ticker: str) -> dict:
    import yfinance as yf

    info = yf.Ticker(ticker).info
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None:
        raise ValueError(f"no price for {ticker}")
    return {"price": float(price), "market_cap": info.get("marketCap"), "pe_ratio": info.get("trailingPE")}


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def get_stock_price(
    ticker: str,
    cache_path: Path,
    fetcher: Callable[[str], dict] | None = None,
    offline: bool = False,
) -> PriceInfo:
    cache = _read(cache_path)
    if not offline:
        try:
            data = (fetcher or _yf_fetch)(ticker)
            entry = {**data, "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            cache[ticker] = entry
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2))
            return PriceInfo(ticker=ticker, stale=False, **entry)
        except Exception:
            pass
    if ticker in cache:
        return PriceInfo(ticker=ticker, stale=True, **cache[ticker])
    raise PriceUnavailable(f"price for {ticker} unavailable (no network, no cache)")
```

- [ ] **Step 4: Verify** — `uv run pytest tests/test_price_tool.py -v` → 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/finassist/tools.py tests/test_price_tool.py
git commit -m "feat: deterministic stock price tool with cache fallback" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Postprocess: citations and number grounding (R4, R5)

**Files:**
- Create: `src/finassist/postprocess.py`
- Test: `tests/test_postprocess.py`

**Interfaces:**
- Consumes: `Chunk`, `Citation`, `FinanceAnswer`.
- Produces: `MARKER_RE`; `extract_citations(answer: str, chunks: list[Chunk]) -> list[Citation]` (deduped, order of first appearance, only ids present in `chunks`); `ground_numbers(out: FinanceAnswer, chunks: list[Chunk]) -> FinanceAnswer` (sets `revenue`/`eps` to `None` unless the number text appears in a retrieved chunk).

- [ ] **Step 1: Failing tests**

`tests/test_postprocess.py`:
```python
from finassist.postprocess import MARKER_RE, extract_citations, ground_numbers
from finassist.schemas import Chunk, FinanceAnswer, Money


def ch(cid, text):
    t, fy, item, _ = cid.split("-")
    return Chunk(chunk_id=cid, text=text, ticker=t, fiscal_year=int(fy), item=item)


CHUNKS = [ch("AAPL-2025-8-000", "Total net sales 416,161 and diluted EPS 7.46"), ch("AAPL-2025-1A-003", "Supply risk")]


def test_marker_regex():
    assert MARKER_RE.findall("a [AAPL-2025-1A-003] b [NVDA-2026-7A-012]") == ["AAPL-2025-1A-003", "NVDA-2026-7A-012"]


def test_extract_citations_dedupes_and_drops_unknown():
    ans = "x [AAPL-2025-8-000] y [AAPL-2025-8-000] z [AAPL-2025-1A-003] w [MSFT-2025-7-001]"
    cites = extract_citations(ans, CHUNKS)
    assert [c.chunk_id for c in cites] == ["AAPL-2025-8-000", "AAPL-2025-1A-003"]
    assert cites[0].item == "8" and cites[0].source == "AAPL 10-K FY2025"
    assert cites[0].quote.startswith("Total net sales")


def test_ground_numbers_keeps_supported_and_nulls_unsupported():
    good = FinanceAnswer(intent="filing", answer="a", revenue=Money(value=416161, period="FY2025"), eps=7.46)
    assert ground_numbers(good, CHUNKS).revenue is not None
    assert ground_numbers(good, CHUNKS).eps == 7.46
    bad = FinanceAnswer(intent="filing", answer="a", revenue=Money(value=999999, period="FY2025"), eps=1.23)
    out = ground_numbers(bad, CHUNKS)
    assert out.revenue is None and out.eps is None
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_postprocess.py -v`

- [ ] **Step 3: Implement**

`src/finassist/postprocess.py`:
```python
from __future__ import annotations

import re

from finassist.schemas import Chunk, Citation, FinanceAnswer

MARKER_RE = re.compile(r"\[([A-Z]+-\d{4}-[0-9A-Z]+-\d{3})\]")


def extract_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    by_id = {c.chunk_id: c for c in chunks}
    seen: list[str] = []
    for cid in MARKER_RE.findall(answer):
        if cid in by_id and cid not in seen:
            seen.append(cid)
    return [
        Citation(
            source=f"{by_id[i].ticker} 10-K FY{by_id[i].fiscal_year}",
            item=by_id[i].item, chunk_id=i, quote=by_id[i].text[:200],
        )
        for i in seen
    ]


def ground_numbers(out: FinanceAnswer, chunks: list[Chunk]) -> FinanceAnswer:
    """Drop revenue/eps unless the printed number appears in a retrieved chunk."""
    corpus = " ".join(c.text for c in chunks)
    update: dict = {}
    if out.revenue is not None and f"{out.revenue.value:,.0f}" not in corpus:
        update["revenue"] = None
    if out.eps is not None and f"{out.eps:.2f}" not in corpus:
        update["eps"] = None
    return out.model_copy(update=update)
```

- [ ] **Step 4: Verify** — `uv run pytest tests/test_postprocess.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/finassist/postprocess.py tests/test_postprocess.py
git commit -m "feat: deterministic citations and number grounding" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: LangGraph workflow with injected deps (R1, R4, R9)

**Files:**
- Create: `src/finassist/graph.py`, `tests/conftest.py`
- Test: `tests/test_graph_routing.py`, `tests/test_citations.py`

**Interfaces:**
- Consumes: Tasks 1, 5, 6 (`IntentResult`, `Chunk`, `PriceInfo`, `FinanceAnswer`, `PriceUnavailable`, `extract_citations`, `ground_numbers`, `MARKER_RE`, `TICKERS`).
- Produces:
  - `Deps(classify, retrieve, price, generate, structure)` dataclass with callables
    `classify(question: str) -> IntentResult`,
    `retrieve(question: str, tickers: list[str]) -> list[Chunk]`,
    `price(ticker: str) -> PriceInfo`,
    `generate(question: str, chunks: list[Chunk], prices: list[PriceInfo]) -> str`,
    `structure(question: str, answer: str, chunks: list[Chunk], prices: list[PriceInfo], intent: Intent) -> FinanceAnswer`.
  - `build_graph(deps: Deps)` returning a compiled graph; state keys `question, intent, tickers, price_data, retrieved_chunks, answer, output, error`.
  - Constants `REFUSAL_TEXT`, `INSUFFICIENT_TEXT`, `PRICE_UNAVAILABLE_TEXT`.

- [ ] **Step 1: Shared fakes and failing tests**

`tests/conftest.py`:
```python
import pytest

from finassist.graph import Deps
from finassist.schemas import Chunk, FinanceAnswer, IntentResult, PriceInfo


class Calls(list):
    pass


@pytest.fixture
def make_deps():
    def factory(intent="filing", tickers=("AAPL",), chunks=None, generate=None, price_fn=None):
        calls = Calls()
        default_chunk = Chunk(chunk_id="AAPL-2025-7-000", text="Total net sales 416,161", ticker="AAPL",
                              fiscal_year=2025, item="7")
        chunks = [default_chunk] if chunks is None else chunks

        def classify(q):
            calls.append("classify")
            return IntentResult(intent=intent, tickers=list(tickers))

        def retrieve(q, tks):
            calls.append("retrieve")
            return chunks

        def price(t):
            calls.append("price")
            if price_fn:
                return price_fn(t)
            return PriceInfo(ticker=t, price=200.0, as_of="2026-10-28T00:00:00+00:00")

        def gen(q, cs, ps):
            calls.append("generate")
            return "Net sales were 416,161 [AAPL-2025-7-000]."

        def structure(q, ans, cs, ps, it):
            calls.append("structure")
            return FinanceAnswer(intent=it, answer=ans)

        deps = Deps(classify=classify, retrieve=retrieve, price=price,
                    generate=generate or gen, structure=structure)
        return deps, calls

    return factory
```
`tests/test_graph_routing.py`:
```python
import pytest

from finassist.graph import INSUFFICIENT_TEXT, REFUSAL_TEXT, build_graph
from finassist.tools import PriceUnavailable


def run(deps, q="q"):
    return build_graph(deps).invoke({"question": q})


@pytest.mark.parametrize("intent,expected", [
    ("price", ["classify", "price", "generate", "structure"]),
    ("filing", ["classify", "retrieve", "generate", "structure"]),
    ("both", ["classify", "price", "retrieve", "generate", "structure"]),
])
def test_each_intent_takes_its_own_path(make_deps, intent, expected):
    deps, calls = make_deps(intent=intent)
    run(deps)
    assert list(calls) == expected


def test_out_of_scope_refuses_without_retrieval(make_deps):
    deps, calls = make_deps(intent="out_of_scope", tickers=())
    out = run(deps)["output"]
    assert list(calls) == ["classify"]
    assert out.intent == "out_of_scope" and out.answer == REFUSAL_TEXT and out.citations == []


def test_in_scope_intent_without_known_ticker_is_refused(make_deps):
    deps, calls = make_deps(intent="filing", tickers=("TSLA",))
    assert run(deps)["output"].intent == "out_of_scope"
    assert "retrieve" not in calls


def test_no_chunks_returns_insufficient_without_calling_llm(make_deps):
    deps, calls = make_deps(intent="filing", chunks=[])
    out = run(deps)["output"]
    assert out.answer == INSUFFICIENT_TEXT
    assert "generate" not in calls


def test_price_failure_is_reported_not_raised(make_deps):
    def fail(_t):
        raise PriceUnavailable("down")
    deps, _ = make_deps(intent="price", price_fn=fail)
    state = run(deps)
    assert "unavailable" in state["output"].answer.lower() or "không" in state["output"].answer.lower()
    assert "down" in state["error"]
```
`tests/test_citations.py`:
```python
from finassist.graph import build_graph


def test_filing_answer_carries_citation_to_retrieved_chunk(make_deps):
    deps, _ = make_deps(intent="filing")
    out = build_graph(deps).invoke({"question": "q"})["output"]
    assert [c.chunk_id for c in out.citations] == ["AAPL-2025-7-000"]


def test_missing_marker_retries_once_then_flags_error(make_deps):
    n = {"calls": 0}

    def gen(q, cs, ps):
        n["calls"] += 1
        return "No marker here."

    deps, _ = make_deps(intent="filing", generate=gen)
    state = build_graph(deps).invoke({"question": "q"})
    assert n["calls"] == 2
    assert state["error"] == "missing_citation"
    assert state["output"].citations == []


def test_hallucinated_chunk_id_is_not_cited(make_deps):
    deps, _ = make_deps(intent="filing", generate=lambda q, c, p: "x [MSFT-2025-7-099] y [AAPL-2025-7-000]")
    out = build_graph(deps).invoke({"question": "q"})["output"]
    assert [c.chunk_id for c in out.citations] == ["AAPL-2025-7-000"]
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_graph_routing.py tests/test_citations.py -v` (ModuleNotFoundError: finassist.graph)

- [ ] **Step 3: Implement**

`src/finassist/graph.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from finassist.config import TICKERS
from finassist.postprocess import MARKER_RE, extract_citations, ground_numbers
from finassist.schemas import Chunk, FinanceAnswer, Intent, IntentResult, PriceInfo
from finassist.tools import PriceUnavailable

REFUSAL_TEXT = (
    "Tôi chỉ hỗ trợ câu hỏi về báo cáo 10-K và giá cổ phiếu của AAPL, MSFT, NVDA, "
    "và không đưa ra tư vấn đầu tư. / I only answer questions about the 10-K filings and "
    "stock prices of AAPL, MSFT and NVDA, and I do not give investment advice."
)
INSUFFICIENT_TEXT = "Không đủ thông tin trong 10-K để trả lời. / Not enough information in the 10-K filings."
PRICE_UNAVAILABLE_TEXT = "Dữ liệu giá hiện không khả dụng. / Price data is currently unavailable."


class GraphState(TypedDict, total=False):
    question: str
    intent: Intent
    tickers: list[str]
    price_data: list[PriceInfo]
    retrieved_chunks: list[Chunk]
    answer: str
    output: FinanceAnswer
    error: str | None


@dataclass
class Deps:
    classify: Callable[[str], IntentResult]
    retrieve: Callable[[str, list[str]], list[Chunk]]
    price: Callable[[str], PriceInfo]
    generate: Callable[[str, list[Chunk], list[PriceInfo]], str]
    structure: Callable[[str, str, list[Chunk], list[PriceInfo], Intent], FinanceAnswer]


def _join(*parts: str | None) -> str | None:
    return "; ".join(p for p in parts if p) or None


def build_graph(deps: Deps):
    def classify_intent(state: GraphState) -> GraphState:
        res = deps.classify(state["question"])
        tickers = [t for t in dict.fromkeys(res.tickers) if t in TICKERS]
        intent = res.intent if (tickers or res.intent == "out_of_scope") else "out_of_scope"
        return {"intent": intent, "tickers": tickers}

    def route(state: GraphState) -> str:
        return {"price": "price_tool", "both": "price_tool", "filing": "retrieve", "out_of_scope": "refuse"}[state["intent"]]

    def price_tool(state: GraphState) -> GraphState:
        prices, errors = [], []
        for t in state["tickers"]:
            try:
                prices.append(deps.price(t))
            except PriceUnavailable as e:
                errors.append(str(e))
        return {"price_data": prices, "error": _join(*errors)}

    def after_price(state: GraphState) -> str:
        return "retrieve" if state["intent"] == "both" else "generate_answer"

    def retrieve(state: GraphState) -> GraphState:
        return {"retrieved_chunks": deps.retrieve(state["question"], state["tickers"])}

    def generate_answer(state: GraphState) -> GraphState:
        intent = state["intent"]
        chunks = state.get("retrieved_chunks", [])
        prices = state.get("price_data", [])
        needs_filing = intent in ("filing", "both")
        if needs_filing and not chunks:
            return {"answer": INSUFFICIENT_TEXT}
        if intent == "price" and not prices:
            return {"answer": PRICE_UNAVAILABLE_TEXT}
        answer = ""
        for _ in range(2):
            answer = deps.generate(state["question"], chunks, prices)
            if not needs_filing or MARKER_RE.search(answer):
                return {"answer": answer}
        return {"answer": answer, "error": _join(state.get("error"), "missing_citation")}

    def format_output(state: GraphState) -> GraphState:
        intent, answer = state["intent"], state["answer"]
        chunks = state.get("retrieved_chunks", [])
        prices = state.get("price_data", [])
        if answer in (INSUFFICIENT_TEXT, PRICE_UNAVAILABLE_TEXT):
            return {"output": FinanceAnswer(intent=intent, answer=answer)}
        out = deps.structure(state["question"], answer, chunks, prices, intent)
        out = ground_numbers(out, chunks)
        out = out.model_copy(update={
            "citations": extract_citations(answer, chunks) if intent in ("filing", "both") else [],
            "stock_price": prices[0] if len(prices) == 1 else out.stock_price,
        })
        return {"output": out}

    def refuse(state: GraphState) -> GraphState:
        return {"answer": REFUSAL_TEXT,
                "output": FinanceAnswer(intent="out_of_scope", answer=REFUSAL_TEXT)}

    g = StateGraph(GraphState)
    for name, fn in [("classify_intent", classify_intent), ("price_tool", price_tool), ("retrieve", retrieve),
                     ("generate_answer", generate_answer), ("format_output", format_output), ("refuse", refuse)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route, ["price_tool", "retrieve", "refuse"])
    g.add_conditional_edges("price_tool", after_price, ["retrieve", "generate_answer"])
    g.add_edge("retrieve", "generate_answer")
    g.add_edge("generate_answer", "format_output")
    g.add_edge("format_output", END)
    g.add_edge("refuse", END)
    return g.compile()
```

- [ ] **Step 4: Verify** — `uv run pytest tests/test_graph_routing.py tests/test_citations.py -v` → all pass (9 tests). Note `test_price_failure_is_reported_not_raised`: with intent `price` and no price, `generate_answer` returns `PRICE_UNAVAILABLE_TEXT` which contains both "unavailable" and "không".

- [ ] **Step 5: Commit**

```bash
git add src/finassist/graph.py tests/conftest.py tests/test_graph_routing.py tests/test_citations.py
git commit -m "feat: LangGraph workflow with intent routing and citation enforcement" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Real LLM chains and runtime wiring

**Files:**
- Create: `src/finassist/chains.py`, `src/finassist/runtime.py`
- Test: `tests/test_chains.py` (prompt/format helpers, no network), `tests/test_llm_smoke.py` extended

**Interfaces:**
- Consumes: `Deps`, `build_graph`, `FilingStore`, `get_stock_price`, `LLMExtraction`, `IntentResult`, `Settings`.
- Produces:
  - `build_llm(settings: Settings, model: str | None = None) -> ChatAnthropic`
  - `format_context(chunks: list[Chunk], prices: list[PriceInfo]) -> str`
  - `message_text(msg) -> str`
  - `make_deps(settings: Settings, store: FilingStore, llm) -> Deps`
  - `build_app(settings: Settings | None = None)` → compiled graph; `ask(app, question: str, tags: list[str] | None = None) -> dict` (final graph state).

- [ ] **Step 1: Failing tests for pure helpers**

`tests/test_chains.py`:
```python
from types import SimpleNamespace

from finassist.chains import format_context, message_text
from finassist.schemas import Chunk, PriceInfo


def test_format_context_lists_chunk_ids_and_price():
    c = Chunk(chunk_id="AAPL-2025-1A-001", text="Supply risk", ticker="AAPL", fiscal_year=2025, item="1A")
    p = PriceInfo(ticker="AAPL", price=200.0, as_of="2026-10-28T00:00:00+00:00", stale=True)
    ctx = format_context([c], [p])
    assert "[AAPL-2025-1A-001]" in ctx and "Supply risk" in ctx
    assert "AAPL" in ctx and "200.0" in ctx and "stale" in ctx


def test_message_text_handles_block_lists():
    assert message_text(SimpleNamespace(content="hi")) == "hi"
    blocks = [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert message_text(SimpleNamespace(content=blocks)) == "ab"
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_chains.py -v`

- [ ] **Step 3: Implement chains**

`src/finassist/chains.py`:
```python
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from finassist.config import TICKERS, Settings
from finassist.graph import Deps
from finassist.schemas import Chunk, FinanceAnswer, Intent, IntentResult, LLMExtraction, PriceInfo
from finassist.tools import get_stock_price
from finassist.vectorstore import FilingStore

CLASSIFY_SYSTEM = f"""You route questions for a finance assistant covering ONLY these companies: AAPL (Apple), MSFT (Microsoft), NVDA (NVIDIA).
Return intent:
- "price": needs current stock price, market cap or P/E only.
- "filing": needs facts from the company's latest 10-K (revenue, EPS, net income, risk factors, MD&A, business).
- "both": needs both a current price/market data and 10-K facts.
- "out_of_scope": other companies, investment advice (should I buy/sell, price predictions), or anything not about these companies' finances.
Return tickers as a list drawn from {list(TICKERS)} (map company names to tickers). Use [] if none apply."""

GENERATE_SYSTEM = """You answer finance questions using ONLY the provided context.
Rules:
1. Every factual claim from a filing must end with the chunk id in square brackets, e.g. [AAPL-2025-1A-012]. Use only ids that appear in the context.
2. Price data comes from the price block; mention it is stale if marked stale. Do not cite chunk ids for price data.
3. If the context lacks the answer, say so. Never guess or use outside knowledge.
4. No investment advice. Answer in the language of the question. Be concise."""

EXTRACT_SYSTEM = """Extract structured fields from the ANSWER using the CONTEXT.
- revenue.value: total revenue in USD millions EXACTLY as printed in the context (e.g. 416,161 -> 416161). Leave null if not asked or not stated.
- eps: diluted EPS exactly as printed. Leave null if not asked or not stated.
- risk_factors: short phrases, only if the question is about risks.
- ticker / fiscal_year: only if the answer concerns a single company."""


def build_llm(settings: Settings, model: str | None = None) -> ChatAnthropic:
    return ChatAnthropic(
        model=model or settings.llm_model, base_url=settings.base_url,
        api_key=settings.api_key, temperature=0, max_tokens=2048,
    )


def message_text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text")


def format_context(chunks: list[Chunk], prices: list[PriceInfo]) -> str:
    parts = [f"[{c.chunk_id}] ({c.ticker} FY{c.fiscal_year} Item {c.item})\n{c.text}" for c in chunks]
    if prices:
        lines = [
            f"{p.ticker}: price={p.price}, market_cap={p.market_cap}, pe={p.pe_ratio}, as_of={p.as_of}"
            + (" (stale)" if p.stale else "")
            for p in prices
        ]
        parts.append("PRICE DATA\n" + "\n".join(lines))
    return "\n\n".join(parts)


def make_deps(settings: Settings, store: FilingStore, llm: ChatAnthropic) -> Deps:
    classifier = llm.with_structured_output(IntentResult)
    extractor = llm.with_structured_output(LLMExtraction)
    cache_path = settings.cache_dir / "prices.json"

    def classify(question: str) -> IntentResult:
        return classifier.invoke([("system", CLASSIFY_SYSTEM), ("human", question)])

    def retrieve(question: str, tickers: list[str]) -> list[Chunk]:
        return [c for t in tickers for c in store.search(question, t, settings.top_k)]

    def price(ticker: str) -> PriceInfo:
        return get_stock_price(ticker, cache_path, offline=settings.price_offline)

    def generate(question: str, chunks: list[Chunk], prices: list[PriceInfo]) -> str:
        msg = llm.invoke([("system", GENERATE_SYSTEM),
                          ("human", f"CONTEXT\n{format_context(chunks, prices)}\n\nQUESTION\n{question}")])
        return message_text(msg)

    def structure(question: str, answer: str, chunks: list[Chunk], prices: list[PriceInfo],
                  intent: Intent) -> FinanceAnswer:
        ext = LLMExtraction()
        if chunks:
            ext = extractor.invoke([("system", EXTRACT_SYSTEM),
                                    ("human", f"QUESTION\n{question}\n\nANSWER\n{answer}\n\nCONTEXT\n{format_context(chunks, [])}")])
        return FinanceAnswer(intent=intent, answer=answer, ticker=ext.ticker, fiscal_year=ext.fiscal_year,
                             revenue=ext.revenue, eps=ext.eps, risk_factors=ext.risk_factors)

    return Deps(classify=classify, retrieve=retrieve, price=price, generate=generate, structure=structure)
```
`src/finassist/runtime.py`:
```python
from __future__ import annotations

from finassist import __version__
from finassist.chains import build_llm, make_deps
from finassist.config import Settings, get_settings
from finassist.graph import build_graph
from finassist.vectorstore import FilingStore, get_embeddings


def build_app(settings: Settings | None = None):
    s = settings or get_settings()
    store = FilingStore(s.chroma_dir, get_embeddings(s))
    return build_graph(make_deps(s, store, build_llm(s)))


def ask(app, question: str, tags: list[str] | None = None) -> dict:
    return app.invoke(
        {"question": question},
        config={
            "run_name": "finassist",
            "tags": tags or ["demo"],
            "metadata": {"app_version": __version__},
        },
    )
```

- [ ] **Step 4: Verify helpers** — `uv run pytest tests/test_chains.py -v` → 2 passed. Then run the full suite: `uv run pytest -v` → all green.

- [ ] **Step 5: Commit**

```bash
git add src/finassist/chains.py src/finassist/runtime.py tests/test_chains.py
git commit -m "feat: Claude-backed chains and runtime wiring" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: CLI (ingest, ask, replay) and first live run (R11)

**Files:**
- Create: `src/finassist/cli.py`, `runs/.gitkeep`, `reports/.gitkeep`
- Test: `tests/test_cli_replay.py`

**Interfaces:**
- Consumes: `build_app`, `ask`, `get_settings`, `load_filing`, `FilingStore`, `get_embeddings`, `TICKERS`.
- Produces: `main()` entry point. Commands: `finassist ingest`; `finassist ask "<q>" [--record FILE] [--replay FILE]`. Record format: `{"question", "output": FinanceAnswer dict, "retrieved": [{chunk_id,item,score,text}], "error"}`.

- [ ] **Step 1: Failing test for replay (no network, no models)**

`tests/test_cli_replay.py`:
```python
import json

from finassist.cli import render_record, main


def test_render_record_prints_answer_and_citations():
    rec = {"question": "q", "error": None, "retrieved": [],
           "output": {"intent": "filing", "answer": "Net sales 416,161 [AAPL-2025-8-000]",
                      "citations": [{"source": "AAPL 10-K FY2025", "item": "8", "chunk_id": "AAPL-2025-8-000", "quote": "x"}],
                      "risk_factors": []}}
    text = render_record(rec)
    assert "Net sales 416,161" in text and "AAPL-2025-8-000" in text and "AAPL 10-K FY2025" in text


def test_replay_runs_without_building_the_app(tmp_path, capsys, monkeypatch):
    f = tmp_path / "demo.json"
    f.write_text(json.dumps({"question": "q", "error": None, "retrieved": [],
                             "output": {"intent": "out_of_scope", "answer": "nope", "citations": [], "risk_factors": []}}))
    monkeypatch.setattr("sys.argv", ["finassist", "ask", "ignored", "--replay", str(f)])
    main()
    assert "nope" in capsys.readouterr().out
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_cli_replay.py -v`

- [ ] **Step 3: Implement**

`src/finassist/cli.py`:
```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_record(rec: dict) -> str:
    out = rec["output"]
    lines = [f"Q: {rec['question']}", f"Intent: {out['intent']}", "", out["answer"], ""]
    if out.get("citations"):
        lines.append("Citations:")
        lines += [f"  - {c['source']} · Item {c['item']} · {c['chunk_id']}" for c in out["citations"]]
        lines.append("")
    if rec.get("error"):
        lines.append(f"[error] {rec['error']}")
    lines += ["Structured output:", json.dumps(out, indent=2, ensure_ascii=False)]
    return "\n".join(lines)


def _record(state: dict) -> dict:
    return {
        "question": state["question"],
        "error": state.get("error"),
        "output": state["output"].model_dump(),
        "retrieved": [
            {"chunk_id": c.chunk_id, "item": c.item, "score": c.score, "text": c.text}
            for c in state.get("retrieved_chunks", [])
        ],
    }


def _cmd_ingest() -> None:
    from finassist.config import TICKERS, get_settings
    from finassist.ingest import load_filing
    from finassist.vectorstore import FilingStore, get_embeddings

    s = get_settings()
    store = FilingStore(s.chroma_dir, get_embeddings(s))
    for t in TICKERS:
        chunks = load_filing(t, s.raw_dir)
        store.upsert(chunks)
        print(f"{t}: {len(chunks)} chunks")
    print(f"total in store: {store.count()}")


def _cmd_ask(question: str, record: str | None, replay: str | None) -> None:
    if replay:
        print(render_record(json.loads(Path(replay).read_text())))
        return
    from finassist.runtime import ask, build_app

    rec = _record(ask(build_app(), question, tags=["demo"]))
    if record:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        Path(record).write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    print(render_record(rec))


def main() -> None:
    p = argparse.ArgumentParser(prog="finassist")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--record")
    a.add_argument("--replay")
    args = p.parse_args()
    if args.cmd == "ingest":
        _cmd_ingest()
    else:
        _cmd_ask(args.question, args.record, args.replay)
```

- [ ] **Step 4: Verify tests** — `uv run pytest tests/test_cli_replay.py -v` → 2 passed.

- [ ] **Step 5: Live run (needs Tasks 3 data + router)**

Run: `uv run finassist ingest`
Expected: `AAPL: N chunks`, `MSFT: …`, `NVDA: …`, `total in store: <sum>` (first run downloads the bge model). Re-run: total unchanged (idempotent).
Run: `uv run finassist ask "What was Apple's total net sales in its most recent fiscal year?" --record runs/demo.json`
Expected: answer containing the figure with `[AAPL-…]` marker, a Citations list, and JSON with `revenue.value`. If `error: missing_citation` appears, tighten `GENERATE_SYSTEM` rule 1 and re-run. Then:
`uv run finassist ask "x" --replay runs/demo.json` → prints the same answer with no network (R11).
Also run a combined question: `uv run finassist ask "What is NVIDIA's current stock price and its biggest export-control risk?"` → populated `stock_price` and citations. This seeds `data/cache/prices.json`.

- [ ] **Step 6: Commit** (include the recorded fallback run)

```bash
git add src/finassist/cli.py tests/test_cli_replay.py runs/demo.json runs/.gitkeep reports/.gitkeep
git commit -m "feat: CLI with ingest, ask and replay" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Golden set of 30 questions (R6)

**Files:**
- Create: `eval/golden_set.json`
- Test: `tests/test_golden_set.py`

**Interfaces:**
- Produces: JSON list of 30 objects `{id, category, question, ticker, expected_intent, ground_truth}`; `category ∈ {numeric, risk, combined, trap}`; trap entries have `expected_intent="out_of_scope"` and `ground_truth=""`.

- [ ] **Step 1: Failing validation test**

`tests/test_golden_set.py`:
```python
import json
from collections import Counter

from finassist.config import ROOT

DATA = json.loads((ROOT / "eval" / "golden_set.json").read_text())
INTENTS = {"price", "filing", "both", "out_of_scope"}


def test_size_and_distribution():
    assert len(DATA) == 30
    assert Counter(d["category"] for d in DATA) == {"numeric": 10, "risk": 10, "combined": 5, "trap": 5}


def test_ids_unique_and_fields_present():
    assert len({d["id"] for d in DATA}) == 30
    for d in DATA:
        assert set(d) >= {"id", "category", "question", "ticker", "expected_intent", "ground_truth"}
        assert d["expected_intent"] in INTENTS


def test_ground_truth_rules():
    for d in DATA:
        if d["category"] == "trap":
            assert d["expected_intent"] == "out_of_scope"
        else:
            assert len(d["ground_truth"]) >= 20, d["id"]
            assert d["ticker"] in {"AAPL", "MSFT", "NVDA"}
```

- [ ] **Step 2: Run, expect FAIL** (file missing) — `uv run pytest tests/test_golden_set.py -v`

- [ ] **Step 3: Author the 30 questions**

Write `eval/golden_set.json` using exactly these questions (ids `q01`..`q30`). **Ground truth must be written by hand from the filings**, never by an LLM and never from memory: for each non-trap question run `uv run python scripts/find_in_filing.py <TICKER> "<term>" [--item N]`, read the chunk, and write a 1–3 sentence answer that restates the printed figure/facts and ends with the chunk id you read it from in parentheses, e.g. `Total net sales were $… million in fiscal 2025 (AAPL-2025-8-004).`

| id | category | ticker | expected_intent | question |
|---|---|---|---|---|
| q01 | numeric | AAPL | filing | What were Apple's total net sales in its most recent fiscal year? |
| q02 | numeric | AAPL | filing | What was Apple's diluted earnings per share in its most recent fiscal year? |
| q03 | numeric | AAPL | filing | How much net sales did Apple's iPhone segment generate in its most recent fiscal year? |
| q04 | numeric | MSFT | filing | What was Microsoft's total revenue in its most recent fiscal year? |
| q05 | numeric | MSFT | filing | What was Microsoft's net income in its most recent fiscal year? |
| q06 | numeric | MSFT | filing | What was Microsoft's diluted earnings per share in its most recent fiscal year? |
| q07 | numeric | NVDA | filing | What was NVIDIA's total revenue in its most recent fiscal year? |
| q08 | numeric | NVDA | filing | What was NVIDIA's net income in its most recent fiscal year? |
| q09 | numeric | NVDA | filing | What was NVIDIA's diluted earnings per share in its most recent fiscal year? |
| q10 | numeric | NVDA | filing | How much revenue did NVIDIA's Data Center market platform generate in its most recent fiscal year? |
| q11 | risk | AAPL | filing | What supply chain risks does Apple disclose in its 10-K? |
| q12 | risk | AAPL | filing | What legal and regulatory risks around the App Store does Apple describe? |
| q13 | risk | AAPL | filing | How does Apple say foreign currency fluctuations could affect its results? |
| q14 | risk | MSFT | filing | What risks related to artificial intelligence does Microsoft disclose? |
| q15 | risk | MSFT | filing | What cybersecurity risks does Microsoft describe in its 10-K? |
| q16 | risk | MSFT | filing | What competitive risks does Microsoft identify? |
| q17 | risk | NVDA | filing | How do export controls affect NVIDIA's business according to its 10-K? |
| q18 | risk | NVDA | filing | What manufacturing and supply dependency risks does NVIDIA disclose? |
| q19 | risk | NVDA | filing | What does NVIDIA say about customer concentration risk? |
| q20 | risk | NVDA | filing | What did NVIDIA's MD&A give as the main drivers of revenue change in its most recent fiscal year? |
| q21 | combined | AAPL | both | What is Apple's current stock price, and what were its total net sales last fiscal year? |
| q22 | combined | MSFT | both | What is Microsoft's current P/E ratio, and what is its biggest competitive risk? |
| q23 | combined | NVDA | both | What is NVIDIA's market capitalization now, and what was its total revenue last fiscal year? |
| q24 | combined | AAPL | both | Compare Apple's and Microsoft's latest annual revenue and tell me both current stock prices. |
| q25 | combined | NVDA | both | What is NVIDIA's current stock price and how do export controls threaten it? |
| q26 | trap | AAPL | out_of_scope | Should I buy NVIDIA stock now? |
| q27 | trap | TSLA | out_of_scope | What was Tesla's revenue last year? |
| q28 | trap | none | out_of_scope | Thời tiết Hà Nội hôm nay thế nào? |
| q29 | trap | AAPL | out_of_scope | Predict Apple's stock price next month. |
| q30 | trap | GOOGL | out_of_scope | What is Alphabet's diluted EPS? |

For `combined` rows (q21–q25) the volatile price figure must NOT go into `ground_truth`; write only the filing part and end with `Stock price/market data are provided by the price tool.` For q24 use `"ticker": "AAPL"` (the test only requires one of the three).
JSON shape per entry: `{"id":"q01","category":"numeric","question":"...","ticker":"AAPL","expected_intent":"filing","ground_truth":"..."}`; trap entries use `"ground_truth": ""`.

- [ ] **Step 4: Verify** — `uv run pytest tests/test_golden_set.py -v` → 3 passed. Note for traps with `ticker` TSLA/GOOGL/none the ticker rule is skipped (they are `trap`).

- [ ] **Step 5: Commit**

```bash
git add eval/golden_set.json tests/test_golden_set.py
git commit -m "feat: add hand-written 30-question golden set" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 11: RAGAS evaluation and baseline report (R7)

**Files:**
- Create: `eval/report.py`, `eval/run_ragas.py`, `eval/__init__.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `build_app`, `ask`, `build_llm`, `get_embeddings`, golden set.
- Produces: `summarize(df: pandas.DataFrame, metric_cols: list[str]) -> dict` (`{"mean": {col: float}, "by_category": {cat: {col: float}}}`); `render_markdown(summary, worst: list[dict], intent_acc: dict, meta: dict) -> str`; `reports/ragas_baseline.json` and `reports/ragas_baseline.md`.

- [ ] **Step 1: Failing test for the pure report functions**

`tests/test_report.py`:
```python
import pandas as pd

from eval.report import render_markdown, summarize


def test_summarize_means_overall_and_by_category():
    df = pd.DataFrame({
        "category": ["numeric", "numeric", "risk"],
        "faithfulness": [1.0, 0.5, 0.2],
        "context_recall": [1.0, 1.0, 0.0],
    })
    s = summarize(df, ["faithfulness", "context_recall"])
    assert round(s["mean"]["faithfulness"], 3) == 0.567
    assert s["by_category"]["numeric"]["faithfulness"] == 0.75
    assert s["by_category"]["risk"]["context_recall"] == 0.0


def test_render_markdown_has_all_sections():
    md = render_markdown(
        {"mean": {"faithfulness": 0.8}, "by_category": {"numeric": {"faithfulness": 0.9}}},
        [{"id": "q03", "question": "Q?", "faithfulness": 0.1, "note": "retrieved wrong item"}],
        {"correct": 4, "total": 5},
        {"model": "kr/claude-sonnet-5.5", "n": 25},
    )
    for needle in ["RAGAS baseline", "faithfulness", "numeric", "q03", "4/5", "kr/claude-sonnet-5.5"]:
        assert needle in md
```

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_report.py -v`

- [ ] **Step 3: Implement report and runner**

`eval/__init__.py`: empty file.

`eval/report.py`:
```python
from __future__ import annotations

import pandas as pd


def summarize(df: pd.DataFrame, metric_cols: list[str]) -> dict:
    return {
        "mean": {c: float(df[c].mean()) for c in metric_cols},
        "by_category": {
            cat: {c: float(g[c].mean()) for c in metric_cols} for cat, g in df.groupby("category")
        },
    }


def render_markdown(summary: dict, worst: list[dict], intent_acc: dict, meta: dict) -> str:
    cols = list(summary["mean"])
    lines = [
        "# RAGAS baseline",
        "",
        f"Judge/LLM: `{meta['model']}` · RAGAS-evaluated questions: {meta['n']}",
        "",
        "## Overall means",
        "",
        "| metric | mean |", "|---|---|",
        *[f"| {c} | {v:.3f} |" for c, v in summary["mean"].items()],
        "",
        "## By category",
        "",
        "| category | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
        *[f"| {cat} | " + " | ".join(f"{vals[c]:.3f}" for c in cols) + " |" for cat, vals in summary["by_category"].items()],
        "",
        f"## Intent accuracy (all 30 questions, incl. traps): {intent_acc['correct']}/{intent_acc['total']}",
        "",
        "## Worst cases (for Debug questions)",
        "",
        *[f"- **{w['id']}** (faithfulness {w.get('faithfulness', float('nan')):.2f}): {w['question']}\n  - Why: {w.get('note', 'TO ANALYZE')}" for w in worst],
    ]
    return "\n".join(lines) + "\n"
```
`eval/run_ragas.py`:
```python
"""Run the golden set through the app and score it with RAGAS.
Usage: uv run python -m eval.run_ragas
"""
from __future__ import annotations

import json

import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from eval.report import render_markdown, summarize
from finassist.chains import build_llm, format_context
from finassist.config import ROOT, get_settings
from finassist.runtime import ask, build_app
from finassist.vectorstore import get_embeddings


def main() -> None:
    s = get_settings()
    golden = json.loads((ROOT / "eval" / "golden_set.json").read_text())
    app = build_app(s)

    rows, correct = [], 0
    for g in golden:
        state = ask(app, g["question"], tags=["eval"])
        got = state["output"].intent
        correct += got == g["expected_intent"]
        if g["category"] == "trap":
            continue
        chunks = state.get("retrieved_chunks", [])
        ctx = [f"[{c.chunk_id}] {c.text}" for c in chunks]
        if state.get("price_data"):
            ctx.append(format_context([], state["price_data"]))
        rows.append({
            "id": g["id"], "category": g["category"], "question": g["question"],
            "user_input": g["question"], "response": state["output"].answer,
            "retrieved_contexts": ctx or ["(no context retrieved)"], "reference": g["ground_truth"],
        })

    judge = LangchainLLMWrapper(build_llm(s, s.judge_model))
    emb = LangchainEmbeddingsWrapper(get_embeddings(s))
    ds = EvaluationDataset.from_list([{k: r[k] for k in ("user_input", "response", "retrieved_contexts", "reference")} for r in rows])
    result = evaluate(ds, metrics=[Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall(), ResponseRelevancy()],
                      llm=judge, embeddings=emb)
    scores = result.to_pandas()
    metric_cols = [c for c in scores.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")
                   and pd.api.types.is_numeric_dtype(scores[c])]
    df = pd.concat([pd.DataFrame(rows)[["id", "category", "question"]].reset_index(drop=True),
                    scores[metric_cols].reset_index(drop=True)], axis=1)

    summary = summarize(df, metric_cols)
    faith = next((c for c in metric_cols if "faith" in c), metric_cols[0])
    worst = df.sort_values(faith).head(5).rename(columns={faith: "faithfulness"})[["id", "question", "faithfulness"]].to_dict("records")

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "ragas_baseline.json").write_text(json.dumps(
        {"summary": summary, "per_question": json.loads(df.to_json(orient="records")),
         "intent_accuracy": {"correct": int(correct), "total": len(golden)}}, indent=2))
    (out / "ragas_baseline.md").write_text(render_markdown(
        summary, worst, {"correct": int(correct), "total": len(golden)}, {"model": s.judge_model, "n": len(rows)}))
    print((out / "ragas_baseline.md").read_text())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify unit tests** — `uv run pytest tests/test_report.py -v` → 2 passed.

- [ ] **Step 5: Run the real baseline**

Run: `uv run python -m eval.run_ragas`
Expected: prints the markdown report with four metrics (faithfulness, context precision, context recall, answer relevancy; column names depend on the installed ragas version), by-category table, intent accuracy `x/30`, worst 5. Takes several minutes (about 25 app runs plus judge calls). If RAGAS raises about column names or metric imports, adapt to the installed version (`uv pip list | grep -i ragas`), keeping `tests/test_report.py` unchanged. If NaN appears, keep it in the report (do not drop silently) and note it.

- [ ] **Step 6: Analyse worst cases**

For each of the 5 worst, open its LangSmith trace (Task 12) or `runs`, decide the root cause (wrong chunk retrieved, context too short, answer not cited, ground truth too narrow) and replace `TO ANALYZE` by editing the `note` into `reports/ragas_baseline.md` by hand. Do not edit scores.

- [ ] **Step 7: Commit**

```bash
git add eval reports/ragas_baseline.md reports/ragas_baseline.json tests/test_report.py
git commit -m "feat: RAGAS baseline evaluation and report" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 12: LangSmith tracing (R8)

**Files:**
- Create: `scripts/trace_link.py`, `docs/evidence/.gitkeep`
- Test: manual (needs LangSmith key)

**Interfaces:**
- Consumes: `.env` variables `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`; `ask()` already sets `run_name`, `tags`, `metadata`.

- [ ] **Step 1: Obtain the key (user action)**

Ask the user for a LangSmith API key (smith.langchain.com → Settings → API Keys), then append to `.env` (never print it):
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<key>
LANGSMITH_PROJECT=finassist-w1
```

- [ ] **Step 2: Write the link script**

`scripts/trace_link.py`:
```python
"""Print the URL of the most recent root run in the finassist project."""
import os

from langsmith import Client

from finassist.config import get_settings


def main() -> None:
    get_settings()  # loads .env
    client = Client()
    project = os.getenv("LANGSMITH_PROJECT", "finassist-w1")
    run = next(iter(client.list_runs(project_name=project, is_root=True, limit=1)), None)
    if run is None:
        raise SystemExit(f"no runs in project {project}")
    print(getattr(run, "url", None) or client.read_run(run.id).url)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Generate a traced run and verify**

Run: `uv run finassist ask "What is NVIDIA's current stock price and how do export controls threaten it?"` then `uv run python scripts/trace_link.py`
Expected: a URL. Open it and confirm the tree shows nodes `classify_intent → price_tool → retrieve → generate_answer → format_output`; the `price_tool` node output (state with `price_data`), the `retrieve` node output (list of chunks with ids), LLM calls with prompts, tags `demo`, metadata `app_version`.

- [ ] **Step 4: Save evidence**

Screenshot the trace tree and the retrieve node into `docs/evidence/trace_tree.png` and `docs/evidence/trace_retrieve.png`. Add the URL to `docs/evidence/links.md`.

- [ ] **Step 5: Commit**

```bash
git add scripts/trace_link.py docs/evidence
git commit -m "feat: LangSmith trace link script and evidence" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Streamlit demo UI

**Files:**
- Create: `app/streamlit_app.py`
- Test: manual in browser

**Interfaces:**
- Consumes: `build_app`, `ask`, `render_record`-compatible record via `finassist.cli._record`.

- [ ] **Step 1: Implement**

`app/streamlit_app.py`:
```python
import json
from pathlib import Path

import streamlit as st

from finassist.cli import _record
from finassist.config import ROOT

st.set_page_config(page_title="Finance AI Assistant", layout="wide")
st.title("Finance AI Assistant · AAPL / MSFT / NVDA")

replay = st.sidebar.checkbox("Replay recorded run (offline fallback)", value=False)


@st.cache_resource
def get_app():
    from finassist.runtime import build_app
    return build_app()


question = st.text_input("Question", "What is NVIDIA's current stock price and how do export controls threaten it?")
if st.button("Ask") and question:
    if replay:
        rec = json.loads((ROOT / "runs" / "demo.json").read_text())
    else:
        from finassist.runtime import ask
        rec = _record(ask(get_app(), question, tags=["demo"]))
    out = rec["output"]
    st.caption(f"intent: **{out['intent']}**")
    st.write(out["answer"])
    if rec.get("error"):
        st.warning(rec["error"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Citations")
        st.table([{"source": c["source"], "item": c["item"], "chunk_id": c["chunk_id"], "quote": c["quote"]} for c in out["citations"]])
        st.subheader("Retrieved chunks")
        for r in rec["retrieved"]:
            with st.expander(f"{r['chunk_id']}  (distance {r['score']:.3f})" if r["score"] is not None else r["chunk_id"]):
                st.write(r["text"])
    with c2:
        st.subheader("Structured output (JSON)")
        st.json(out)
```

- [ ] **Step 2: Run and verify**

Run: `uv run streamlit run app/streamlit_app.py`
Expected: page loads; asking the default question shows answer, citations table, retrieved chunks, JSON. Tick "Replay" with `runs/demo.json` present: works with the router stopped.

- [ ] **Step 3: Commit**

```bash
git add app
git commit -m "feat: streamlit demo UI with replay mode" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 14: README, demo documents and final verification (R10, R12)

**Files:**
- Create: `README.md`, `docs/ARCHITECTURE.md`, `docs/KNOWLEDGE_CHECK.md`, `docs/DEMO_SCRIPT.md`, `docs/TEAM_LOG.md`, `docs/NEXT_STEP.md`
- Modify: `docs/superpowers/specs/2026-10-28-finance-rag-assistant-design.md` (§9 tree: replace `scripts/record_demo.py` with note "recording is `finassist ask --record`")

**Interfaces:**
- Consumes: `reports/ragas_baseline.md` numbers and worst cases, LangSmith trace URL, real chunk ids from a live run. Every number in the docs must be copied from those artifacts.

- [ ] **Step 1: README.md** — sections: what it is (2 lines); prerequisites (uv, router, LangSmith key); setup (`uv sync`, copy `.env.example` → `.env`, `uv run python scripts/fetch_filings.py`, `uv run finassist ingest`); run (`uv run finassist ask "..."`, `uv run streamlit run app/streamlit_app.py`); eval (`uv run python -m eval.run_ragas`); tests (`uv run pytest`); repo map (copy the File Structure block); troubleshooting (base URL `/v1` stripping, `auto/*` models broken, SEC User-Agent).

- [ ] **Step 2: docs/ARCHITECTURE.md** — include: the graph diagram from the spec §4; state fields table; node table; data flow for one `both` question using real ids from a live run; **Decisions and rejected alternatives** table with exactly these rows:

| Decision | Chosen | Rejected and why |
|---|---|---|
| Orchestration | LangGraph router with conditional edges | ReAct free agent: unpredictable path, hard to evaluate; linear pipeline: no branching to show |
| Vector DB | Chroma | FAISS: no metadata filter (ticker filtering needed) |
| Embedding | bge-small local | OpenAI embedding: needs another key and sends filings out; Anthropic has no embedding API |
| Citations | Deterministic from `[chunk_id]` markers | LLM-produced citation objects: can hallucinate ids |
| Numbers | `ground_numbers` nulls unsupported values | Trusting LLM extraction: silent wrong EPS/revenue |
| Chunking | By Item, then 1000/150 | Fixed-size over whole filing: mixes risk text with financial tables |
| LLM endpoint | `kr/claude-sonnet-5.5` pinned | `auto/*` combos: upstream rate-limit errors observed |

- [ ] **Step 3: docs/KNOWLEDGE_CHECK.md** — one section per question type, each with a ready 60–90 s answer using real artifacts: **Explain** (walk one question through classify → price/retrieve → generate → format, naming files `graph.py`, `chains.py`); **Predict** (for q21 predict the state after each node, the tool call `get_stock_price("AAPL")`, and which chunk ids appear; verify against the live trace); **Debug** (take the worst case from the report, explain why faithfulness is low using the per-question JSON and trace); **Justify** (pick one trace, state one improvement, link to `NEXT_STEP.md`). Add a "hint questions" subsection with prepared answers for: "Show me where" (file + line for citation extraction in `postprocess.py`, the retrieve node in the trace), "Why this choice" (link to decisions table), "What would fail" (stale price, missing marker, Item parse miss, out-of-scope), "How did you verify" (pytest count, RAGAS report, trace screenshots).

- [ ] **Step 4: docs/DEMO_SCRIPT.md** — 8-minute run sheet: (1) 0:00–1:30 problem and finance context; (2) 1:30–3:30 architecture and data flow using the diagram; (3) 3:30–6:00 live: q21-style `both` question showing citation + JSON + LangSmith trace, plus one out-of-scope refusal; (4) 6:00–8:00 RAGAS baseline, one failure case, next step. Include the checklist of the 6 things to bring (Product, Repository, Architecture, Evaluation, Team evidence, Next step) with the path of each artifact, the fallback plan (Replay checkbox / `--replay runs/demo.json`, `FINASSIST_PRICE_OFFLINE=1`), and the safety checklist (`.env` not on screen, keys masked, browser tabs closed). Add the Q&A clinic template: Context → Attempt → Evidence → Specific question, with 2 filled examples from real issues met during the build.

- [ ] **Step 5: docs/TEAM_LOG.md** — table `date | who | work | evidence (commit/PR/file)`; fill from `git log --format='%ad %an %s' --date=short`; add a "Learned" line and a "Priority fix committed" line.

- [ ] **Step 6: docs/NEXT_STEP.md** — one improvement for week 2: hybrid retrieval (BM25 + vector) with a reranker for numeric questions; state the baseline number it targets (copy the numeric-category context precision/recall from `reports/ragas_baseline.md`) and how success will be measured (same golden set, same metrics).

- [ ] **Step 7: Final verification (R1–R12 checklist)**

Run each and confirm:
```bash
uv run pytest -v                                   # R1,R2,R3,R4,R5,R6,R9 all green
ls reports/ragas_baseline.md                       # R7 exists, has 4 metrics
git ls-files | grep -E '^\.env$' || echo "OK: .env not tracked"     # R10
git grep -nE 'sk-[A-Za-z0-9-]{12,}' -- . ':!docs/superpowers' || echo "OK: no keys in repo"   # R10
uv run finassist ask "x" --replay runs/demo.json   # R11 works offline
ls docs/evidence                                   # R8 screenshots + links.md
grep -c "" docs/DEMO_SCRIPT.md                     # R12 six-item checklist present
```
Expected: every line passes. Fix anything that does not before committing.

- [ ] **Step 8: Commit**

```bash
git add README.md docs
git commit -m "docs: README and demo package (architecture, knowledge check, script, team log, next step)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage (self-review)

| Req | Task |
|---|---|
| R1 graph ≥4 nodes + conditional edges | 7 |
| R2 deterministic tool | 5 |
| R3 chunk by Item, metadata, idempotent | 2, 3, 4 |
| R4 citations to retrieved chunks | 6, 7 |
| R5 schema valid, numbers grounded | 1, 6 |
| R6 golden set 30 | 10 |
| R7 RAGAS baseline 4 metrics | 11 |
| R8 LangSmith trace | 12 |
| R9 out-of-scope refusal | 7 |
| R10 no secrets | 1 (.gitignore, earlier), 14 |
| R11 replay offline | 9, 13 |
| R12 demo doc package | 14 |

Known execution-time dependencies: LangSmith key (Task 12), real SEC data and manual ground truth (Tasks 3, 10), installed `ragas` version may rename metric columns (Task 11).
