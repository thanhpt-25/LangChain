# Finance AI Assistant (finassist) - W1

Trợ lý AI tài chính cho 3 công ty **AAPL, MSFT, NVDA**: trả lời câu hỏi từ báo cáo 10-K (RAG, có trích dẫn `chunk_id`) và giá cổ phiếu (tool tất định), đầu ra JSON có cấu trúc. Câu hỏi ngoài phạm vi hoặc xin tư vấn đầu tư bị từ chối.

A LangGraph assistant that routes a question to a price tool, a Chroma retriever over 10-K filings, or a refusal, then returns a validated `FinanceAnswer` with citations. Built for the W1 workshop demo (domain AI chat assistant).

## Prerequisites / Yêu cầu

- `uv` (project is pinned to Python 3.12, see `pyproject.toml:4`; do not use the system Python).
- An Anthropic-compatible LLM endpoint. The project was built against a local router (`ANTHROPIC_BASE_URL=http://localhost:20128/v1`, model `kr/claude-sonnet-5.5`).
- A LangSmith API key (optional, only for traces).
- Network once for: SEC EDGAR download, the `BAAI/bge-small-en-v1.5` embedding model download, `yfinance` prices.

## Setup

```bash
uv sync
cp .env.example .env            # then edit .env: router key, LangSmith key, SEC_USER_AGENT
uv run python scripts/fetch_filings.py   # downloads latest 10-K of AAPL/MSFT/NVDA into data/raw/
uv run finassist ingest                  # chunk by Item, embed locally, store in data/chroma/
```

`.env` and `data/` are git-ignored (`.gitignore`). Never commit them.

## Run

```bash
uv run finassist ask "What is NVIDIA's current stock price and how do export controls threaten it?"
uv run finassist ask "..." --record runs/demo.json     # run live and save the result
uv run finassist ask x --replay runs/demo.json         # offline: replay the saved run (question text is ignored)
uv run streamlit run app/streamlit_app.py              # UI; sidebar checkbox "Replay recorded run" = offline fallback
FINASSIST_PRICE_OFFLINE=1 uv run finassist ask "..."   # price tool uses cache only (data/cache/prices.json)
```

## Evaluation (RAGAS)

```bash
uv run python -m eval.run_ragas     # about 14 minutes (814 s in the baseline run); OVERWRITES reports/ragas_baseline.{json,md} (copy the hand-annotated .md first)
```

Baseline (copied from `reports/ragas_baseline.md`): faithfulness 0.939, context precision 0.309, context recall 0.387, answer relevancy 0.687; intent accuracy 30/30; numeric sanity 7/10. Read the "Caveats" section there before quoting these numbers (the judge model is the same model as the app).

Known limits: the embedding model is English-only, so non-English questions are retrieved with an English `search_query` rewritten by the classifier (English questions are unchanged; table questions in Vietnamese can still miss); and `ground_numbers` checks that a figure appears in a retrieved chunk, not what it means (baseline q10/q20 report NVIDIA "total revenue $139,297" from a headerless segment table, true value 215,938). See `docs/ARCHITECTURE.md` section 7.

## Tests

```bash
uv run pytest -q                                       # 145 passed, 12 skipped (integration tests skipped)
RUN_INTEGRATION=1 uv run pytest tests/test_llm_smoke.py -q   # needs the router up and a populated data/chroma
```

## Documents for the demo / Tài liệu demo

| File | Content |
|---|---|
| `docs/ARCHITECTURE.md` | graph, state, nodes, a real `both` data flow, decisions and rejected alternatives |
| `docs/KNOWLEDGE_CHECK.md` | ready answers for Explain / Predict / Debug / Justify and hint questions |
| `docs/DEMO_SCRIPT.md` | 8-minute run sheet, 6-item checklist, fallback and safety checklists, Q&A clinic |
| `docs/TEAM_LOG.md` | who did what, with commits |
| `docs/NEXT_STEP.md` | the one improvement planned for week 2 |
| `docs/evidence/` | LangSmith trace links and text trace trees |
| `docs/superpowers/specs/`, `docs/superpowers/plans/` | design spec and implementation plan |

## Repo map

```
pyproject.toml  uv.lock  .env.example  .gitignore  .python-version
src/finassist/
  config.py       Settings, normalize_base_url, TICKERS
  schemas.py      pydantic models (Chunk, FinanceAnswer, PriceInfo, Intent...)
  ingest.py       html_to_text, split_items, chunk_filing, load_filing
  vectorstore.py  get_embeddings, FilingStore (Chroma, filter by ticker)
  tools.py        get_stock_price (deterministic, cache fallback, stale flag)
  postprocess.py  extract_citations, ground_numbers
  graph.py        GraphState, Deps, build_graph (6 nodes)
  chains.py       real LLM implementations of Deps (classify, generate, structure)
  runtime.py      build_app, ask
  cli.py          finassist ingest | ask [--record|--replay]
app/streamlit_app.py
scripts/          fetch_filings.py  find_in_filing.py  trace_link.py  trace_evidence.py
eval/             golden_set.json  run_ragas.py  report.py  _ragas_compat.py
reports/          ragas_baseline.json  ragas_baseline.md
runs/             demo.json  (recorded answer for offline replay)
docs/  tests/
data/             (git-ignored) raw/ chroma/ cache/
```

## Troubleshooting

- **`ANTHROPIC_BASE_URL` ends with `/v1`** (curl style). The Anthropic SDK appends `/v1/messages` itself, so `normalize_base_url` strips a trailing `/v1` (`src/finassist/config.py:13-15`).
- **`auto/*` router models fail** with upstream rate-limit errors; the model is pinned to `kr/claude-sonnet-5.5` (`config.py:39`).
- **Shell variables beat `.env`**: `get_settings` uses `load_dotenv(..., override=True)` (`config.py:34`) so the project `.env` wins over `ANTHROPIC_*` already exported in your shell. Unit tests neutralise this in `tests/conftest.py`.
- **SEC rejects the download**: set `SEC_USER_AGENT="Your Name you@email"` in `.env` (SEC requires contact info; `config.py:45`).
- **`ImportError` for `ChatVertexAI` when importing ragas**: ragas 0.4.3 expects modules that `langchain-community` no longer ships; `eval/_ragas_compat.py` registers inert stubs. Import it before ragas (already done in `eval/run_ragas.py:11`).
- **LLM/router down during a demo**: `uv run finassist ask x --replay runs/demo.json`, or tick "Replay recorded run" in Streamlit.
- **Price fetch fails**: the tool falls back to `data/cache/prices.json` and sets `stale=true` (`src/finassist/tools.py:83-85`).
