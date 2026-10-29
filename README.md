# FinAssist (`finassist`)

A LangGraph assistant for AAPL, MSFT and NVDA: it routes a question to a stock price tool, a Chroma retriever over 10-K filings, or a refusal, and returns a validated `FinanceAnswer` with citations.

FinAssist là trợ lý AI tài chính cho ba công ty Apple (AAPL), Microsoft (MSFT) và NVIDIA (NVDA). Câu hỏi về nội dung báo cáo 10-K được trả lời bằng RAG trên kho Chroma cục bộ, kèm trích dẫn `chunk_id`. Câu hỏi về giá, vốn hóa và P/E được trả lời bằng một công cụ tất định gọi yfinance. Kết quả là một đối tượng JSON có cấu trúc (`FinanceAnswer`). Câu hỏi ngoài phạm vi, kể cả xin tư vấn đầu tư, bị từ chối. Dự án được làm cho buổi demo W1 (miền: trợ lý chat AI).

## Kiến trúc tóm tắt

Một câu hỏi đi vào `classify_intent`, node này chọn một trong bốn đường đi. Sơ đồ đầy đủ, state, từng node, data flow thật, các kiểu sự cố và bảng quyết định nằm trong [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```mermaid
flowchart TD
    Start(["question"]) --> Classify["classify_intent"]
    Classify -->|"price"| Price["price_tool"]
    Classify -->|"both"| Price
    Classify -->|"filing"| Retrieve["retrieve"]
    Classify -->|"out_of_scope"| Refuse["refuse"]

    Price -->|"intent là both"| Retrieve
    Price -->|"intent là price"| Generate["generate_answer"]
    Retrieve --> Generate
    Generate --> Format["format_output"]

    Format --> End(["FinanceAnswer"])
    Refuse --> End
```

LLM hiểu câu hỏi, viết câu trả lời và trích số liệu. Mã tất định tạo citation, lấy giá từ công cụ và loại các con số không có trong chunk đã retrieve (`ground_numbers`).

## Yêu cầu

- `uv`. Dự án ghim Python 3.12 (`pyproject.toml:4`), đừng dùng Python của hệ thống.
- Một endpoint LLM tương thích Anthropic. Dự án được xây với router cục bộ (`ANTHROPIC_BASE_URL=http://localhost:20128/v1`, model `kr/claude-sonnet-5.5`).
- Khóa LangSmith (tùy chọn, chỉ cần để xem trace).
- Cần mạng một lần để tải filing từ SEC EDGAR, tải model embedding `BAAI/bge-small-en-v1.5` và lấy giá từ `yfinance`.

## Cài đặt

```bash
uv sync
cp .env.example .env            # then edit .env: router key, LangSmith key, SEC_USER_AGENT
uv run python scripts/fetch_filings.py   # downloads latest 10-K of AAPL/MSFT/NVDA into data/raw/
uv run finassist ingest                  # chunk by Item, embed locally, store in data/chroma/
```

`.env` và `data/` nằm trong `.gitignore`. Không commit hai thứ này.

## Chạy

```bash
uv run finassist ask "What is NVIDIA's current stock price and how do export controls threaten it?"
uv run finassist ask "..." --record runs/demo.json     # run live and save the result
uv run finassist ask x --replay runs/demo.json         # offline: replay the saved run (question text is ignored)
uv run streamlit run app/streamlit_app.py              # UI; sidebar checkbox "Replay recorded run" = offline fallback
FINASSIST_PRICE_OFFLINE=1 uv run finassist ask "..."   # price tool uses cache only (data/cache/prices.json)
```

Dùng câu hỏi tiếng Anh làm chính, vì model embedding chỉ hỗ trợ tiếng Anh. Câu tiếng Việt vẫn chạy nhờ `search_query` tiếng Anh do bộ phân loại viết lại (xem giới hạn bên dưới).

## Đánh giá (RAGAS)

```bash
uv run python -m eval.run_ragas
```

Lệnh này mất khoảng 14 phút (814 giây ở lần chạy baseline) và ghi đè `reports/ragas_baseline.{json,md}`, kể cả các ghi chú viết tay. Hãy sao lưu `reports/ragas_baseline.md` trước khi chạy.

Baseline (lấy từ `reports/ragas_baseline.md`): faithfulness 0.939, context precision 0.309, context recall 0.387, answer relevancy 0.687; intent accuracy 30/30; numeric sanity 7/10. Hãy đọc mục "Caveats" trong báo cáo đó trước khi trích số, vì judge dùng cùng model với ứng dụng.

## Kiểm thử

```bash
uv run pytest -q                                       # 145 passed, 12 skipped (integration tests skipped)
RUN_INTEGRATION=1 uv run pytest tests/test_llm_smoke.py -q   # needs the router up and a populated data/chroma
```

12 test bị bỏ qua là các integration test cần router và `data/chroma`.

## Giới hạn đã biết

Danh sách đầy đủ nằm ở [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) mục 8 (Giới hạn đã biết). Hai giới hạn cần biết trước khi trích số liệu:

- Model embedding chỉ có tiếng Anh. Câu hỏi không phải tiếng Anh được retrieve bằng `search_query` tiếng Anh do bộ phân loại viết lại, còn câu tiếng Anh giữ nguyên. Câu hỏi tiếng Việt về bảng số liệu vẫn có thể trượt chunk.
- `ground_numbers` chỉ kiểm tra một con số có xuất hiện trong chunk đã retrieve, không kiểm tra nghĩa của nó. Ở baseline, q10 và q20 trả "total revenue $139,297" cho NVIDIA từ một bảng segment không có tiêu đề cột, trong khi giá trị đúng là 215,938.

## Bản đồ repo

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

## Bản đồ tài liệu

| File | Nội dung |
|---|---|
| [`docs/README.md`](docs/README.md) | mục lục tài liệu, đọc theo vai trò |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | thành phần, control flow, data flow, sự cố, quyết định kiến trúc, giới hạn |
| [`docs/KNOWLEDGE_CHECK.md`](docs/KNOWLEDGE_CHECK.md) | câu trả lời sẵn cho Giải thích, Dự đoán, Gỡ lỗi, Bảo vệ lựa chọn và các câu hỏi gợi ý |
| [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) | kịch bản demo 8 phút, danh sách 6 thứ cần chuẩn bị, phương án dự phòng, kiểm tra an toàn, mẫu hỏi đáp |
| [`docs/TEAM_LOG.md`](docs/TEAM_LOG.md) | ai đã làm gì, kèm commit |
| [`docs/NEXT_STEP.md`](docs/NEXT_STEP.md) | cải tiến dự kiến cho tuần 2 |
| [`docs/evidence/`](docs/evidence/) | link trace LangSmith và bản text của cây trace |
| `docs/superpowers/specs/`, `docs/superpowers/plans/` | spec thiết kế và kế hoạch triển khai |

## Xử lý sự cố

- `ANTHROPIC_BASE_URL` kết thúc bằng `/v1` (kiểu curl). SDK Anthropic tự thêm `/v1/messages`, nên `normalize_base_url` cắt phần `/v1` ở cuối (`src/finassist/config.py:13-15`).
- Các model `auto/*` của router báo lỗi giới hạn tần suất từ upstream, nên model được ghim là `kr/claude-sonnet-5.5` (`src/finassist/config.py:39`).
- Biến shell thắng `.env` theo mặc định của `python-dotenv`. `get_settings` gọi `load_dotenv(..., override=True)` (`src/finassist/config.py:34`) để `.env` của dự án thắng các biến `ANTHROPIC_*` đã export sẵn. Unit test vô hiệu hóa việc nạp `.env` trong `tests/conftest.py`.
- SEC từ chối tải về: đặt `SEC_USER_AGENT="Your Name you@email"` trong `.env`. SEC yêu cầu thông tin liên hệ (`src/finassist/config.py:45`).
- `ImportError` cho `ChatVertexAI` khi import ragas: ragas 0.4.3 cần các module mà `langchain-community` không còn cung cấp. `eval/_ragas_compat.py` đăng ký module giả rỗng và phải được import trước ragas (đã làm ở `eval/run_ragas.py:11`).
- Router LLM lỗi lúc demo: chạy `uv run finassist ask x --replay runs/demo.json`, hoặc tick "Replay recorded run" trong Streamlit.
- Lấy giá thất bại: công cụ dùng `data/cache/prices.json` và đặt `stale=true` (`src/finassist/tools.py:83-85`).
