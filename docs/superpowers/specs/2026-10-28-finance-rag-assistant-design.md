# Finance AI Chat Assistant: Spec (W1 Workshop)

Ngày: 2026-10-28 · Domain: **Finance** · Trạng thái: chờ duyệt

## 1. Mục tiêu

Chatbot phân tích tài chính trả lời câu hỏi về 3 công ty (AAPL, MSFT, NVDA) dựa trên:
- ngữ cảnh nội bộ: báo cáo 10-K của SEC (phi cấu trúc) qua RAG;
- ngữ cảnh ngoài: dữ liệu giá cổ phiếu (có cấu trúc) qua tool tất định;
- đầu ra có cấu trúc (Pydantic/JSON) và có trích dẫn nguồn.

Exit criteria (từ đề bài): demo chạy được một kịch bản thật có trích dẫn, kèm báo cáo RAGAS baseline.

## 2. Phạm vi

**Trong phạm vi:** 4 năng lực của W1, bộ tài liệu demo/knowledge check, CLI + UI demo nhỏ.
**Ngoài phạm vi (YAGNI):** multi-agent, auth, deploy, tư vấn đầu tư cá nhân hóa, reranker, hybrid search (để dành cho gợi ý tuần 2).

## 3. Quyết định công nghệ

| Hạng mục | Chọn | Lý do / phương án loại |
|---|---|---|
| Ngôn ngữ | Python 3.12 qua `uv` | Máy mặc định 3.14, nhiều thư viện chưa hỗ trợ |
| Orchestration | LangGraph `StateGraph` có nhánh điều kiện | Loại ReAct tự do: khó dự đoán, khó đánh giá |
| LLM | Claude qua router local Omnirouter (`ANTHROPIC_BASE_URL=http://localhost:20128/v1`, model `kr/claude-sonnet-5.5`; `auto/*` đang lỗi rate limit) | Đã kiểm tra gọi thật HTTP 200 |
| Embedding | `BAAI/bge-small-en-v1.5` chạy local (HuggingFace) | Anthropic không có API embedding; không cần key thêm |
| VectorDB | Chroma, lưu tại `data/chroma/` | Có metadata filter, chạy local; loại FAISS vì filter yếu |
| Dữ liệu giá | `yfinance` + cache JSON `data/cache/prices.json` | Demo không phụ thuộc mạng |
| Đánh giá | RAGAS, judge là Claude, embedding local | Theo đề bài |
| Observability | LangSmith (biến môi trường `LANGSMITH_*`) | Theo đề bài |
| Kiểm thử | pytest | |

## 4. Kiến trúc

```
question
   │
classify_intent ──► price | filing | both | out_of_scope
   │
   ├─ price ───────► price_tool ──────────────┐
   ├─ filing ──────► retrieve ────────────────┤
   ├─ both ────────► price_tool ► retrieve ───┤
   └─ out_of_scope ─► refuse ──► END          │
                                      generate_answer
                                              │
                                        format_output ──► FinanceAnswer
```

### 4.1 State (`TypedDict`)
`question: str`, `intent: Intent`, `tickers: list[str]`, `price_data: dict | None`, `retrieved_chunks: list[Chunk]`, `answer: str`, `output: FinanceAnswer | None`, `error: str | None`.

### 4.2 Node
| Node | Việc | Phụ thuộc |
|---|---|---|
| `classify_intent` | LLM structured output → `intent`, `tickers` | LLM |
| `price_tool` | Tool tất định `get_stock_price(ticker)` → giá, vốn hóa, P/E, thời điểm; lỗi mạng thì dùng cache và gắn cờ `stale=true` | yfinance, cache |
| `retrieve` | top-k=5 từ Chroma, filter theo `ticker`; trả chunk kèm metadata | Chroma, embedding |
| `generate_answer` | Prompt chỉ cho phép dùng chunk/giá đã cung cấp; mỗi khẳng định gắn `[chunk_id]` | LLM |
| `format_output` | Ép về `FinanceAnswer` bằng `with_structured_output` | LLM |
| `refuse` | Từ chối lịch sự câu ngoài phạm vi hoặc xin tư vấn đầu tư | không |

### 4.3 Schema đầu ra `FinanceAnswer`
```
ticker: str | None
fiscal_year: int | None
revenue: Money | None            # {value: float, unit: "USD", period: str}
eps: float | None
risk_factors: list[str]
stock_price: PriceInfo | None    # {price, market_cap, pe_ratio, as_of, stale}
answer: str
citations: list[Citation]        # {source, item, chunk_id, quote}
intent: Intent
```
Mọi giá trị số phải truy được về một citation hoặc `stock_price`; trường không có bằng chứng để `null`.

## 5. Dữ liệu và ingestion

- Nguồn: 10-K mới nhất của AAPL, MSFT, NVDA từ SEC EDGAR (HTML), lưu `data/raw/`.
- Parse HTML → text, tách theo **Item** (1, 1A, 7, 7A, 8). Chunk trong từng Item bằng `RecursiveCharacterTextSplitter` (≈1000 ký tự, overlap 150).
- Metadata mỗi chunk: `ticker, fiscal_year, item, chunk_id, source_url`.
- `chunk_id` ổn định (`AAPL-2025-1A-012`) để eval và trích dẫn tái lập được.
- Ingestion idempotent: chạy lại không nhân đôi chunk.
- SEC yêu cầu header `User-Agent` có thông tin liên hệ; cấu hình qua `SEC_USER_AGENT`.

## 6. Đánh giá (RAGAS)

- `eval/golden_set.json`: 30 câu, mỗi câu có `question, ground_truth, ticker, category, expected_intent, expected_chunk_ids?`.
- Phân bố: 10 số liệu (revenue/EPS), 10 risk/MD&A, 5 kết hợp giá + filing, 5 bẫy/ngoài phạm vi.
- Metric: faithfulness, context precision, context recall, answer relevancy. Câu ngoài phạm vi chấm riêng bằng độ chính xác intent (không đưa vào RAGAS).
- Ground truth do người viết tay từ 10-K, không để LLM tự sinh rồi tự chấm.
- Đầu ra: `reports/ragas_baseline.json` + `reports/ragas_baseline.md` (điểm trung bình, điểm theo category, 5 case tệ nhất kèm phân tích nguyên nhân).
- Case thất bại được giữ nguyên trong báo cáo để phục vụ câu hỏi Debug.

## 7. Observability (LangSmith)

- Bật bằng `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=finassist-w1`.
- Mỗi run gắn metadata `intent, tickers, app_version` và tag `demo|eval`.
- Trace phải thấy được: state sau từng node, tool call (input/output), các chunk được retrieve kèm score.
- Script `scripts/trace_link.py` in link trace của run gần nhất.

## 8. Giao diện demo

- CLI: `finassist ask "..."` in JSON cấu trúc + citation.
- UI Streamlit tối giản: ô hỏi, câu trả lời, bảng citation, JSON, link LangSmith.
- Chế độ dự phòng `--replay runs/demo.json` phát lại run ghi sẵn khi API hoặc mạng lỗi.

## 9. Cấu trúc repo

```
pyproject.toml  .env.example  .gitignore  README.md
src/finassist/{config,schemas,ingest,vectorstore,tools,graph,cli}.py
app/streamlit_app.py
eval/{golden_set.json,run_ragas.py}
scripts/{fetch_filings.py,trace_link.py,record_demo.py}
reports/  runs/  data/{raw,chroma,cache}/
docs/{ARCHITECTURE,KNOWLEDGE_CHECK,DEMO_SCRIPT,TEAM_LOG,NEXT_STEP}.md
tests/
```

## 10. Yêu cầu có thể kiểm chứng (traceability)

| ID | Yêu cầu | Kiểm chứng |
|---|---|---|
| R1 | Graph có ≥ 4 node và nhánh điều kiện theo intent | `tests/test_graph_routing.py` (4 intent → 4 đường đi) |
| R2 | Có ≥ 1 tool tất định; cùng input cho cùng output (từ cache) | `tests/test_price_tool.py` |
| R3 | Ingest chia chunk theo Item, metadata đủ, idempotent | `tests/test_ingest.py` |
| R4 | Mọi câu trả lời `filing/both` có ≥ 1 citation trỏ tới chunk đã retrieve | `tests/test_citations.py` |
| R5 | Đầu ra validate được theo `FinanceAnswer`; không có số liệu thiếu bằng chứng | `tests/test_schema.py` |
| R6 | Golden set đủ 30 câu, đúng phân bố | `tests/test_golden_set.py` |
| R7 | Báo cáo RAGAS baseline có đủ 4 metric | `reports/ragas_baseline.md` sinh bởi `run_ragas.py` |
| R8 | Trace LangSmith hiển thị state, tool call, chunk | Kiểm tay + chụp ảnh lưu `docs/evidence/` |
| R9 | Câu ngoài phạm vi/xin tư vấn đầu tư bị từ chối | `tests/test_graph_routing.py` |
| R10 | Không lộ secret: `.env` bị ignore, chỉ có `.env.example` | kiểm `.gitignore` + quét repo |
| R11 | Replay chạy được khi không có mạng/API | `finassist ask --replay` |
| R12 | Bộ tài liệu demo đủ 6 thứ theo đề bài | checklist trong `docs/DEMO_SCRIPT.md` |

## 11. Xử lý lỗi

- yfinance lỗi/mạng mất → dùng cache, gắn `stale=true`, nói rõ trong câu trả lời.
- Retrieval không có chunk đủ liên quan → trả lời "không đủ thông tin trong 10-K", không bịa.
- LLM trả sai schema → thử lại tối đa 2 lần, sau đó trả `error` trong state.
- Ticker ngoài 3 công ty → `out_of_scope`.

## 12. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Chưa có LangSmith key | Cần người dùng cung cấp trước bước R8; phần còn lại làm được trước |
| `ANTHROPIC_BASE_URL` không có key hoặc không hỗ trợ một số tham số | Kiểm tra bằng một lệnh gọi thật ở bước đầu của kế hoạch |
| RAGAS với Claude judge bất ổn (parse lỗi) | Đặt temperature=0, retry, ghi rõ NaN thay vì loại ngầm |
| Parse HTML 10-K sai ranh giới Item | Test với từng công ty, kiểm số chunk theo Item |
| Mạng demo kém | Replay + cache giá |

## 13. Gợi ý cho tuần 2 (`docs/NEXT_STEP.md`)

Một cải tiến ưu tiên: thêm **hybrid search (BM25 + vector) kèm reranker** cho các câu số liệu, vì context precision/recall của câu revenue/EPS thường thấp nhất ở baseline. Con số thực tế sẽ lấy từ báo cáo RAGAS.
