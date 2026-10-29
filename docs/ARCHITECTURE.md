# ARCHITECTURE

Tài liệu mô tả kiến trúc của `finassist` theo đúng code hiện có. Tham chiếu `file:line` đã được kiểm tra với commit hiện tại.

## 1. Sơ đồ graph

LangGraph `StateGraph` với 6 node và 2 điểm rẽ nhánh có điều kiện (`src/finassist/graph.py:120-130`).

```
question
   |
classify_intent --(route, graph.py:66)--> price | filing | both | out_of_scope
   |
   |-- price ---------> price_tool --(after_price, graph.py:78)--> generate_answer
   |-- both ----------> price_tool --(after_price)--> retrieve --> generate_answer
   |-- filing --------> retrieve ----------------------------------> generate_answer
   '-- out_of_scope --> refuse --> END

generate_answer --> format_output --> END   (FinanceAnswer)
```

Mapping của `route`: `price`/`both` -> `price_tool`, `filing` -> `retrieve`, `out_of_scope` -> `refuse` (`graph.py:67`). Sau `price_tool`, chỉ `both` đi tiếp sang `retrieve` (`graph.py:78-79`).

## 2. State (`GraphState`, `graph.py:33-42`)

| Field | Type | Ai ghi | Ghi chú |
|---|---|---|---|
| `question` | `str` | caller (`runtime.ask`, `runtime.py:18-20`) | đầu vào |
| `intent` | `Intent` (`price`/`filing`/`both`/`out_of_scope`) | `classify_intent` | bị ép về `out_of_scope` nếu không có ticker hợp lệ (`graph.py:62`) |
| `tickers` | `list[str]` | `classify_intent` | lọc theo `TICKERS = (AAPL, MSFT, NVDA)` (`config.py:10`, `graph.py:61`) |
| `price_data` | `list[PriceInfo]` | `classify_intent` khởi tạo `[]`, `price_tool` ghi | |
| `retrieved_chunks` | `list[Chunk]` | `classify_intent` khởi tạo `[]`, `retrieve` ghi | mỗi chunk kèm `score` = khoảng cách Chroma (thấp hơn là gần hơn) |
| `answer` | `str` | `generate_answer` / `refuse` | văn bản có marker `[chunk_id]` |
| `output` | `FinanceAnswer` | `format_output` / `refuse` | kết quả cuối |
| `error` | `str | None` | `price_tool`, `generate_answer` | ví dụ `missing_citation` hoặc lỗi giá |
| `short_circuit` | `bool` | `classify_intent` = False; `generate_answer` = True khi trả lời mẫu | báo cho `format_output` bỏ qua LLM structuring (`graph.py:106-107`) |

## 3. Node

| Node | Việc | Code | Phụ thuộc |
|---|---|---|---|
| `classify_intent` | LLM structured output -> `intent`, `tickers`; lọc ticker ngoài phạm vi | `graph.py:59-64`; prompt `chains.py:17-23`, gọi `chains.py:115-116` | LLM |
| `price_tool` | gọi `get_stock_price` cho từng ticker; lỗi thì ghi `error` và đi tiếp | `graph.py:69-76`; tool `tools.py:60-90` | yfinance, cache `data/cache/prices.json` |
| `retrieve` | top-k=5 (mặc định, `config.py:46`) mỗi ticker, filter metadata `ticker` | `graph.py:81-82`; `chains.py:118-119`; `vectorstore.py:41-52` | Chroma, bge-small |
| `generate_answer` | trả lời chỉ từ chunk/giá; bắt buộc có marker hợp lệ, thử tối đa 2 lần | `graph.py:84-99`; prompt `chains.py:25-30` | LLM |
| `format_output` | LLM trích `revenue`/`eps`/`risk_factors`, rồi `ground_numbers`, citation tất định, giá từ tool | `graph.py:101-114`; `chains.py:129-155` | LLM |
| `refuse` | trả văn bản từ chối cố định (Việt/Anh) | `graph.py:116-118`, `REFUSAL_TEXT` `graph.py:13-17` | không |

## 4. Schema đầu ra `FinanceAnswer` (`schemas.py:43-53`)

`ticker`, `fiscal_year`, `revenue: Money | None` (USD millions, đúng như in trong filing), `eps`, `risk_factors`, `stock_price`, `stock_prices`, `answer`, `citations`, `intent`. Hai trường `stock_price` / `stock_prices` luôn lấy từ tool, không do LLM sinh (`graph.py:105`, `graph.py:110-113`).

## 5. Data flow thật của một câu hỏi `both`

Câu hỏi: "What is NVIDIA's current stock price and how do export controls threaten it?" (LangSmith project `finassist-w1`, trace `01a18548-f611-70d1-ad75-5bb267a4a30e`, URL trong `docs/evidence/links.md`; text tree trong `docs/evidence/trace_tree.md`). Tổng thời gian 13.65s.

| Bước | Node (thời gian) | State sau node (từ trace) |
|---|---|---|
| 1 | `classify_intent` (0.30s, 1 lần `ChatAnthropic` 0.29s) | `intent = both`; `tickers = ["NVDA"]` (suy ra từ `price_tool` chỉ có NVDA); `retrieved_chunks = []`, `price_data = []` |
| 2 | `price_tool` (1.67s) | `price_data = [{ticker NVDA, price 215.38, market_cap 5200780722176.0, pe_ratio 27.5422, as_of 2026-10-29T01:31:55+00:00, stale false}]` |
| 3 | `retrieve` (0.74s) | 5 chunk: `NVDA-2026-1A-126`, `NVDA-2026-1A-106`, `NVDA-2026-15-085`, `NVDA-2026-15-025`, `NVDA-2026-15-020` (Item 1A là risk factors, Item 15 là financial statements/notes) |
| 4 | `generate_answer` (5.50s, 1 lần LLM 5.49s) | `answer` mở đầu "NVIDIA (NVDA) was priced at $215.38 as of 2026-10-29 01:31 UTC, with a market cap of about $5.20 trillion and a P/E of 27.54. The price data is not marked stale." rồi phần export controls |
| 5 | `format_output` (5.27s, 1 lần LLM 5.24s) | `FinanceAnswer(intent=both, ...)` + `citations` suy ra từ marker `[chunk_id]` trong `answer` + `stock_price` từ tool |

Nhánh khác, cũng thật (`01a18549-d281-74e8-93cf-cb568132d110`, 0.25s): "Should I buy NVIDIA stock now?" -> `classify_intent` (`intent = out_of_scope`) -> `refuse`. Không có `price_tool`, `retrieve`, `generate_answer`.

Ghi chú: trace tree lưu chuỗi `answer` đã cắt; phần `format_output` không in giá trị cuối trong text export. Muốn xem đầy đủ phải mở URL LangSmith.

Câu trả lời đã ghi sẵn dùng cho replay là `runs/demo.json` (câu `filing`: Apple net sales FY2025 = 416,161 USD millions, citation `AAPL-2025-7-005` và `AAPL-2025-8-023`).

## 6. Quyết định và phương án bị loại

| Decision | Chosen | Rejected and why |
|---|---|---|
| Orchestration | LangGraph router with conditional edges (`graph.py:125-126`) | ReAct free agent: unpredictable path, hard to evaluate; linear pipeline: no branching to show |
| Vector DB | Chroma | FAISS: no metadata filter (ticker filtering needed, `vectorstore.py:44`) |
| Embedding | `bge-small-en-v1.5` local (`vectorstore.py:12-18`) | OpenAI embedding: needs another key and sends filings out; Anthropic has no embedding API |
| Citations | Deterministic from `[chunk_id]` markers (`postprocess.py:7-22`) | LLM-produced citation objects: can hallucinate ids |
| Numbers | `ground_numbers` nulls unsupported values (`postprocess.py:36-44`) | Trusting LLM extraction: silent wrong EPS/revenue |
| Chunking | By Item, then 1000/150 (`ingest.py:19`, `ingest.py:65-74`) | Fixed-size over whole filing: mixes risk text with financial tables |
| LLM endpoint | `kr/claude-sonnet-5.5` pinned (`config.py:39`) | `auto/*` combos: upstream rate-limit errors observed |
| Items kept | 1, 1A, 7, 7A, 8, 15 (`ingest.py:13`); Item 15 added | Only 1/1A/7/7A/8 (original plan): NVIDIA's 10-K has the financial statements in Item 15 and Item 8 is a stub; after ingest NVDA has 1 chunk in Item 8 vs 156 in Item 15 |
| Running page headers | Merge consecutive repeats of the same Item (`ingest.py:53-54`) | Treating each repeated header as a new section: MSFT repeats "Item 8" in the page header and would split one section into many |
| Short-circuit | Explicit `short_circuit` flag in state (`graph.py:42`, `91-99`, `106-107`) | Detecting fixed answers by `startswith` on the text: brittle (replaced in commit `3603e20`) |
| Investment-advice refusal | Classifier decides `out_of_scope` (`chains.py:22`); no keyword rule | A hard-coded keyword rule: not implemented, the refusal rests on the classifier prompt. Trade-off: if the classifier mislabels advice as `price`, nothing else stops it (see KNOWLEDGE_CHECK, "What would fail") |
| Judge model | Same `kr/claude-sonnet-5.5` for app and judge | Separate judge model: not set up in this build (`FINASSIST_JUDGE_MODEL` defaults to the same model, `config.py:40`). Cost: self-grading likely inflates faithfulness (disclosed in `reports/ragas_baseline.md` Caveats) |
| Missing citation | One retry, then fixed `INSUFFICIENT_TEXT` + `error=missing_citation` (`graph.py:95-99`) | Returning an uncited answer: breaks the "every claim has a source" rule |

## 7. Giới hạn đã biết

- Retrieval top-5 theo vector similarity bỏ sót chunk bảng số liệu (xem `reports/ragas_baseline.md`, "Common pattern": 5/5 worst cases không retrieve golden chunk).
- Chunk bảng mất tiêu đề cột; ví dụ `MSFT-2026-8-000` bắt đầu bằng "ITEM 8. FINANCIAL STATE\nMENTS" và các số nằm trên từng dòng riêng.
- Chi tiết kế hoạch cải thiện: `docs/NEXT_STEP.md`.
