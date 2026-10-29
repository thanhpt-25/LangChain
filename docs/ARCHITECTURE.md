# Kiến trúc hệ thống FinAssist

Tài liệu mô tả kiến trúc của `finassist` theo đúng code hiện có. Tham chiếu `file:line` đã được kiểm tra với commit hiện tại. Nếu mới làm quen với dự án, hãy đọc mục 1, 3, 5 và 7 theo thứ tự: chúng trả lời lần lượt hệ thống đi đâu, mỗi phần làm gì, một câu hỏi thật được xử lý ra sao và giới hạn hiện tại là gì.

## 1. Sơ đồ luồng xử lý

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

## 2. Trạng thái (`GraphState`, `graph.py:33-42`)

| Field | Type | Ai ghi | Ghi chú |
|---|---|---|---|
| `question` | `str` | caller (`runtime.ask`, `runtime.py:18-20`) | đầu vào |
| `intent` | `Intent` (`price`/`filing`/`both`/`out_of_scope`) | `classify_intent` | bị ép về `out_of_scope` nếu không có ticker hợp lệ (`graph.py:62`) |
| `search_query` | `str` | `classify_intent` | câu dùng để retrieve: bản viết lại tiếng Anh khi `language = other` và có `search_query`, ngược lại là câu hỏi gốc; `generate_answer` luôn nhận câu hỏi gốc |
| `tickers` | `list[str]` | `classify_intent` | lọc theo `TICKERS = (AAPL, MSFT, NVDA)` (`config.py:10`, `graph.py:61`) |
| `price_data` | `list[PriceInfo]` | `classify_intent` khởi tạo `[]`, `price_tool` ghi | |
| `retrieved_chunks` | `list[Chunk]` | `classify_intent` khởi tạo `[]`, `retrieve` ghi | mỗi chunk kèm `score` = khoảng cách Chroma (thấp hơn là gần hơn) |
| `answer` | `str` | `generate_answer` / `refuse` | văn bản có marker `[chunk_id]` |
| `output` | `FinanceAnswer` | `format_output` / `refuse` | kết quả cuối |
| `error` | `str | None` | `price_tool`, `generate_answer` | ví dụ `missing_citation` hoặc lỗi giá |
| `short_circuit` | `bool` | `classify_intent` = False; `generate_answer` = True khi trả lời mẫu | báo cho `format_output` bỏ qua LLM structuring (`graph.py:106-107`) |

## 3. Các node xử lý

| Node | Việc | Code | Phụ thuộc |
|---|---|---|---|
| `classify_intent` | LLM structured output -> `intent`, `tickers`; lọc ticker ngoài phạm vi | `graph.py:59-64`; prompt `chains.py:17-23`, gọi `chains.py:115-116` | LLM |
| `price_tool` | gọi `get_stock_price` cho từng ticker; lỗi thì ghi `error` và đi tiếp | `graph.py:69-76`; tool `tools.py:60-90` | yfinance, cache `data/cache/prices.json` |
| `retrieve` | top-k=5 (mặc định, `config.py:46`) mỗi ticker, filter metadata `ticker` | `graph.py:81-82`; `chains.py:118-119`; `vectorstore.py:41-52` | Chroma, bge-small |
| `generate_answer` | trả lời chỉ từ chunk/giá; bắt buộc có marker hợp lệ, thử tối đa 2 lần | `graph.py:84-99`; prompt `chains.py:25-30` | LLM |
| `format_output` | LLM trích `revenue`/`eps`/`risk_factors`, rồi `ground_numbers`, citation tất định, giá từ tool | `graph.py:101-114`; `chains.py:129-155` | LLM |
| `refuse` | trả văn bản từ chối cố định (Việt/Anh) | `graph.py:116-118`, `REFUSAL_TEXT` `graph.py:13-17` | không |

## 4. Lược đồ đầu ra `FinanceAnswer` (`schemas.py:43-53`)

`ticker`, `fiscal_year`, `revenue: Money | None` (USD millions, đúng như in trong filing), `eps`, `risk_factors`, `stock_price`, `stock_prices`, `answer`, `citations`, `intent`. Hai trường `stock_price` / `stock_prices` luôn lấy từ tool, không do LLM sinh (`graph.py:105`, `graph.py:110-113`).

## 5. Luồng dữ liệu thực tế của một câu hỏi `both`

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

## 6. Quyết định thiết kế và phương án bị loại

| Quyết định | Phương án chọn | Phương án không chọn và lý do |
|---|---|---|
| Điều phối | Router LangGraph có cạnh điều kiện (`graph.py:125-126`) | ReAct tự do: đường đi khó dự đoán và khó đánh giá; pipeline tuyến tính: không thể hiện được nhánh xử lý. |
| Kho vector | Chroma | FAISS (LangChain) có tham số `filter=` nhưng không có pre-filter gốc: lọc theo metadata chạy sau tìm kiếm ANN trên `fetch_k` ứng viên, nên lọc theo ticker (`vectorstore.py:44`) có thể trả ít hơn k kết quả hoặc bỏ sót chunk đúng; Chroma lọc ngay trong truy vấn. |
| Embedding | `bge-small-en-v1.5` chạy cục bộ (`vectorstore.py:12-18`) | OpenAI Embedding cần thêm khóa và gửi hồ sơ ra ngoài; Anthropic không có API embedding. |
| Trích dẫn | Tất định từ marker `[chunk_id]` (`postprocess.py:7-22`) | Đối tượng citation do LLM sinh có thể bịa ID. |
| Số liệu | `ground_numbers` đặt `null` cho giá trị không được hỗ trợ (`postprocess.py:36-44`) | Tin vào LLM khi trích số có thể sai EPS/doanh thu một cách âm thầm. |
| Chia đoạn | Theo Item, sau đó 1.000/150 (`ingest.py:19`, `ingest.py:65-74`) | Chia cố định toàn bộ filing sẽ trộn văn bản rủi ro với bảng tài chính. |
| Endpoint LLM | Ghim `kr/claude-sonnet-5.5` (`config.py:39`) | Đã quan sát lỗi rate-limit từ upstream với các tổ hợp `auto/*`. |
| Item được giữ | 1, 1A, 7, 7A, 8, 15 (`ingest.py:13`); đã thêm Item 15 | Chỉ giữ 1/1A/7/7A/8 theo kế hoạch cũ: 10-K của NVIDIA để báo cáo tài chính ở Item 15, còn Item 8 chỉ là stub; sau ingest, NVDA có 1 chunk ở Item 8 so với 156 ở Item 15. |
| Running header | Gộp các lần lặp liên tiếp của cùng Item (`ingest.py:53-54`) | Xem mỗi header lặp là một phần mới sẽ làm MSFT bị chia Item 8 thành nhiều phần do header trang lặp lại. |
| Trả lời cố định | Cờ `short_circuit` rõ ràng trong state (`graph.py:42`, `91-99`, `106-107`) | Nhận diện bằng `startswith` trên văn bản dễ vỡ (đã thay trong commit `3603e20`). |
| Từ chối tư vấn đầu tư | Classifier quyết định `out_of_scope` (`chains.py:22`), không dùng luật từ khóa | Luật từ khóa cứng chưa được triển khai; nếu classifier gán nhầm lời khuyên là `price`, không còn lớp chặn nào khác (xem KNOWLEDGE_CHECK, "Điều gì có thể hỏng"). |
| Model chấm điểm | Cùng `kr/claude-sonnet-5.5` cho app và judge | Chưa thiết lập judge độc lập (`FINASSIST_JUDGE_MODEL` mặc định cùng model, `config.py:40`); tự chấm có thể làm faithfulness cao hơn thực tế, như đã nêu trong Caveats của `reports/ragas_baseline.md`. |
| Thiếu trích dẫn | Thử lại một lần, sau đó trả `INSUFFICIENT_TEXT` và `error=missing_citation` (`graph.py:95-99`) | Trả câu không có nguồn sẽ phá vỡ nguyên tắc mỗi khẳng định phải có nguồn. |

## 7. Giới hạn đã biết và cách đọc chúng

- Retrieval top-5 theo vector similarity bỏ sót chunk bảng số liệu (xem `reports/ragas_baseline.md`, "Common pattern": 5/5 worst cases không retrieve golden chunk).
- Chunk bảng mất tiêu đề cột; ví dụ `MSFT-2026-8-000` bắt đầu bằng "ITEM 8. FINANCIAL STATE\nMENTS" và các số nằm trên từng dòng riêng.
- Câu hỏi tiếng Việt (hoặc ngôn ngữ khác tiếng Anh): embedding `bge-small-en-v1.5` chỉ dành cho tiếng Anh nên câu hỏi gốc retrieve kém (đo thực tế với câu hỏi tiếng Việt về rủi ro kiểm soát xuất khẩu của NVIDIA: không có chunk Item 1A nào, distance 0.71-0.74, trong khi câu tiếng Anh tương đương đạt 0.27). Giảm thiểu: classifier trả thêm `language` và `search_query` (bản viết lại tiếng Anh), `retrieve` dùng `search_query` còn câu trả lời vẫn theo ngôn ngữ của người hỏi; câu tiếng Anh không đổi. Sau sửa, câu đó đạt distance 0.27-0.47 với 5/5 chunk Item 1A. Giới hạn còn lại: chất lượng phụ thuộc bản viết lại của classifier và các điểm yếu retrieval sẵn có (ví dụ "Lợi nhuận ròng của Microsoft" vẫn không lấy được chunk Item 8 chứa net income). Embedding đa ngôn ngữ là hướng dài hạn.
- Grounding chỉ kiểm tra văn bản, không kiểm tra ý nghĩa: `ground_numbers` chỉ xác nhận chuỗi số có trong một chunk đã retrieve. Trong baseline, q10 và q20 (NVDA) trả "total revenue $139,297" từ chunk `NVDA-2026-7-029` (bảng segment không có tiêu đề cột: 130,141 + 9,156 = 139,297), trong khi doanh thu FY2026 thật là 215,938 (q07 trả đúng). Trường `revenue` = 139297 vẫn qua `ground_numbers`, và judge vẫn chấm faithfulness 1.0 (q20) và 0.941 (q10) vì con số có trong context. Hệ thống không phải lúc nào cũng hedge khi retrieval lỗi.
- `MARKER_RE` (`postprocess.py:7`) chỉ nhận một chunk id trong mỗi cặp ngoặc vuông: nếu LLM viết `[A, B]` thì không trích được citation nào, dẫn tới thử lại rồi `INSUFFICIENT_TEXT`.
- Chi tiết kế hoạch cải thiện: `docs/NEXT_STEP.md`.
