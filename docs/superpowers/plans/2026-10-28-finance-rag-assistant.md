# Kế hoạch triển khai trợ lý AI tài chính

Ngày lập kế hoạch: 2026-10-28
Tài liệu thiết kế liên quan: [`../specs/2026-10-28-finance-rag-assistant-design.md`](../specs/2026-10-28-finance-rag-assistant-design.md)

## 1. Mục đích của tài liệu

Đây là lộ trình để một người phát triển có thể hiểu, chạy lại và kiểm chứng dự án `finassist`. Hệ thống trả lời câu hỏi về báo cáo 10-K và dữ liệu thị trường của Apple (AAPL), Microsoft (MSFT) và NVIDIA (NVDA). Mục tiêu không phải là đưa ra lời khuyên đầu tư; mục tiêu là trả lời có nguồn, biết từ chối khi nằm ngoài phạm vi và công khai giới hạn của kết quả.

Kế hoạch ban đầu đã được triển khai. Vì vậy, tài liệu này mô tả **trình tự xây dựng và các tiêu chí kiểm chứng** thay vì sao chép toàn bộ mã nguồn của từng bước. Mã nguồn hiện tại và các kiểm thử là chuẩn để xác nhận hành vi thực tế.

## 2. Kết quả mong đợi

1. Phân loại câu hỏi thành giá thị trường (`price`), báo cáo 10-K (`filing`), kết hợp hai loại (`both`) hoặc ngoài phạm vi (`out_of_scope`).
2. Lấy giá, vốn hóa và P/E bằng một công cụ tất định; khi mất mạng, sử dụng bản lưu và đánh dấu dữ liệu cũ.
3. Tìm các đoạn liên quan trong 10-K theo mã cổ phiếu, tạo câu trả lời chỉ từ dữ liệu đã lấy và đính kèm trích dẫn kiểm chứng được.
4. Trả về `FinanceAnswer` có cấu trúc, trong đó số liệu không có bằng chứng phải là `null`.
5. Có CLI, giao diện Streamlit, chế độ phát lại không cần mạng, trace LangSmith và báo cáo đánh giá RAGAS.

## 3. Giới hạn và nguyên tắc bắt buộc

| Chủ đề | Quy ước |
|---|---|
| Phiên bản chạy | Python 3.12 qua `uv`; không dùng Python hệ thống 3.14. |
| Phạm vi dữ liệu | Chỉ AAPL, MSFT và NVDA; nguồn hồ sơ là SEC EDGAR 10-K. |
| Mô hình | Router cục bộ với `kr/claude-sonnet-5.5`; không dùng `auto/*` do đã gặp lỗi giới hạn tần suất. |
| Embedding | `BAAI/bge-small-en-v1.5` chạy cục bộ, không cần thêm khóa API. |
| Dữ liệu giá | `yfinance` với bản lưu `data/cache/prices.json`. |
| Bảo mật | Không commit hay in `.env`, khóa API hoặc dữ liệu nhạy cảm. |
| Đơn vị doanh thu | Triệu USD, đúng như cách trình bày trong 10-K. |
| Kiểm thử | Unit test không gọi mạng hoặc LLM. Kiểm thử tích hợp chỉ chạy khi có `RUN_INTEGRATION=1`. |

## 4. Bức tranh kiến trúc

```text
Câu hỏi
  │
  ├─ classify_intent
  │     ├─ price          → price_tool → generate_answer → format_output
  │     ├─ filing         → retrieve   → generate_answer → format_output
  │     ├─ both           → price_tool → retrieve → generate_answer → format_output
  │     └─ out_of_scope   → refuse → kết thúc
  │
  └─ Kết quả: FinanceAnswer
```

`LangGraph StateGraph` điều phối các nhánh trên. Những thao tác có tác động bên ngoài—gọi LLM, kho vector và dữ liệu giá—được truyền qua `Deps`. Nhờ vậy, logic đồ thị có thể kiểm thử bằng fake thay vì phụ thuộc mạng. LLM không được tự tạo đối tượng citation hoặc giá cổ phiếu: citation được trích từ marker `[chunk_id]`, còn giá được lấy từ công cụ.

Giải thích sâu hơn về state, node, luồng trace thật và các phương án bị loại nằm trong [`docs/ARCHITECTURE.md`](../../ARCHITECTURE.md).

## 5. Lộ trình triển khai

### Giai đoạn 1 — Nền tảng dự án

**Mục tiêu:** tạo môi trường có thể chạy nhất quán và các kiểu dữ liệu làm hợp đồng giữa các thành phần.

- Khai báo dependency trong `pyproject.toml`, ghim Python 3.12 và lệnh `finassist`.
- Tạo `config.py`: đọc `.env`, chuẩn hóa `ANTHROPIC_BASE_URL`, định nghĩa ba ticker và các thư mục dữ liệu.
- Tạo các schema Pydantic trong `schemas.py`: `Chunk`, `Citation`, `Money`, `PriceInfo`, `FinanceAnswer`, `IntentResult`, `LLMExtraction`.
- Viết unit test cho cấu hình và schema; smoke test router phải được đánh dấu `integration`.

**Điều cần chú ý:** URL cấu hình có thể kết thúc bằng `/v1`, nhưng Anthropic SDK tự thêm đường dẫn API. `normalize_base_url` phải loại `/v1` ở cuối để tránh lỗi 404.

**Kiểm chứng:** `uv run pytest tests/test_config.py tests/test_schema.py -v`. Khi router sẵn sàng: `RUN_INTEGRATION=1 uv run pytest tests/test_llm_smoke.py -v`.

### Giai đoạn 2 — Chuẩn bị dữ liệu 10-K

**Mục tiêu:** biến báo cáo HTML thô thành các đoạn văn có định danh ổn định.

- Tải 10-K mới nhất của từng công ty về `data/raw/`; SEC yêu cầu `SEC_USER_AGENT` có thông tin liên hệ thật.
- Chuyển HTML thành văn bản, chia trước theo Item 1, 1A, 7, 7A, 8 và 15, sau đó chia đoạn khoảng 1.000 ký tự với phần chồng 150 ký tự.
- Gắn metadata `ticker`, năm tài chính, Item, URL nguồn và `chunk_id` dạng `AAPL-2025-1A-012`.
- Bảo đảm chạy ingest lại không tạo đoạn trùng lặp.

**Lý do chia theo Item:** không trộn một đoạn rủi ro với bảng tài chính không liên quan. Lưu ý thực tế: NVIDIA chứa phần báo cáo tài chính chủ yếu ở Item 15; Microsoft có running header lặp lại. Parser phải kiểm thử hai trường hợp này.

**Kiểm chứng:** `uv run python scripts/fetch_filings.py`, sau đó `uv run pytest tests/test_ingest.py -v`. Dùng `scripts/find_in_filing.py` để đọc lại một đoạn khi viết ground truth.

### Giai đoạn 3 — Kho truy hồi và công cụ giá

**Mục tiêu:** tạo hai nguồn dữ liệu tách biệt và có thể kiểm chứng.

- `FilingStore` dùng Chroma cục bộ, tìm theo vector và luôn lọc metadata theo ticker. Kết quả kèm `score`, trong đó nhỏ hơn nghĩa là gần hơn.
- `get_stock_price` trả `PriceInfo` gồm giá, vốn hóa, P/E, thời điểm và cờ `stale`.
- Công cụ giá thử lấy dữ liệu trực tuyến trước; nếu lỗi hoặc chạy offline, dùng cache. Nếu không có cả hai, trả lỗi có kiểm soát thay vì bịa giá.

**Kiểm chứng:** `uv run pytest tests/test_vectorstore.py tests/test_price_tool.py -v`. Sau khi đã có dữ liệu: `uv run finassist ingest`; chạy lại và xác nhận tổng số chunk không tăng.

### Giai đoạn 4 — Hàng rào chống bịa nguồn và số liệu

**Mục tiêu:** mọi thông tin có thể lần về nguồn vừa truy hồi hoặc công cụ giá.

- `extract_citations` chỉ nhận marker `[chunk_id]` tồn tại trong danh sách chunk đã retrieve, giữ thứ tự xuất hiện đầu tiên và loại trùng.
- `ground_numbers` bỏ `revenue` hoặc `eps` nếu văn bản số không xuất hiện trong chunk đã retrieve.
- Nếu câu trả lời về filing không có marker hợp lệ, hệ thống thử tạo lại một lần; vẫn thất bại thì trả thông báo không đủ thông tin với `error=missing_citation`.

**Kiểm chứng:** `uv run pytest tests/test_postprocess.py tests/test_citations.py -v`.

### Giai đoạn 5 — Đồ thị LangGraph và các chain thật

**Mục tiêu:** điều phối đúng theo ý định, giữ phần quyết định được kiểm thử tách khỏi phần gọi LLM.

- `classify_intent` trả structured output và lọc ticker ngoài phạm vi.
- `price_tool`, `retrieve`, `generate_answer`, `format_output`, `refuse` là các node chính.
- `generate_answer` chỉ nhận context từ state. `format_output` tạo JSON nhưng citation, số được grounding và giá được gắn lại tất định.
- `runtime.py` nối cấu hình, Chroma, LLM và đồ thị thành ứng dụng gọi được qua `ask`.

**Kiểm chứng:** `uv run pytest tests/test_graph_routing.py tests/test_citations.py tests/test_chains.py -v`, rồi chạy toàn bộ `uv run pytest -q`.

### Giai đoạn 6 — CLI, bản ghi phát lại và giao diện

**Mục tiêu:** demo vẫn đáng tin cậy khi không có mạng hoặc router.

- CLI hỗ trợ `finassist ingest` và `finassist ask`; tùy chọn `--record` lưu output và các chunk, `--replay` đọc lại bản ghi mà không gọi mạng.
- Streamlit hiển thị câu trả lời, citation, chunk retrieve và JSON. Chế độ Replay phải nói rõ rằng câu hỏi trong ô nhập không được thực thi mà bản ghi được phát lại.

**Kiểm chứng:**

```bash
uv run finassist ask "What was Apple's total net sales in its most recent fiscal year?" --record runs/demo.json
uv run finassist ask x --replay runs/demo.json
uv run streamlit run app/streamlit_app.py
```

### Giai đoạn 7 — Golden set và đánh giá RAGAS

**Mục tiêu:** đo chất lượng bằng tập câu hỏi có ground truth do con người viết từ 10-K.

- Tạo 30 câu: 10 numeric, 10 risk/MD&A, 5 combined và 5 trap ngoài phạm vi.
- Với mọi câu không phải trap, đọc hồ sơ gốc rồi viết `ground_truth`; không để LLM tự sinh đáp án rồi tự chấm.
- Chấm faithfulness, context precision, context recall và answer relevancy cho 25 câu có nội dung. Năm trap được chấm bằng intent accuracy.
- Xuất `reports/ragas_baseline.json` và `reports/ragas_baseline.md`; giữ cả case thất bại và nguyên nhân, không xóa điểm NaN một cách âm thầm.

**Kiểm chứng:** `uv run pytest tests/test_golden_set.py tests/test_report.py -v`, sau đó `uv run python -m eval.run_ragas`.

**Cách đọc kết quả:** baseline hiện tại có faithfulness 0.939, context precision 0.309, context recall 0.387, answer relevancy 0.687, intent 30/30 và numeric sanity 7/10. Đây là số của một lần đo; judge dùng cùng model với ứng dụng, nên không được diễn giải như đánh giá độc lập. Xem caveat trong báo cáo trước khi trích dẫn.

### Giai đoạn 8 — Quan sát, tài liệu và bàn giao demo

**Mục tiêu:** một người khác có thể theo dõi một run thật, trình bày demo và biết giới hạn của hệ thống.

- Bật LangSmith qua `LANGSMITH_*`, gắn tag `demo` hoặc `eval`, metadata version/intent/ticker và kiểm tra tree có state, tool call và các chunk retrieve.
- Lưu link trace và text rendering vào `docs/evidence/`; ảnh chụp UI chỉ thực hiện khi có quyền đăng nhập.
- Hoàn thiện tài liệu kiến trúc, knowledge check, kịch bản demo, nhật ký đóng góp và bước cải tiến tiếp theo.

**Kiểm chứng:** mở URL từ `uv run python scripts/trace_link.py`; xác nhận nhánh `both` đi qua `classify_intent → price_tool → retrieve → generate_answer → format_output`, còn câu tư vấn đầu tư đi qua `classify_intent → refuse`.

## 6. Bản đồ mã nguồn

| Khu vực | Vai trò |
|---|---|
| `src/finassist/config.py` | cấu hình, ticker và đường dẫn dữ liệu |
| `src/finassist/schemas.py` | hợp đồng dữ liệu Pydantic |
| `src/finassist/ingest.py` | parse 10-K, chia Item/chunk |
| `src/finassist/vectorstore.py` | embedding và Chroma |
| `src/finassist/tools.py` | giá cổ phiếu, cache và lỗi có kiểm soát |
| `src/finassist/postprocess.py` | citation tất định, grounding số liệu |
| `src/finassist/graph.py` | GraphState, router và node |
| `src/finassist/chains.py` | prompt và triển khai LLM thật |
| `src/finassist/runtime.py` | khởi tạo app, thực thi câu hỏi |
| `src/finassist/cli.py` | CLI và record/replay |
| `eval/` | golden set, chạy RAGAS và tạo báo cáo |
| `app/streamlit_app.py` | giao diện demo |

## 7. Ma trận yêu cầu và bằng chứng

| Yêu cầu | Ý nghĩa | Bằng chứng chính |
|---|---|---|
| R1 | đồ thị có nhánh theo intent | `tests/test_graph_routing.py` |
| R2 | có công cụ giá tất định | `tests/test_price_tool.py` |
| R3 | chunk theo Item, metadata đầy đủ, ingest idempotent | `tests/test_ingest.py`, `tests/test_vectorstore.py` |
| R4 | citation chỉ trỏ chunk đã retrieve | `tests/test_citations.py` |
| R5 | output hợp lệ và số có bằng chứng | `tests/test_schema.py`, `tests/test_postprocess.py` |
| R6 | golden set đủ 30 câu | `tests/test_golden_set.py` |
| R7 | báo cáo có bốn metric | `reports/ragas_baseline.md` |
| R8 | trace thấy node, tool và chunk | `docs/evidence/` |
| R9 | từ chối ngoài phạm vi/tư vấn đầu tư | `tests/test_graph_routing.py` |
| R10 | không lộ secret | `.gitignore`, quét Git |
| R11 | phát lại offline | `tests/test_cli_replay.py`, `runs/demo.json` |
| R12 | bộ tài liệu demo hoàn chỉnh | `docs/DEMO_SCRIPT.md` |

## 8. Xác nhận trước khi bàn giao

```bash
uv run pytest -q
git ls-files | grep -E '^\.env$' || echo "OK: .env không được theo dõi"
git grep -nE 'sk-[A-Za-z0-9-]{12,}|lsv2_pt_' -- . ':!docs/DEMO_SCRIPT.md'
uv run finassist ask x --replay runs/demo.json
```

Ngoài các lệnh trên, hãy kiểm tra `reports/ragas_baseline.md`, `docs/evidence/links.md` và checklist trong [`docs/DEMO_SCRIPT.md`](../../DEMO_SCRIPT.md). Không coi một test xanh là bằng chứng rằng retrieval đã tốt: các điểm yếu về bảng số liệu phải được nêu rõ trong [`docs/NEXT_STEP.md`](../../NEXT_STEP.md).

## 9. Rủi ro còn lại và hướng cải thiện

Rủi ro nổi bật là retrieval có thể không lấy được đúng đoạn bảng: cả năm case baseline tệ nhất đều bỏ lỡ golden chunk. Hàng rào `ground_numbers` chỉ kiểm tra số có xuất hiện trong chunk đã retrieve, không kiểm tra ý nghĩa của số; vì vậy ở q10 và q20, con số 139,297 (lấy từ bảng mảng không có tiêu đề cột của chunk `NVDA-2026-7-029`) vẫn được chấp nhận là doanh thu, trong khi doanh thu thật là 215,938. Hướng cải thiện ưu tiên là hybrid retrieval (BM25 + vector), có reranker và giữ tiêu đề cột/nhãn dòng khi chunk bảng. Tiêu chí thành công là cải thiện context precision và context recall cho nhóm numeric/combined trên cùng golden set, đồng thời không làm giảm faithfulness đáng kể.

Kế hoạch đo lường chi tiết: [`docs/NEXT_STEP.md`](../../NEXT_STEP.md).
