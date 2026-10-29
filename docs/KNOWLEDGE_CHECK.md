# KNOWLEDGE CHECK (Explain / Predict / Debug / Justify)

Mỗi mục có một câu trả lời sẵn 60-90 giây (đọc to khoảng 150-200 từ). Mọi số liệu lấy từ `reports/ragas_baseline.{md,json}`, `docs/evidence/` và `runs/demo.json`. `file:line` đã kiểm tra với code hiện tại.

## 1. Explain: "Hãy giải thích luồng xử lý của một câu hỏi"

Dùng câu `both`: "What is NVIDIA's current stock price and how do export controls threaten it?" (trace thật trong `docs/evidence/trace_tree.md`).

**Câu trả lời (60-90 s):**

"Hệ thống là một LangGraph `StateGraph` trong `src/finassist/graph.py`. Câu hỏi đi vào `classify_intent`, node này gọi LLM với structured output `IntentResult` (prompt ở `chains.py:17-23`) và trả `intent` cùng `tickers`. Với câu này intent là `both`, ticker NVDA. Hàm `route` (`graph.py:66`) đưa `both` vào `price_tool`. Node này gọi hàm tất định `get_stock_price` (`tools.py:60`), không để LLM tự nói giá; kết quả là 215.38 USD, market cap 5.20 nghìn tỷ, P/E 27.54. Vì intent là `both`, `after_price` (`graph.py:78`) đi tiếp sang `retrieve`, lấy 5 chunk của NVDA từ Chroma, lọc theo metadata ticker; trong trace là `NVDA-2026-1A-126`, `1A-106`, `15-085`, `15-025`, `15-020`. `generate_answer` yêu cầu LLM chỉ dùng chunk và giá đã cung cấp, mỗi khẳng định gắn `[chunk_id]`. Cuối cùng `format_output` trích các trường số bằng LLM, nhưng citation được tạo tất định từ marker (`postprocess.py:10-22`), số liệu không có trong chunk bị đặt `null` (`ground_numbers`, `postprocess.py:36`), và giá lấy từ tool. Kết quả là `FinanceAnswer` JSON. Câu ngoài phạm vi đi nhánh `refuse`, không gọi LLM sinh nội dung."

Các file nên nêu tên: `graph.py` (điều phối), `chains.py` (LLM thật cho từng node), `postprocess.py` (citation và grounding), `tools.py` (giá).

## 2. Predict: "Với câu q21, state và tool call sẽ thế nào?"

q21 (`eval/golden_set.json`): "What is Apple's current stock price, and what were its total net sales last fiscal year?", `expected_intent = both`.

**Dự đoán, theo từng node:**

| Node | Dự đoán |
|---|---|
| `classify_intent` | `intent = both`, `tickers = ["AAPL"]`; `retrieved_chunks = []`, `price_data = []`, `short_circuit = False` |
| `route` / `price_tool` | đi vào `price_tool`; tool call `get_stock_price("AAPL", cache_path, offline=False)` (`chains.py:121-122`); `price_data = [PriceInfo(AAPL, price, market_cap, pe_ratio, as_of, stale=False)]`. Nếu yfinance lỗi: dùng `data/cache/prices.json` và `stale=True` (`tools.py:83-85`) |
| `after_price` | `both` -> `retrieve` |
| `retrieve` | 5 chunk AAPL (top-k=5). Chunk mong muốn là `AAPL-2025-7-005` (bảng segment, có Total net sales). Dự đoán trung thực: có khả năng không lấy được chunk này, vì chunk bảng số khó khớp embedding |
| `generate_answer` | trả giá từ khối PRICE DATA; phần net sales chỉ có số nếu chunk có; nếu chunk không chứa FY2025 total thì câu trả lời nói rõ thiếu |
| `format_output` | `intent = both`; `stock_price` = giá tool; `citations` chỉ gồm marker trùng với chunk đã retrieve; `revenue` null nếu số không có trong chunk |

**Đối chiếu với run thật** (`reports/ragas_baseline.json`, `per_question` q21): `predicted_intent = both`; giá lấy từ tool là 343.83 (as_of 2026-10-29 01:09 UTC, market cap khoảng 5.02 nghìn tỷ, P/E khoảng 39.8, không stale); 5 chunk retrieve đều là Item 8: `AAPL-2025-8-077`, `8-041`, `8-006`, `8-003`, `8-004`. Chunk `AAPL-2025-7-005` không có mặt. Câu trả lời nói ngữ cảnh không có total net sales FY2025 và chỉ nêu net sales FY2023 (383,285) và doanh thu Mỹ FY2025 (151,790) kèm trích dẫn `[AAPL-2025-8-077]`; `revenue = null`. Phần dự đoán về tool call và đường đi khớp; phần dự đoán "có thể không lấy được `7-005`" khớp. Lưu ý: giá và chunk id của một lần chạy khác sẽ khác nhau theo thời điểm.

**Câu trả lời (60-90 s):** "Với q21 em dự đoán `classify_intent` trả `both` và ticker AAPL, nên đi `price_tool` trước, gọi `get_stock_price('AAPL')` và state có một `PriceInfo`. `after_price` thấy intent `both` nên sang `retrieve`, lấy 5 chunk AAPL. Em dự đoán rủi ro nằm ở đây: chunk có Total net sales FY2025 là `AAPL-2025-7-005` có thể không nằm trong top-5. Rồi `generate_answer` ghép giá và chunk, `format_output` chỉ giữ citation khớp chunk đã retrieve và đặt `revenue` là null nếu số không có trong chunk. Em đã chạy thật để đối chiếu: intent và tool call đúng như dự đoán, giá 343.83 từ tool, nhưng 5 chunk đều là Item 8 và không có `7-005`, nên câu trả lời hedge thay vì bịa số."

## 3. Debug: "Một case tệ trong báo cáo: vì sao?"

Case: **q05** "What was Microsoft's net income in its most recent fiscal year?" (faithfulness 0.714; precision, recall, relevancy đều 0.0).

**Bằng chứng** (`reports/ragas_baseline.json`, `per_question` q05; mục "Worst cases" trong `reports/ragas_baseline.md`):
- Golden chunk là `MSFT-2026-8-001` (income statement, net income 133,749). Không nằm trong 5 chunk được retrieve: `MSFT-2026-8-079`, `7-002`, `8-134`, `7-018`, `7-020` (distance tốt nhất 0.439).
- Câu trả lời từ chối đúng ("context doesn't include Microsoft's net income for fiscal year 2026"), nêu operating income 155,237 từ `7-020` và các số pro forma của FY2024/2023 từ `8-079`. Không bịa số; `revenue = null`, `eps = null`; `answer_has_expected_number = false`.
- Nội dung chunk `MSFT-2026-8-001` (đọc từ `data/raw` qua `load_filing`) bắt đầu bằng "Net income $ 133,749 $ 101,832 ..." mỗi số một dòng riêng, không có từ "Microsoft" hay "fiscal year" trong đoạn đầu.

**Phân tích nguyên nhân:** lỗi nằm ở bước `retrieve`, không phải ở LLM. Retrieval dựa vào độ tương đồng embedding bge-small giữa câu hỏi và chunk; chunk bảng gồm chủ yếu chữ số nên khớp kém, trong khi chunk MD&A và note có nhiều chữ mô tả hơn. Đây là giả thuyết phù hợp với dữ liệu (cùng mẫu ở q04, q21, q24, q03: "In 5 of 5 worst cases the golden chunk id ... was not retrieved"); chưa kiểm chứng bằng thí nghiệm cô lập. Faithfulness 0.71 là do một số câu trong answer (so sánh với pro forma, "closest figure") không được judge xem là suy ra trực tiếp từ context; judge chính là model của app nên điểm này còn có thể bị chấm lỏng (Caveats).

**Cách debug:** (1) mở `per_question` q05 và xem `retrieved` so với chunk id trong `ground_truth`; (2) mở trace LangSmith, node `retrieve` để thấy danh sách chunk; (3) tìm chunk đúng bằng `scripts/find_in_filing.py`; (4) kết luận lỗi ở retrieval nên sửa ở đó (`docs/NEXT_STEP.md`), không sửa prompt.

**Câu trả lời (60-90 s):** "Em chọn q05, net income của Microsoft. Faithfulness 0.71 và cả precision, recall đều 0. Em nhìn `per_question` thì thấy chunk đúng `MSFT-2026-8-001` không nằm trong 5 chunk retrieve; thay vào đó là các chunk MD&A và note pro forma. Câu trả lời nói thẳng là context không có net income FY2026 và nêu operating income 155,237 từ `7-020`, nên không bịa số, `revenue` và `eps` đều null. Vậy lỗi nằm ở retrieval chứ không phải generation: chunk bảng chỉ toàn chữ số nên embedding khớp kém với câu hỏi. Đó là giả thuyết, đồng thời cùng mẫu xuất hiện ở 5 trên 5 case tệ nhất. Em chưa chứng minh bằng thí nghiệm riêng. Cách sửa em đề xuất là hybrid BM25 cộng vector có reranker, và giữ tiêu đề cột khi chunk bảng, đo lại bằng cùng golden set."

## 4. Justify: "Chọn một trace, nêu một cải tiến, và lý do"

Trace chọn: `01a18548-f611-70d1-ad75-5bb267a4a30e` (intent `both`, NVDA; URL trong `docs/evidence/links.md`).

**Câu trả lời (60-90 s):** "Trong trace này `generate_answer` mất 5.50 s và `format_output` mất 5.27 s, mỗi node một lần gọi LLM, còn `retrieve` chỉ 0.74 s. Chất lượng của câu trả lời phụ thuộc vào 5 chunk mà `retrieve` trả về, và baseline cho thấy chỗ yếu nhất của hệ thống là đó: context precision 0.309, context recall 0.387, trong khi faithfulness là 0.939 (số này có thể cao vì judge cùng model với app). Nên em chọn cải tiến retrieval cho chunk báo cáo tài chính: hybrid BM25 cộng vector với reranker, và/hoặc chunk theo bảng giữ tiêu đề cột. Em đo bằng cùng golden set 30 câu, cùng 4 metric, nên so sánh được trước và sau. Em không chọn tăng top-k đơn thuần vì sẽ kéo thêm nhiễu cho LLM và chi phí token. Kế hoạch chi tiết ở `docs/NEXT_STEP.md`."

## 5. Hint questions: câu trả lời chuẩn bị sẵn

### "Show me where" (chỉ cho tôi ở đâu)

| Câu hỏi | Chỉ vào đâu |
|---|---|
| Citation được trích ra ở đâu? | `src/finassist/postprocess.py:7` (`MARKER_RE`), `postprocess.py:10-22` (`extract_citations`: chỉ giữ marker khớp chunk đã retrieve, khử trùng, `quote` = 200 ký tự đầu của chunk); gán vào output ở `graph.py:110-113` |
| Retrieve node ở đâu? | code `graph.py:81-82` -> `chains.py:118-119` -> `vectorstore.py:41-52`; trong trace là node `retrieve` (0.74 s, 5 chunk) |
| Tool giá ở đâu? | `tools.py:60-90`; trong trace là node `price_tool` (1.67 s) với `price_data` |
| Từ chối ở đâu? | `graph.py:116-118`; trace `01a18549-d281-74e8-93cf-cb568132d110` chỉ có `classify_intent` và `refuse` |
| Chặn số bịa ở đâu? | `ground_numbers`, `postprocess.py:36-44` |
| Chunk theo Item ở đâu? | `ingest.py:46-62` (`split_items`), `ingest.py:65-74` (`chunk_filing`) |

### "Why this choice" (vì sao chọn)

Xem bảng quyết định trong `docs/ARCHITECTURE.md` mục 6. Tóm tắt một câu cho mỗi lựa chọn: LangGraph có nhánh (dễ đánh giá hơn ReAct tự do), Chroma (có filter ticker, FAISS không), bge-small local (không cần key và không gửi filing ra ngoài), citation tất định (LLM có thể bịa id), `ground_numbers` (tránh EPS/revenue sai âm thầm), chunk theo Item (không trộn rủi ro với bảng), model ghim `kr/claude-sonnet-5.5` (`auto/*` lỗi rate limit).

### "What would fail" (cái gì sẽ hỏng)

| Tình huống | Điều xảy ra (theo code) | Đã có test/evidence? |
|---|---|---|
| Giá cũ hoặc yfinance lỗi | dùng cache với `stale=True` (`tools.py:83-85`); không cache thì `PriceUnavailable`, node ghi `error` và câu trả lời thêm ghi chú "price unavailable" (`graph.py:22-23`, `graph.py:75-76`) | `tests/test_price_tool.py`, `tests/test_graph_routing.py` |
| LLM không gắn marker `[chunk_id]` | thử lại một lần, vẫn không có thì `INSUFFICIENT_TEXT` và `error = missing_citation` (`graph.py:95-99`) | `tests/test_citations.py` |
| Parse Item sai (cross-reference "see Item 1A") | `_is_heading` loại tham chiếu trong văn xuôi (`ingest.py:33-43`); Item bị thiếu thì không có chunk cho Item đó | `tests/test_ingest.py`; NVDA Item 8 chỉ có 1 chunk vì financial statements nằm ở Item 15 |
| Ngoài phạm vi (Tesla, thời tiết, dự đoán giá, Alphabet) | `refuse` (intent accuracy 30/30 trên 5 bẫy trong baseline) | `reports/ragas_baseline.md` |
| Classifier gán nhầm lời khuyên đầu tư thành `price` | không có chốt chặn thứ hai (không có keyword rule); đây là rủi ro đã biết | chỉ có integration smoke test cho câu "Should I buy NVIDIA stock now?" (`tests/test_llm_smoke.py:20`), không có test cho các cách hỏi khác |
| Retrieval bỏ sót chunk bảng | model hedge thay vì bịa; số liệu null | baseline: 5/5 worst case, numeric sanity 7/10 |
| Câu hỏi so sánh nhiều công ty | `retrieve` lấy top-5 cho mỗi ticker (`chains.py:119`), nhưng chunk đúng vẫn có thể thiếu (q24 recall 0.0 và precision 0.0) | `reports/ragas_baseline.md` |

### "How did you verify" (bạn kiểm chứng bằng cách nào)

- `uv run pytest -q`: 135 passed, 11 skipped (11 integration test cần `RUN_INTEGRATION=1`; chạy với router bật thì 11 passed).
- RAGAS baseline: `reports/ragas_baseline.md` (25 câu đánh giá bằng RAGAS, 5 câu bẫy chấm bằng intent: 30/30), kèm Caveats (judge cùng model với app; relevancy 0.0 ở 6 dòng là artifact của RAGAS).
- Ground truth do người viết tay từ 10-K, 30 câu (`eval/golden_set.json`).
- Trace LangSmith: `docs/evidence/links.md` và `docs/evidence/trace_tree.md` (không có ảnh chụp màn hình vì UI cần đăng nhập; trace tree lấy qua API).
- Replay offline: `uv run finassist ask x --replay runs/demo.json`.
