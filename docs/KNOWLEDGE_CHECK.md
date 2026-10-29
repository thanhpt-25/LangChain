# Câu hỏi kiểm tra kiến thức (Giải thích / Dự đoán / Gỡ lỗi / Bảo vệ lựa chọn)

Mỗi mục có một câu trả lời sẵn 60-90 giây (đọc to khoảng 150-200 từ). Số liệu lấy từ `reports/ragas_baseline_dense.json` (vector thuần), `reports/ragas_retrieval_v2.{md,json}` (cấu hình hiện tại: vector top-30 và reranker), `reports/retrieval_experiments.md`, `docs/evidence/` và `runs/demo.json`. `file:line` đã kiểm tra với code hiện tại.

## 1. Giải thích: "Hãy giải thích luồng xử lý của một câu hỏi"

Dùng câu `both`: "What is NVIDIA's current stock price and how do export controls threaten it?" (trace thật trong `docs/evidence/trace_tree.md`).

Câu trả lời (60-90 s):

"Hệ thống là một LangGraph `StateGraph` trong `src/finassist/graph.py`. Câu hỏi đi vào `classify_intent`, node này gọi LLM với structured output `IntentResult` (prompt ở `chains.py:18-27`) và trả `intent` cùng `tickers` (kèm `language` và `search_query`; với câu không phải tiếng Anh, `search_query` là bản viết lại tiếng Anh dùng để retrieve). Với câu này intent là `both`, ticker NVDA. Hàm `route` (`graph.py:68-69`) đưa `both` vào `price_tool`. Node này gọi hàm tất định `get_stock_price` (`tools.py:60`), không để LLM tự nói giá; kết quả là 215.38 USD, market cap 5.20 nghìn tỷ, P/E 27.54. Vì intent là `both`, `after_price` (`graph.py:80-81`) đi tiếp sang `retrieve`. Ở đây `Retriever` lấy 30 chunk ứng viên của NVDA từ Chroma, lọc theo metadata ticker, rồi một cross-encoder cục bộ chấm lại và giữ 5 chunk. Trace thật được ghi trước khi thêm reranker nên 5 chunk trong đó, `NVDA-2026-1A-126`, `1A-106`, `15-085`, `15-025`, `15-020`, là top-5 của vector. `generate_answer` yêu cầu LLM chỉ dùng chunk và giá đã cung cấp, mỗi khẳng định gắn `[chunk_id]`. Cuối cùng `format_output` trích các trường số bằng LLM, nhưng citation được tạo tất định từ marker (`postprocess.py:10-22`), số liệu không có trong chunk bị đặt `null` (`ground_numbers`, `postprocess.py:36`), và giá lấy từ tool. Kết quả là `FinanceAnswer` JSON. Câu ngoài phạm vi đi nhánh `refuse`, không gọi LLM sinh nội dung."

Các file nên nêu tên: `graph.py` (điều phối), `chains.py` (LLM thật cho từng node), `postprocess.py` (citation và grounding), `tools.py` (giá).

## 2. Dự đoán: "Với câu q21, state và tool call sẽ thế nào?"

q21 (`eval/golden_set.json`): "What is Apple's current stock price, and what were its total net sales last fiscal year?", `expected_intent = both`.

Dự đoán, theo từng node:

| Node | Dự đoán |
|---|---|
| `classify_intent` | `intent = both`, `tickers = ["AAPL"]`, `search_query` = câu hỏi gốc (câu tiếng Anh); `retrieved_chunks = []`, `price_data = []`, `short_circuit = False` |
| `route` / `price_tool` | đi vào `price_tool`; tool call `get_stock_price("AAPL", cache_path, offline=False)` (`chains.py:128-129`); `price_data = [PriceInfo(AAPL, price, market_cap, pe_ratio, as_of, stale=False)]`. Nếu yfinance lỗi: dùng `data/cache/prices.json` và `stale=True` (`tools.py:83-85`) |
| `after_price` | intent `both` đi tiếp sang `retrieve` |
| `retrieve` | 30 ứng viên AAPL từ Chroma, reranker giữ 5. Chunk mong muốn là `AAPL-2025-7-005` (bảng segment, có Total net sales). Dự đoán trung thực: có khả năng không lấy được chunk này, vì chunk bảng số khó khớp embedding; reranker có thể đưa một chunk khác chứa con số vào top-5 |
| `generate_answer` | trả giá từ khối PRICE DATA; phần net sales chỉ có số nếu chunk có; nếu chunk không chứa FY2025 total thì câu trả lời nói rõ thiếu |
| `format_output` | `intent = both`; `stock_price` = giá tool; `citations` chỉ gồm marker trùng với chunk đã retrieve; `revenue` null nếu số không có trong chunk |

Đối chiếu với hai run thật. Run vector thuần (`reports/ragas_baseline_dense.json`, `per_question` q21): `predicted_intent = both`; giá lấy từ tool là 343.83 (as_of 2026-10-29 01:09 UTC, market cap khoảng 5.02 nghìn tỷ, P/E khoảng 39.8, không stale); 5 chunk retrieve đều là Item 8: `AAPL-2025-8-077`, `8-041`, `8-006`, `8-003`, `8-004`. Chunk `AAPL-2025-7-005` không có mặt. Câu trả lời nói ngữ cảnh không có total net sales FY2025 và chỉ nêu net sales FY2023 (383,285) kèm trích dẫn `[AAPL-2025-8-077]`; `revenue = null`. Run cấu hình hiện tại (`reports/ragas_retrieval_v2.json`): cùng intent, giá 343.83 (as_of 03:18 UTC); 5 chunk là `AAPL-2025-8-006`, `8-001`, `8-024`, `8-004`, `8-023`. `7-005` vẫn vắng, nhưng chunk `8-023` có con số, nên câu trả lời nêu 416,161 kèm `[AAPL-2025-8-023]` và `revenue = 416161`, qua `ground_numbers`. Faithfulness của câu này là 0.778 vì nó thêm một suy luận (FY2026 có lẽ vừa kết thúc) mà không chunk nào hỗ trợ. Phần dự đoán về tool call và đường đi khớp ở cả hai run; phần dự đoán "có thể không lấy được `7-005`" khớp ở cả hai, và reranker đã đổi kết quả từ hedge sang câu trả lời đúng. Lưu ý: giá và chunk id của một lần chạy khác sẽ khác nhau theo thời điểm.

Câu trả lời (60-90 s): "Với q21 em dự đoán `classify_intent` trả `both` và ticker AAPL, nên đi `price_tool` trước, gọi `get_stock_price('AAPL')` và state có một `PriceInfo`. `after_price` thấy intent `both` nên sang `retrieve`, lấy 30 ứng viên AAPL rồi reranker giữ 5. Em dự đoán rủi ro nằm ở đây: chunk có Total net sales FY2025 là `AAPL-2025-7-005` có thể không nằm trong top-5. Rồi `generate_answer` ghép giá và chunk, `format_output` chỉ giữ citation khớp chunk đã retrieve và đặt `revenue` là null nếu số không có trong chunk. Em đã chạy thật để đối chiếu: intent và tool call đúng như dự đoán, giá 343.83 từ tool. Ở run vector thuần, 5 chunk đều là Item 8 và không có `7-005`, nên câu trả lời hedge. Ở run có reranker, `7-005` vẫn vắng nhưng một chunk Item 8 khác chứa 416,161, nên câu trả lời đúng. Đây là một ví dụ mà reranker giúp."

## 3. Gỡ lỗi: "Một trường hợp có kết quả kém trong báo cáo: vì sao?"

Case: q05 "What was Microsoft's net income in its most recent fiscal year?". Câu này vẫn trượt ở cấu hình hiện tại (faithfulness 0.867; precision, recall, relevancy đều 0.0).

Bằng chứng (`reports/ragas_retrieval_v2.json`, `per_question` q05; mục "Worst cases" trong `reports/ragas_retrieval_v2.md`; số của run vector thuần lấy từ `reports/ragas_baseline_dense.json`):
- Golden chunk là `MSFT-2026-8-001` (income statement, net income 133,749). Con số này cũng nằm trong `MSFT-2026-8-000` và `MSFT-2026-7-016`. Theo bộ đo truy hồi, chunk bằng chứng tốt nhất của q05 đứng ở hạng 12 sau khi chấm lại, tức ngoài top-5. Test `test_default_retriever_q05_known_miss` (`tests/test_retrieval_integration.py`) ghi nhận điều này bằng một `xfail` strict.
- 5 chunk được retrieve ở run hiện tại: `MSFT-2026-8-079` (bảng pro forma), `8-134`, `7-017`, `7-002`, `8-136`. Ở run vector thuần là `8-079`, `7-002`, `8-134`, `7-018`, `7-020`. Hai danh sách khác nhau nhưng đều không có chunk chứa 133,749.
- Câu trả lời hiện tại từ chối đúng ("context doesn't include Microsoft's reported net income"), nêu adjusted diluted EPS 17.28 (non-GAAP) và số 88,308 pro forma của FY2024 kèm ghi chú rằng đó không phải net income của năm gần nhất. Không bịa số; `revenue = null`, `eps = null`; `answer_has_expected_number = false`. Run vector thuần cũng hedge, và nêu operating income 155,237 làm số gần nhất.
- Nội dung chunk `MSFT-2026-8-001` bắt đầu bằng "Net income $ 133,749 $ 101,832 ..." với mỗi số một dòng riêng (đọc từ `data/raw` qua `load_filing`), nên phần chữ mô tả rất ít.

Phân tích nguyên nhân: lỗi nằm ở bước `retrieve`; LLM chỉ nhận đúng những chunk được đưa cho nó. Chunk bảng gồm chủ yếu chữ số nên khớp kém với câu hỏi bằng lời, và cross-encoder chỉ xếp lại 30 ứng viên đã có, nên nó không đổi được kết quả khi chunk đúng đứng ở hạng 12 theo cả hai tầng. Ta đã thử hai cách sửa theo giả thuyết "chunk bảng khó khớp": chunk theo hàng bảng kèm tiền tố embed, và hybrid BM25. Cả hai đều không đưa q05 vào top-5 (`reports/retrieval_experiments.md`, giai đoạn 2 và 3), nên giả thuyết ban đầu chưa được xác nhận và chưa có lời giải thích khác được chứng minh. Faithfulness 0.867 cho thấy judge thấy một số câu trong answer chưa được context hỗ trợ đầy đủ; JSON không lưu lý do của judge. Judge là model của app nên điểm này còn có thể bị chấm lỏng (Caveats).

Cách debug: (1) mở `per_question` q05 và xem `retrieved` so với chunk id trong `ground_truth`; (2) chạy `uv run python -m eval.retrieval_eval --label x --show-misses` để xem hạng của từng nhóm bằng chứng; (3) mở trace LangSmith, node `retrieve` để thấy danh sách chunk; (4) tìm chunk đúng bằng `scripts/find_in_filing.py`; (5) kết luận lỗi ở retrieval nên sửa ở đó (`docs/NEXT_STEP.md`), không sửa prompt.

Câu trả lời (60-90 s): "Em chọn q05, net income của Microsoft. Câu này vẫn hụt sau khi thêm reranker. Chunk có 133,749 đứng ở hạng 12, ngoài top-5, nên 5 chunk gửi cho LLM là các chunk MD&A và pro forma. Câu trả lời nói thẳng là context không có net income đã báo cáo và nêu rõ số 88,308 là số pro forma, nên không bịa số, `revenue` và `eps` đều null; precision, recall, relevancy đều 0. Vậy lỗi nằm ở retrieval, còn LLM làm đúng với những gì nó nhận. Em đã thử hai giả thuyết sửa: hybrid BM25 và chunk theo hàng bảng. Cả hai đều không đưa chunk này vào top-5, nên em chưa biết nguyên nhân chắc chắn, chỉ biết chunk toàn chữ số và nằm ngoài top-5 ở mọi cấu hình đã đo. Hướng tiếp theo là truy vấn gần nội dung bảng hơn, đo lại bằng cùng bộ 25 câu."

### Ca Debug thứ hai: reranker làm q04 tệ đi

Case: q04 "What was Microsoft's total revenue in its most recent fiscal year?". Đây là ca hồi quy thật do thay đổi truy hồi, so sánh `reports/ragas_baseline_dense.json` với `reports/ragas_retrieval_v2.json`.

Bằng chứng:

| | vector thuần (`ragas_baseline_dense`) | vector top-30 và reranker (`ragas_retrieval_v2`) |
|---|---|---|
| 5 chunk | `7-002`, `8-079`, `8-134`, `7-020`, `7-021` | `8-134`, `7-002`, `7-017`, `8-079`, `7-021` |
| câu trả lời | 331,839 triệu USD, tăng 18% so với 281,724, trích `[MSFT-2026-7-020]` | nói context không nêu tổng doanh thu, chỉ nêu mức tăng 18% và Microsoft Cloud 214,4 tỷ USD |
| faithfulness | 0.714 | 0.875 |
| precision | 0.250 | 0.000 |
| recall | 1.0 | 0.0 |
| relevancy | 0.881 | 0.0 |

Chunk `MSFT-2026-7-020` chứa `Total Revenue $ 331,839 $ 281,724` và nằm trong top-5 của vector thuần. Cross-encoder đánh giá nó thấp hơn `7-017` nên đẩy nó ra, và theo bộ đo bằng chứng của q04 nay ở hạng 7. Một lần chạy lại vector thuần (`ragas_baseline_rerun`) cũng có recall 1.0 ở q04, nên kết quả này lặp lại ở cả hai lần chạy vector thuần. Lưu ý thêm: ở run vector thuần, bảng trong chunk `7-020` không có tiêu đề cột, và model phải tự suy ra cột đầu là FY2026 rồi nói rõ điều đó trong câu trả lời.

Phân tích nguyên nhân: reranker chấm theo độ khớp văn bản giữa câu hỏi và chunk, và `7-017` là đoạn văn MD&A nói về "total revenue increased 18%", khớp từ ngữ hơn một bảng toàn số. Đây là cùng nguyên nhân với q05 nhìn từ phía ngược lại: bảng số bị thiệt cả ở vector lẫn ở cross-encoder. Giả thuyết này nhất quán với số liệu nhưng chưa được kiểm chứng riêng. Hệ thống hedge và không bịa số, nên ca này mất chất lượng nhưng không mất an toàn.

Câu trả lời (60-90 s): "Ca thứ hai là q04, total revenue của Microsoft, và đây là ca hồi quy do chính thay đổi của mình. Ở vector thuần câu trả lời đúng, 331,839 triệu USD, recall 1.0, vì chunk `7-020` có dòng Total Revenue nằm trong top-5. Sau khi thêm reranker, `7-020` bị đẩy ra và thay bằng một đoạn MD&A nói về mức tăng 18%, nên câu trả lời chỉ nêu tăng 18% và Microsoft Cloud, còn precision, recall, relevancy đều về 0. Bộ đo truy hồi cũng nói như vậy: bằng chứng của q04 ở hạng 7. Em đọc đây là cross-encoder ưu tiên đoạn văn khớp từ ngữ hơn bảng toàn số, nhưng em chưa kiểm chứng riêng giả thuyết này. Nó cho thấy mức tăng any@5 từ 48% lên 64% đi kèm ít nhất một câu tệ đi."

### Ca Debug thứ ba: con số sai vẫn qua grounding (q10, q20 ở baseline)

Case: q10 "How much revenue did NVIDIA's Data Center market platform generate...?" và q20 "What did NVIDIA's MD&A give as the main drivers of revenue change...?" (`reports/ragas_baseline_dense.json`, `per_question`).

Bằng chứng: ở run vector thuần, cả hai câu trả lời nêu "total revenue $139,297" (so với $87,960 năm trước) từ chunk `NVDA-2026-7-029`, một bảng segment không có tiêu đề cột (Compute & Networking 130,141; Graphics 9,156; Total 139,297; 130,141 + 9,156 = 139,297). Doanh thu thật FY2026 là 215,938 (q07 trả đúng, năm trước 130,497). Trường `revenue` = 139297 ở cả hai dòng. Faithfulness: q20 = 1.0, q10 = 0.941. Ở run `ragas_retrieval_v2` hai câu này không lặp lại lỗi: q10 hedge (chunk `7-029` vẫn nằm trong top-5 nhưng câu trả lời không dùng số đó) và q20 không còn lấy chunk `7-029`. Nghĩa là lỗi biến mất vì truy hồi đổi, còn phép kiểm tra không đổi.

Phân tích nguyên nhân: (1) retrieval không lấy được chunk đúng (bảng doanh thu theo thị trường `NVDA-2026-15-141` cho q10; `7-012` cho q20) và lấy một bảng không có tiêu đề; (2) `ground_numbers` (`postprocess.py:36-44`) chỉ kiểm tra chuỗi "139,297" có trong một chunk đã retrieve, không kiểm tra con số đó là gì, nên không chặn; (3) faithfulness đo xem câu trả lời có được context hỗ trợ không, và con số có nguyên văn trong context nên điểm cao (judge còn là cùng model với app). Đây là lỗi grounded-but-misread, khác với q05 là hedge đúng. Hướng sửa: giữ tiêu đề cột và nhãn dòng khi chunk bảng, và/hoặc kiểm tra nhãn ngữ nghĩa của số trước khi điền `revenue` (ví dụ so khớp với nhãn "Revenue" hoặc "Net sales" trong chunk); thêm kiểm tra sanity so với số đã biết (`docs/NEXT_STEP.md`).

Câu trả lời (60-90 s): "Ca thứ ba là q10 và q20 của NVIDIA ở baseline. Cả hai trả 'total revenue 139,297', nhưng doanh thu thật là 215,938, và q07 trả đúng số này. Con số lấy từ chunk `7-029`, một bảng segment mất tiêu đề cột: 130,141 cộng 9,156 bằng 139,297. `revenue` bị điền 139297 và vẫn qua `ground_numbers`, vì hàm này chỉ kiểm tra chuỗi số có trong chunk đã retrieve, không kiểm tra nghĩa. Judge chấm faithfulness 1.0 cho q20 vì con số có nguyên văn trong context; judge còn là cùng model với app. Ở lần chạy với reranker, hai câu này không còn sai, nhưng đó là do truy hồi đổi, còn phép kiểm tra thì giữ nguyên, nên lỗi vẫn có thể quay lại với một câu hỏi khác. Cách sửa là giữ tiêu đề cột khi chunk bảng và kiểm tra nhãn của số trước khi điền trường revenue."

## 4. Bảo vệ lựa chọn: "Chọn một trace, nêu một cải tiến và lý do"

Trace chọn: `01a18548-f611-70d1-ad75-5bb267a4a30e` (intent `both`, NVDA; URL trong `docs/evidence/links.md`).

Câu trả lời (60-90 s): "Trong trace này `generate_answer` mất 5.50 s và `format_output` mất 5.27 s, mỗi node một lần gọi LLM, còn `retrieve` chỉ 0.74 s. Chất lượng câu trả lời phụ thuộc vào 5 chunk mà `retrieve` trả về, và baseline cho thấy đó là chỗ yếu nhất: context precision 0.309, context recall 0.387, trong khi faithfulness 0.939 (có thể cao vì judge cùng model với app). Nên em cải tiến retrieval, và đo bằng bộ 25 câu có nhóm bằng chứng viết tay, không gọi LLM. Em thử theo từng giai đoạn với ngưỡng đặt trước. Reranker MiniLM trên 30 ứng viên của vector đưa any@5 từ 48% lên 64% và all@5 từ 28% lên 36%, thêm khoảng 140 ms. Em chọn MiniLM thay vì `bge-reranker-base` vì bge chỉ đạt 56% và mất 809 ms. Em không dùng hybrid BM25 vì đo được tăng 0 điểm, và không dùng chunk theo hàng bảng vì nó làm giảm còn 56% và 24%. Kết quả vẫn dưới mục tiêu 70% và 40%, và q04 tệ đi, nên em báo là chưa đạt, và phần RAGAS chưa kết luận được vì nhiễu giữa hai lần chạy là 0.02 đến 0.03. Em không chọn tăng top-k đơn thuần ở lần này vì nó kéo thêm nhiễu và chi phí token, nhưng nó là một hướng cần đo tiếp. Hướng tiếp theo em đề xuất là bỏ phần giá khỏi truy vấn của câu `both`, vì nhóm combined vẫn chỉ đạt 40% ở any@5. Kế hoạch chi tiết ở `docs/NEXT_STEP.md`."

## 5. Câu hỏi gợi ý và câu trả lời chuẩn bị sẵn

### "Chỉ cho tôi ở đâu"

| Câu hỏi | Chỉ vào đâu |
|---|---|
| Citation được trích ra ở đâu? | `src/finassist/postprocess.py:7` (`MARKER_RE`), `postprocess.py:10-22` (`extract_citations`: chỉ giữ marker khớp chunk đã retrieve, khử trùng, `quote` = 200 ký tự đầu của chunk); gán vào output ở `graph.py:113-116` |
| Retrieve node ở đâu? | `graph.py:83-84` gọi `chains.py:122-126`, rồi `Retriever.retrieve` trong `retrieval.py` (hàm `_for_ticker`: 30 ứng viên, chấm lại, giữ 5), rồi `vectorstore.py:67-78` (`search`); trong trace là node `retrieve` (0.74 s, 5 chunk, ghi trước khi có reranker) |
| Tool giá ở đâu? | `tools.py:60-90`; trong trace là node `price_tool` (1.67 s) với `price_data` |
| Từ chối ở đâu? | `graph.py:121-123`; trace `01a18549-d281-74e8-93cf-cb568132d110` chỉ có `classify_intent` và `refuse` |
| Chặn số bịa ở đâu? | `ground_numbers`, `postprocess.py:36-44` |
| Chunk theo Item ở đâu? | `ingest.py:63-79` (`split_items`), `ingest.py:82-91` (`chunk_filing`) |
| Công tắc truy hồi ở đâu? | `config.py:53-57` trong `get_settings` (các biến `FINASSIST_RETRIEVAL`, `FINASSIST_RERANKER`, `FINASSIST_CANDIDATES`, `FINASSIST_EMBED_PREFIX`, `FINASSIST_TABLE_ROWS`); `retrieval.py` hàm `make_retriever` |

### "Vì sao chọn phương án này"

Các quyết định kiến trúc và phương án đã loại nằm ở `docs/ARCHITECTURE.md` mục 7 (bảng ở mục 7.6). Mỗi lựa chọn tóm lại trong một câu: LangGraph có nhánh (dễ đánh giá hơn ReAct tự do), Chroma (lọc ticker ngay trong truy vấn; FAISS của LangChain có `filter=` nhưng chỉ lọc sau tìm kiếm ANN trên `fetch_k`, không có pre-filter gốc), bge-small local (không cần key và không gửi filing ra ngoài), citation tất định (LLM có thể bịa id), `ground_numbers` (tránh EPS/revenue sai âm thầm), chunk theo Item (không trộn rủi ro với bảng), model ghim `kr/claude-sonnet-5.5` (`auto/*` lỗi rate limit), reranker cục bộ MiniLM trên 30 ứng viên (đo được any@5 48% lên 64%, p50 thêm khoảng 140 ms; `bge-reranker-base`, hybrid BM25 và chunk theo hàng bảng đã đo và bị loại, số liệu ở `docs/ARCHITECTURE.md` mục 7.6).

### "Điều gì có thể hỏng"

Bảng sự cố đầy đủ hơn nằm ở `docs/ARCHITECTURE.md` mục 6, và các giới hạn đã biết ở mục 8.

| Tình huống | Điều xảy ra (theo code) | Đã có test/evidence? |
|---|---|---|
| Giá cũ hoặc yfinance lỗi | dùng cache với `stale=True` (`tools.py:83-85`); không cache thì `PriceUnavailable`, node ghi `error` và câu trả lời thêm ghi chú "price unavailable" (`graph.py:22-23`, `graph.py:76-77`) | `tests/test_price_tool.py`, `tests/test_graph_routing.py` |
| LLM không gắn marker `[chunk_id]` | thử lại một lần, vẫn không có thì `INSUFFICIENT_TEXT` và `error = missing_citation` (`graph.py:97-101`) | `tests/test_citations.py` |
| Parse Item sai (cross-reference "see Item 1A") | `_is_heading` loại tham chiếu trong văn xuôi (`ingest.py:50-60`); Item bị thiếu thì không có chunk cho Item đó | `tests/test_ingest.py`; NVDA Item 8 chỉ có 1 chunk vì financial statements nằm ở Item 15 |
| Ngoài phạm vi (Tesla, thời tiết, dự đoán giá, Alphabet) | `refuse` (intent accuracy: 5/5 bẫy, 30/30 tổng trong baseline) | `reports/ragas_baseline.md` |
| Classifier gán nhầm lời khuyên đầu tư thành `price` | không có chốt chặn thứ hai (không có keyword rule); đây là rủi ro đã biết | chỉ có integration smoke test cho câu "Should I buy NVIDIA stock now?" (`tests/test_llm_smoke.py:20`), không có test cho các cách hỏi khác |
| Retrieval bỏ sót chunk bảng | 9 trên 25 câu của bộ đo vẫn trượt top-5 (q04, q05, q10, q11, q15, q16, q22, q24, q25). Thường model hedge và số liệu null (q05, q24, và q04 ở cấu hình hiện tại); q21 từng hedge nhưng nay trả đúng; ở baseline q10/q20 trả 139,297 sai (xem dòng sau). Reranker chỉ xếp lại 30 ứng viên nên không cứu được chunk nằm ngoài chúng (q11, q22, q25) | `reports/retrieval_experiments.md`; numeric sanity 7/10 ở cả baseline và `ragas_retrieval_v2` |
| Reranker đẩy chunk đúng ra khỏi top-5 | q04: chunk `7-020` có `Total Revenue $ 331,839` rời top-5, recall từ 1.0 xuống 0.0; all@5 của nhóm risk giảm từ 10% xuống 0% | `reports/ragas_retrieval_v2.json`, `reports/retrieval_experiments.md`; ca Debug thứ hai |
| Câu filing đầu tiên sau khi khởi động | `Retriever` dựng ở lần truy hồi đầu, nên model reranker (khoảng 90 MB) được tải ở lần đầu trên máy rồi nạp vào bộ nhớ; câu đó chậm hơn hẳn. Biến `FINASSIST_RETRIEVAL` sai cũng chỉ lộ ra ở câu này | `src/finassist/retrieval.py` (`make_retriever`), `chains.py:122-126` |
| Độ trễ truy hồi | p50 khoảng 134 đến 149 ms mỗi lần truy hồi (vector thuần 12 ms), p95 khoảng 177 đến 178 ms | `reports/retrieval_experiments.md` |
| q05 bất ngờ vào top-5 sau một thay đổi | test `test_default_retriever_q05_known_miss` là `xfail` strict nên sẽ báo lỗi (XPASS), nhắc cập nhật tài liệu và bỏ `xfail` | `tests/test_retrieval_integration.py` |
| Số sai nhưng "có trong chunk" (grounded-but-misread) | `ground_numbers` chỉ kiểm tra chuỗi số xuất hiện trong chunk đã retrieve, không kiểm tra nghĩa; bảng không có tiêu đề cột cho phép lấy nhầm số (q10, q20: 139,297 thay vì 215,938) | `reports/ragas_baseline.md` Caveats; ca Debug thứ ba. Lỗi này thấy ở baseline vector thuần và không lặp lại ở `ragas_retrieval_v2`, nhưng chưa có kiểm tra ngữ nghĩa nào |
| Câu hỏi tiếng Việt (không phải tiếng Anh) | embedding chỉ tiếng Anh nên câu gốc retrieve kém (câu rủi ro xuất khẩu NVIDIA: không có chunk Item 1A, distance 0.71-0.74 so với 0.27). Giảm thiểu: `search_query` tiếng Anh do classifier viết lại. Còn yếu với các câu về bảng Item 8 (ví dụ "Lợi nhuận ròng của Microsoft") | `tests/test_graph_routing.py` (fake), `tests/test_llm_smoke.py::test_graph_live_vietnamese_export_control_hits_item_1a` |
| LLM viết `[A, B]` trong một cặp ngoặc | `MARKER_RE` chỉ nhận một chunk id mỗi cặp, nên không có citation hợp lệ, thử lại, rồi `INSUFFICIENT_TEXT` với `error = missing_citation` | chưa có test cho định dạng này |
| Câu hỏi so sánh nhiều công ty | `retrieve` lấy top-5 cho mỗi ticker (`chains.py:122-126`), nhưng chunk đúng vẫn có thể thiếu (q24: recall 0.0 ở vector thuần, 0.5 ở cấu hình hiện tại, chỉ từ phía Microsoft; precision 0.0 ở cả hai) | `reports/ragas_baseline_dense.json`, `reports/ragas_retrieval_v2.json` |

### "Bạn đã kiểm chứng bằng cách nào"

- `uv run pytest -q`: 182 passed, 15 skipped (15 integration test cần `RUN_INTEGRATION=1`). `tests/test_retrieval_integration.py` chạy được chỉ với model cục bộ: 2 passed, 1 xfailed (q05 là miss đã biết). Với router bật, các integration test khác cũng chạy được; lần chạy gần nhất ghi nhận 14 passed và 1 xfailed.
- Bộ đo truy hồi: 25 câu có nhóm bằng chứng viết tay, không gọi LLM, chạy bằng `uv run python -m eval.retrieval_eval --label ...`. Kết quả từng giai đoạn ở `reports/retrieval_experiments.md`: any@5 48% lên 64%, all@5 28% lên 36%, mục tiêu 70% và 40% chưa đạt. Bộ đo chỉ có 25 câu, mỗi câu là 4 điểm phần trăm.
- RAGAS: `reports/ragas_baseline.md` (vector thuần, đã chú giải), `reports/ragas_baseline_rerun.md` (chạy lại, để đo nhiễu) và `reports/ragas_retrieval_v2.md` (cấu hình hiện tại). 25 câu chấm bằng RAGAS, 30 câu kiểm intent (30/30), kèm Caveats (judge cùng model với app; relevancy 0.0 ở các câu hedge là artifact của RAGAS). Chênh lệch giữa các cấu hình nằm trong nhiễu 0.02 đến 0.03 ở faithfulness và recall, nên chưa kết luận được.
- Ground truth do người viết tay từ 10-K, 30 câu (`eval/golden_set.json`).
- Trace LangSmith: `docs/evidence/links.md` và `docs/evidence/trace_tree.md` (không có ảnh chụp màn hình vì UI cần đăng nhập; trace tree lấy qua API).
- Replay offline: `uv run finassist ask x --replay runs/demo.json`.
