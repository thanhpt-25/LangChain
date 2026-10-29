# Bước tiếp theo: những gì còn lại sau đợt cải thiện truy hồi

Đợt cải thiện truy hồi đã xong và có kết quả đo được. Sau bước viết lại truy vấn cho câu `both`, hai mục tiêu của bộ đo (any@5 70% và all@5 40%) đã đạt, nhưng với biên rất mỏng. Tài liệu này ghi lại kết quả, các hướng còn lại mà thí nghiệm gợi ra, và các hạng mục tồn đọng cũ còn áp dụng. Danh sách giới hạn của hệ thống nằm ở [`ARCHITECTURE.md`](ARCHITECTURE.md) mục 8. Số liệu từng giai đoạn nằm ở [`../reports/retrieval_experiments.md`](../reports/retrieval_experiments.md).

## Kết quả của đợt truy hồi

Cấu hình cuối là vector top-30 từ Chroma, cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` chấm lại, 5 chunk mỗi ticker gửi cho LLM, và câu `both` truy hồi bằng `search_query` tiếng Anh chỉ về phần 10-K do bộ phân loại viết lại (mục 4.4 của `ARCHITECTURE.md`). Hybrid BM25 với RRF, chunk theo hàng bảng (có hoặc không có tiền tố embed) và `bge-reranker-base` đã được cài và đo, rồi không được chọn. Mã của ba thứ đầu vẫn nằm trong repo dưới các biến môi trường tắt mặc định.

Bộ đo truy hồi (25 câu, không gọi LLM):

| Chỉ số | Vector thuần | Reranker, chưa viết lại | Cấu hình cuối (có viết lại cho `both`) | Mục tiêu | Kết quả |
|---|---|---|---|---|---|
| any@5 | 48% | 64% | 72% | 70% | đạt, hơn 2 điểm (18 trên 25 câu) |
| all@5 | 28% | 36% | 40% | 40% | đạt vừa đúng ngưỡng (10 trên 25 câu) |
| MRR | 0,371 | 0,376 | 0,404 | không đặt | tăng nhẹ |
| any@5 nhóm combined (5 câu) | 0% | 40% | 80% | không đặt | tăng 2 câu |
| độ trễ p50 mỗi lần truy hồi | 12 ms | khoảng 134 đến 149 ms | khoảng 171 đến 193 ms, dao động theo tải máy | thêm không quá 3 giây | đạt |

RAGAS đầy đủ trên cùng 25 câu, ba lần chạy: `ragas_baseline_dense` (baseline đã chú giải, vector thuần), `ragas_baseline_rerun` (chạy lại vector thuần) và `ragas_retrieval_v2` (cấu hình cuối).

| Chỉ số | baseline_dense | baseline_rerun | retrieval_v2 |
|---|---|---|---|
| faithfulness | 0,939 | 0,961 | 0,939 |
| context precision | 0,309 | 0,309 | 0,342 |
| context recall | 0,387 | 0,420 | 0,400 |
| answer relevancy | 0,687 | 0,665 | 0,766 |

Cách đọc đúng các số này. Precision tăng và relevancy tăng, phần lớn relevancy đến từ việc ít câu bị từ chối hơn. Faithfulness và recall không đổi trong phạm vi nhiễu: hai lần chạy vector thuần lệch nhau 0,02 đến 0,03, nên các chênh lệch ở RAGAS chưa đủ để kết luận. Về từng câu, q03 và q21 nay lấy được bằng chứng và trả lời đúng (209,586 và 416,161), q04 tệ đi (recall từ 1,0 xuống 0,0), còn q05, q04 và q24 vẫn thiếu bằng chứng trong ngữ cảnh. Phần lợi chắc chắn nhất là bộ đo truy hồi: +16 điểm any@5 trên 25 câu là 4 câu, một con số mỏng.

Bảy câu còn trượt ở top-5 trong cấu hình cuối (đã có bản viết lại cho câu `both`), kèm hạng của từng nhóm bằng chứng: q04 [7], q05 [12], q10 [13], q11 [None, None], q15 [7, None], q16 [15, 13, 20], q22 [None, None]. `None` nghĩa là không có chunk bằng chứng nào trong 30 ứng viên. Trước khi viết lại có chín câu: q24 [10, 12] và q25 [None, None] đã vào top-5.

### Viết lại truy vấn cho câu `both`: đã làm

Hướng này là hướng đầu tiên của danh sách cũ. Quy tắc nay là: câu `both` (sau khi lọc ticker) và câu không phải tiếng Anh truy hồi bằng `search_query` tiếng Anh của bộ phân loại; mọi câu khác tìm bằng câu hỏi gốc; `generate_answer` luôn nhận câu hỏi gốc. Chi tiết ở mục 4.4 của `ARCHITECTURE.md`, thí nghiệm và quyết định ở cuối `../reports/retrieval_experiments.md`.

Bộ đo truy hồi, control và bản viết lại (25 câu, không gọi LLM): any@5 64% lên 72%, all@5 36% lên 40%, MRR 0,376 lên 0,404; nhóm combined any@5 40% lên 80% và all@5 40% lên 60%. Hạng đổi ở q21 (5 lên 2), q24 ([10, 12] lên [4, 4]) và q25 ([None, None] lên [9, 4]); q22 và q23 không đổi, và 20 câu không phải `both` trùng khít với control. Bản phát lại từ `reports/query_rewrites.json` cho đúng số liệu này.

RAGAS đầy đủ, bốn lần chạy (hai lần cũ, hai lần có bản viết lại):

| Chỉ số | retrieval_v2 | demo_check | both_rewrite_1 | both_rewrite_2 |
|---|---|---|---|---|
| faithfulness | 0,939 | 0,933 | 0,952 | 0,929 |
| context precision | 0,342 | 0,342 | 0,377 | 0,372 |
| context recall | 0,400 | 0,413 | 0,420 | 0,460 |
| answer relevancy | 0,766 | 0,770 | 0,772 | 0,799 |
| intent accuracy (30 câu) | 30/30 | 30/30 | 30/30 | 30/30 |
| numeric sanity | 7/10 | 7/10 | 7/10 | 7/10 |

Nhóm combined: recall từ 0,267 và 0,200 lên 0,333 và 0,533, precision từ 0,107 lên 0,282 và 0,257.

Kết luận trung thực. Ba tiêu chí của spec đều đạt và hai mục tiêu của bộ đo nay đạt, nhưng chúng đạt với biên rất mỏng. any@5 hơn ngưỡng 2 điểm, all@5 bằng đúng ngưỡng, và cả hai nhờ 5 câu `both` (hai câu đổi hạng đủ để đổi any@5, một câu đổi all@5). Nhiễu giữa hai lần chạy của cấu hình mới (recall 0,04, faithfulness 0,023) lớn hơn của cấu hình cũ, nên trong RAGAS chỉ precision (cao hơn khoảng 0,03) là thay đổi thật; recall (+0,03) và relevancy (+0,02) nằm trong nhiễu. Ở mức từng câu, q24 từ "không so sánh được" sang nêu cả hai doanh thu là cải thiện thật. q22 vẫn thiếu bằng chứng và q23 có recall 0,0 ở mọi lần chạy vì chunk vàng `NVDA-2026-7-012` không được lấy về. Bản viết lại do LLM tạo nên có thể khác giữa các lần chạy; bảng đã lưu chỉ phát lại một lần.

## Các hướng còn lại

Mỗi hướng dưới đây bám vào một quan sát cụ thể. Cách đo luôn giống nhau: chạy lại `uv run python -m eval.retrieval_eval --label "..."` (thêm `--rewrites reports/query_rewrites.json` để dùng bản viết lại đã lưu) rồi một lần RAGAS đầy đủ (`uv run python -m eval.run_ragas --tag <tên>`), và so với các bảng ở trên. Khi so RAGAS, chênh lệch dưới 0,03 ở faithfulness hoặc recall nằm trong nhiễu đã đo (với cấu hình có bản viết lại, nhiễu của recall là 0,04), nên cần chạy mỗi cấu hình ít nhất hai lần.

| Hướng | Quan sát dẫn tới nó | Mục tiêu đo được |
|---|---|---|
| Truy vấn gần nội dung bảng hơn cho câu số liệu (mở rộng truy vấn hoặc HyDE) | Chunk bằng chứng duy nhất của q10 nằm ở hạng 13 sau khi chấm lại. Ở kho theo hàng bảng nó rơi hẳn ra ngoài 30 ứng viên của vector, nên việc thay đổi định dạng chunk không giúp được. Tăng ứng viên lên 50 làm any@5 giảm từ 64% xuống 60% | q10 vào top-5 mà không làm hỏng câu nào đang đạt; any@5 không dưới 72% |
| Giảm tác hại của reranker lên bằng chứng thứ hai | MiniLM làm all@5 của nhóm risk giảm từ 10% xuống 0% vì đẩy chunk bằng chứng thứ hai ra khỏi top-5, và ở q04 chunk `MSFT-2026-7-020`, có mặt trong top-5 của vector thuần ở lần chạy RAGAS baseline, bị thay bằng `7-017`. Chunk `7-020` chứa `Total Revenue $ 331,839` và theo bộ đo bằng chứng của q04 nay ở hạng 7. Hướng thử: giữ đa dạng sau khi chấm lại (ví dụ MMR) hoặc trộn thứ hạng của vector và của reranker | all@5 nhóm risk không dưới 10%, q04 có bằng chứng trong top-5, any@5 chung không dưới 72% và all@5 không dưới 40% |
| Reranker hợp với bảng hơn, hoặc một reranker tinh chỉnh trên 10-K | `bge-reranker-base` có MRR cao nhất (0,444) nhưng any@5 chỉ 56% và p50 809 ms. Cross-encoder `ms-marco` được huấn luyện trên văn bản thường, còn chunk của ta phần lớn là con số | any@5 trên 72% với độ trễ thêm không quá 3 giây ở p50. Cần bộ đo lớn hơn 25 câu trước khi tin kết quả |
| Gửi nhiều hơn 5 chunk mỗi ticker cho LLM | any@10 là 80% và any@20 là 92%, so với any@5 là 72% (có bản viết lại). Biến `FINASSIST_TOP_K` đã có sẵn nên thử không cần sửa mã | RAGAS recall tăng quá nhiễu 0,03, faithfulness không giảm quá 0,03, và thời gian cùng chi phí token được ghi lại |
| Kiểm tra ngữ nghĩa của con số (grounding) | `ground_numbers` chỉ kiểm tra chuỗi số có trong chunk. Ở baseline q10 và q20 điền `revenue = 139297` từ một bảng segment không có tiêu đề cột (doanh thu thật 215,938), và faithfulness vẫn chấm 1,0 và 0,941. Ở lần chạy `retrieval_v2` hai câu này không lặp lại lỗi (q10 hedge, q20 không lấy chunk đó), nhưng điều đó chỉ do truy hồi đổi, phép kiểm tra vẫn như cũ | Một test cho chunk bảng không tiêu đề: số lấy từ cột không được nhận là `revenue` khi nhãn dòng hoặc cột không khớp. Chạy lại q10 và q20 trên kho cũ cho `revenue = null` hoặc 215,938 |
| Bộ đo lớn hơn (tăng độ ưu tiên) | 25 câu, mỗi câu là 4 điểm phần trăm; các câu này cũng dùng chung với RAGAS nên có nguy cơ overfit. Hai mục tiêu nay đạt nhờ 5 câu `both` (any@5 hơn ngưỡng 2 điểm, all@5 vừa đúng ngưỡng), nên mất một câu là về 68% và 36%. Mọi kết luận ở trên chỉ nói về 25 câu này | Thêm ít nhất 25 câu mới viết tay, trong đó có nhiều câu `both` (ít nhất 10), đo lại control, bản viết lại và vector thuần trên cả hai bộ. Chạy `--classify` nhiều lần để biết bản viết lại biến thiên bao nhiêu |

## Hạng mục tồn đọng cũ còn áp dụng

- Cache của router làm `CLASSIFY_SYSTEM` khó sửa an toàn. Router lưu cache phản hồi của bộ phân loại theo tin nhắn người dùng, nên khi đổi prompt hệ thống, phản hồi cũ vẫn được trả mà không báo gì. Điều này đã xảy ra khi thêm luật viết lại, và chỉ hết khi tin nhắn người dùng đổi thành `QUESTION (v2)` (`src/finassist/chains.py`, hàm `classify`). Hiện chỉ có quy ước ghi trong tài liệu là phải tăng hậu tố này mỗi lần sửa prompt. Việc cần làm: gắn mã băm ngắn của `CLASSIFY_SYSTEM` vào tin nhắn người dùng (ví dụ `QUESTION (<hash>)`) để cache tự đổi theo prompt, kèm một unit test kiểm tra rằng sửa prompt làm tin nhắn đổi. Kiểm chứng: sửa một chữ trong prompt rồi chạy `tests/test_both_rewrite.py` với router phải thấy kết quả mới.

- MSFT diluted EPS: lần chạy live đầu tiên cho `eps = null` vì retrieval miss. Chưa đo lại riêng câu này với cấu hình mới.
- Numeric sanity là 7/10 ở `retrieval_v2`, bằng baseline.
- Câu hỏi tiếng Việt: đã giảm thiểu bằng `search_query` tiếng Anh, nhưng embedding và reranker đều chỉ có tiếng Anh, và câu về bảng Item 8 ("Lợi nhuận ròng của Microsoft") vẫn dễ retrieve hụt. Hướng thử: embedding đa ngôn ngữ.
- `answer_relevancy = 0.0` với câu từ chối hoặc hedge là artifact của RAGAS. Số câu như vậy đã giảm ở `retrieval_v2`, nhưng chưa có cách đo khác, nên một phần mức tăng relevancy là do ít từ chối hơn chứ chưa chắc câu trả lời hay hơn.
- Judge cùng model với app (tự chấm): thử judge khác model nếu có.
- LLM chạy ở `temperature=0` (`src/finassist/chains.py:50`), nên lần thử lại trong `generate_answer` có thể cho đúng câu trả lời cũ. Chưa có phép đo cho thấy lần thử thứ hai cải thiện được gì.
- Ảnh chụp trace LangSmith (`docs/evidence/trace_tree.png`) chưa có, vì UI cần đăng nhập. Người có quyền truy cập có thể mở URL trong `docs/evidence/links.md` và chụp.

## Đã làm trong đợt gia cố

Bốn hạng mục nhỏ từ danh sách tồn đọng cũ đã xong, mỗi cái có test viết trước.

- Citation nhiều id: `postprocess.marker_ids` đọc mọi cặp ngoặc vuông mà sau khi bỏ dấu phẩy, chấm phẩy, khoảng trắng, "và" và "and" chỉ còn chunk id. `[A, B]`, `[A,B]`, `[A; B]` và `[A][B]` đều cho hai citation, còn `[see A]` và ghi chú "price unavailable" không cho citation nào. `extract_citations` và `chains.structure` dùng hàm này. `MARKER_RE` vẫn còn cho code cũ.
- Timeout yfinance: biến `FINASSIST_PRICE_TIMEOUT` (giây, mặc định 8). Lời gọi live chạy trong một thread daemon và hết giờ thì ném `TimeoutError`, nên luồng cũ lấy giá cache (`stale`) hoặc báo `PriceUnavailable`. `get_stock_price(..., timeout=None)` giữ hành vi cũ.
- Kho rỗng: `make_retriever` ném `EmptyStoreError` kèm lệnh `uv run finassist ingest` khi `store.count()` bằng 0. Lỗi này nổi lên ở câu hỏi filing đầu tiên, còn câu hỏi chỉ về giá vẫn chạy. CLI in thông báo đó mà không kèm gợi ý router, Streamlit chỉ hiện thông báo lỗi, không nhắc ô Replay.
- Test lời khuyên đầu tư: `tests/test_advice_refusal.py` có 14 cách hỏi (tiếng Anh và tiếng Việt) và 9 câu đối chứng. Chạy thật với router (`RUN_INTEGRATION=1`), classifier từ chối cả 14 và giữ đúng intent của cả 9 câu đối chứng, nên `CLASSIFY_SYSTEM` không phải sửa. Một test không cần router kiểm tra đồ thị từ chối mà không truy hồi hay gọi giá, với từng câu trong 14 câu.

Kiểm thử sau đợt gia cố đó cho 248 passed, 38 skipped. Sau đợt viết lại truy vấn, `uv run pytest -q` cho 266 passed, 46 skipped.
