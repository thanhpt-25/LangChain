# Cải thiện truy hồi 10-K: đặc tả thiết kế

Ngày: 2026-10-29 · Trạng thái: chờ duyệt · Tiếp nối [`docs/NEXT_STEP.md`](../../NEXT_STEP.md)

## 1. Mục tiêu

Hệ thống thường không đưa được chunk chứa đáp án vào top-5, nên câu trả lời hedge hoặc dựa vào chunk sai. Đợt này làm cho chunk đúng xuất hiện trong top-5 chunk gửi tới LLM, và chứng minh điều đó bằng một bộ đo chạy trong vài giây.

Ngoài phạm vi (giữ cho các đợt sau):

- kiểm tra ngữ nghĩa của số trong `ground_numbers` (lỗi q10 và q20, số 139,297);
- các việc nhỏ trong backlog: marker citation nhiều ID, timeout yfinance, test cho câu xin tư vấn đầu tư, embedding đa ngôn ngữ, judge khác model.

Chunk bảng có thể làm giảm lỗi 139,297 như một tác dụng phụ. Đợt này chỉ đo điều đó, không cam kết.

## 2. Số đo hiện tại

Tôi đo thứ hạng của chunk vàng (chunk mà ground truth trích dẫn) trong 100 kết quả đầu của vector search hiện tại, cho 25 câu không phải câu bẫy. Đo bằng embedding cục bộ, không gọi LLM.

| Chunk vàng nằm trong top-k | k=5 | k=10 | k=20 | k=50 |
|---|---|---|---|---|
| Có ít nhất một chunk vàng | 24% | 44% | 64% | 92% |
| Có đủ mọi chunk vàng | 8% | 20% | 44% | 72% |

Với 10 câu số liệu, chunk vàng nằm ở hạng 4 đến 44, thường từ 10 đến 25. Hệ thống tìm được chunk nhưng xếp quá thấp, nên cải thiện thứ hạng (reranker, kết hợp từ khóa) hứa hẹn hơn việc mở rộng kho. Các chunk vàng chỉ là những chunk người viết ground truth đã trích, nên nhóm câu rủi ro có thể còn chunk đúng khác không được tính.

## 3. Thiết kế

### 3.1 Bộ đo hit-rate: `eval/retrieval_eval.py`

Chạy toàn bộ golden set (25 câu không phải câu bẫy) qua bộ truy hồi, không gọi LLM. Với câu tiếng Anh, dùng câu hỏi gốc làm truy vấn, đúng như app làm hiện nay.

Chunk id đổi khi cách chia chunk đổi, nên đáp án không thể là chunk id. Mỗi câu trong `eval/golden_set.json` nhận thêm trường `evidence`: danh sách các nhóm, mỗi nhóm là danh sách chuỗi con phải cùng xuất hiện trong một chunk (so khớp không phân biệt hoa thường, gộp khoảng trắng).

```json
"evidence": [["133,749", "Net income"]]
```

- Một nhóm được thỏa khi có ít nhất một chunk trong top-k chứa mọi chuỗi của nhóm.
- `hit@k` (any) là tỷ lệ câu có ít nhất một nhóm được thỏa.
- `hit@k` (all) là tỷ lệ câu có mọi nhóm được thỏa.
- MRR tính theo hạng của chunk đầu tiên thỏa một nhóm.
- Báo cáo theo nhóm câu (numeric, risk, combined) và độ trễ truy hồi (p50, p95).

Trường `evidence` được viết tay từ filing, như ground truth. Một test kiểm rằng mọi chuỗi bằng chứng xuất hiện trong kho chunk hiện tại, và mọi nhóm có ít nhất một chuỗi.

### 3.2 Bộ truy hồi: `src/finassist/retrieval.py`

Một lớp `Retriever` gom ba tầng sau một hàm `retrieve(query, tickers) -> list[Chunk]`. `graph.py` và `Deps` giữ nguyên chữ ký. `chains.make_deps` gọi `Retriever` thay cho `FilingStore.search`.

Với từng ticker, quy trình là:

1. Vector: top-N chunk từ Chroma (N mặc định 30), có lọc ticker.
2. BM25 (tầng tùy chọn): top-N chunk từ chỉ mục BM25 trong bộ nhớ của ticker đó. Chỉ mục dựng từ chính các chunk trong Chroma khi khởi động (khoảng 1.200 chunk, vài trăm mili giây). Bộ tách từ giữ nguyên số có dấu phẩy như `133,749` thành một token.
3. Trộn: gộp hai danh sách bằng Reciprocal Rank Fusion, điểm là tổng của `1/(60 + hạng)`.
4. Reranker (tầng tùy chọn): cross-encoder cục bộ chấm lại các ứng viên, trả `top_k` chunk (mặc định 5), tức là cùng số chunk gửi tới LLM như hiện nay.

Mỗi tầng bật bằng biến môi trường, mặc định giữ hành vi hiện tại cho tới khi số đo chứng minh tầng đó đáng giữ:

| Biến | Giá trị | Mặc định |
|---|---|---|
| `FINASSIST_RETRIEVAL` | `dense` hoặc `hybrid` | `dense` |
| `FINASSIST_RERANKER` | `none` hoặc tên model cross-encoder | `none` |
| `FINASSIST_CANDIDATES` | số ứng viên mỗi tầng trước khi chấm lại | `30` |

Reranker dùng `sentence_transformers.CrossEncoder`. `sentence-transformers` đã có sẵn trong phụ thuộc của dự án. Thư viện BM25 là `rank_bm25`, một gói thuần Python nhỏ, thêm vào `pyproject.toml`. Các ứng viên model để thử: `cross-encoder/ms-marco-MiniLM-L-6-v2` trước, rồi `BAAI/bge-reranker-base`. Bộ đo chọn model, không chọn theo cảm tính.

Truy vấn cho cả ba tầng là `search_query` của đồ thị: câu hỏi gốc với câu tiếng Anh, bản viết lại tiếng Anh với câu ngôn ngữ khác.

### 3.3 Chunk bảng

`html_to_text` dùng `soup.get_text("\n")`, nên mỗi ô của bảng nằm trên một dòng riêng. Số mất nhãn hàng và tiêu đề cột, ví dụ `Net income`, `133,749`, `101,832` thành ba dòng rời nhau. Đây là giả thuyết về gốc của lỗi trong `NEXT_STEP.md`, và đợt này kiểm chứng nó.

Thay đổi thử nghiệm:

- Xuất mỗi hàng bảng (`<tr>`) thành một dòng, các ô cách nhau bằng ` | `: `Net income | 133,749 | 101,832 | 88,136`. Hàng tiêu đề cột được giữ nguyên như một hàng thường.
- Gắn một tiền tố ngắn (`AAPL FY2025 Item 8`) vào văn bản đem embed. Chỉ mục BM25 không dùng tiền tố, vì BM25 dựng riêng cho từng ticker nên ticker và năm giống hệt nhau ở mọi chunk và không phân biệt được gì. Văn bản lưu trong Chroma và gửi cho LLM giữ nguyên nội dung gốc.
- `finassist ingest` thêm cờ `--rebuild` để xóa collection trước khi nạp. Chunk id đổi khi cách chia đổi, và nạp chồng sẽ để lại các chunk cũ.

Các hàm `split_items`, `chunk_filing` và `chunk_id` giữ nguyên. Số trong hàng có dấu `|` vẫn qua `ground_numbers` vì `_appears` chỉ nhìn các ký tự số và dấu phân cách ngay cạnh.

### 3.4 Thang thí nghiệm

Mỗi tầng chỉ được giữ khi bộ đo cho thấy nó đáng giữ. Ngưỡng là chốt chặn để tránh giữ tầng vô ích, không phải mục tiêu cuối.

| Giai đoạn | Việc | Giữ lại khi |
|---|---|---|
| 0 | Dựng bộ đo, viết `evidence`, ghi số baseline | luôn luôn |
| 1 | Reranker trên vector top-30 | hit@5 (any) tăng ít nhất 15 điểm phần trăm so với giai đoạn 0 |
| 2 | Thêm BM25 và RRF trước reranker | hit@5 (any) hoặc (all) tăng ít nhất 5 điểm so với giai đoạn 1 |
| 3 | Chunk bảng và tiền tố embed (nạp lại với `--rebuild`) | hit@5 (any) tăng ít nhất 5 điểm so với giai đoạn trước, hoặc sửa được ít nhất một trong 5 câu tệ nhất (q05, q04, q21, q24, q03) mà không làm hỏng câu nào khác |
| 4 | Chạy RAGAS đầy đủ một lần | để báo cáo kết quả cuối |

Giai đoạn bị loại vẫn ghi vào báo cáo cùng số đo, để người đọc biết vì sao nó không có mặt.

## 4. Thay đổi trong repo

- Thêm: `src/finassist/retrieval.py`, `eval/retrieval_eval.py`, `tests/test_retrieval.py`, `tests/test_retrieval_eval.py`, `reports/retrieval_experiments.md`.
- Sửa: `src/finassist/chains.py` (dựng `Retriever`), `src/finassist/config.py` (ba biến mới), `src/finassist/ingest.py` (xuất hàng bảng, ở giai đoạn 3), `src/finassist/vectorstore.py` (tiền tố embed, hỗ trợ xóa collection), `src/finassist/cli.py` (`ingest --rebuild`), `eval/golden_set.json` (trường `evidence`), `pyproject.toml` (`rank_bm25`), tài liệu `ARCHITECTURE.md`, `NEXT_STEP.md`, `README.md`.
- Không đổi: `graph.py`, `Deps`, `postprocess.py`, `schemas.py`.

## 5. Kiểm thử

- Unit test không gọi mạng và không tải model: `Retriever` nhận hàm chấm điểm giả và embedding giả (`DeterministicFakeEmbedding`). Kiểm tra RRF trộn đúng thứ tự, bộ tách từ giữ số có dấu phẩy, reranker đổi thứ tự, trả đúng `top_k` mỗi ticker, và cấu hình `dense` cho đúng kết quả của `FilingStore.search`.
- Test bộ đo trên một kho nhỏ dựng sẵn, kiểm các phép đo (any, all, MRR) với dữ liệu tính tay.
- Test dữ liệu: mọi chuỗi `evidence` xuất hiện trong kho chunk hiện tại (chạy khi có `data/raw`).
- Integration test (`RUN_INTEGRATION=1`): đường đầy đủ với model reranker thật trả chunk có bằng chứng cho q05.
- Chạy lại `uv run pytest -q`: 145 test hiện có phải tiếp tục đạt.

## 6. Tiêu chí thành công

Giai đoạn 0 đã cho số baseline thật (`reports/retrieval_experiments.md`): any@5 là 48%, all@5 là 28%, MRR 0,371, p50 12 ms. Các số 24% và 8% ở phép đo cũ thấp hơn vì phép đo cũ đòi đúng chunk id vàng, còn bộ đo mới chấp nhận mọi chunk chứa chuỗi bằng chứng (với câu số liệu có 6 đến 7 chunk như vậy), nên hai bộ số không so sánh trực tiếp được. Mục tiêu được giữ nguyên:

- any@5 từ 48% lên ít nhất 70%, và all@5 từ 28% lên ít nhất 40%;
- độ trễ truy hồi thêm không quá 3 giây ở p50 so với vector thuần;
- RAGAS đầy đủ ở giai đoạn 4: context precision và context recall tăng ở nhóm numeric và combined, faithfulness không giảm quá 0,03, và q05, q04, q21, q24, q03 lấy được chunk có bằng chứng.

Các ngưỡng 70%, 40%, 3 giây và 0,03 là ước lượng của tôi, chưa có căn cứ thực nghiệm. Nếu số baseline cho thấy chúng quá dễ hoặc quá khó thì sửa trước khi bắt đầu giai đoạn 1.

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| `evidence` viết tay có thể quá chặt (đòi một cụm từ mà chunk đúng diễn đạt khác) hoặc quá lỏng | Mỗi nhóm dùng chuỗi ngắn và đặc trưng; test dữ liệu kiểm chuỗi có trong kho; xem lại các câu bộ đo và RAGAS không đồng ý |
| Cross-encoder chậm trên CPU | Đo p50 và p95 ở giai đoạn 1; giảm `FINASSIST_CANDIDATES` nếu cần |
| Tải model reranker lần đầu (từ vài chục MB đến hơn 1 GB tùy model) | Ghi rõ trong README; model được cache sau lần đầu |
| Chunk bảng làm đổi chunk id, làm hỏng các tài liệu trích chunk id cũ (`docs/evidence/`, `runs/demo.json`, báo cáo RAGAS baseline) | Chỉ giữ giai đoạn 3 nếu có lợi; các tài liệu đó là bằng chứng của lần chạy cũ và được ghi rõ là cũ; `runs/demo.json` ghi lại sau khi chốt |
| Điểm RAGAS ở giai đoạn 4 không so sánh trực tiếp được với baseline nếu judge hoặc app thay đổi | Giữ nguyên model, prompt và golden set; chỉ đổi tầng truy hồi |
| Overfit golden set 25 câu | Bộ đo và RAGAS dùng chung một golden set nên kết quả chỉ nói về 25 câu này; ghi rõ trong báo cáo |
