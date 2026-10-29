# Bước tiếp theo: những gì còn lại sau đợt cải thiện truy hồi

Đợt cải thiện truy hồi đã xong và có kết quả đo được, nhưng chưa đạt mục tiêu. Tài liệu này ghi lại kết quả, các hướng còn lại mà thí nghiệm gợi ra, và các hạng mục tồn đọng cũ còn áp dụng. Danh sách giới hạn của hệ thống nằm ở [`ARCHITECTURE.md`](ARCHITECTURE.md) mục 8. Số liệu từng giai đoạn nằm ở [`../reports/retrieval_experiments.md`](../reports/retrieval_experiments.md).

## Kết quả của đợt truy hồi

Cấu hình cuối là vector top-30 từ Chroma, cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` chấm lại, 5 chunk mỗi ticker gửi cho LLM. Hybrid BM25 với RRF, chunk theo hàng bảng (có hoặc không có tiền tố embed) và `bge-reranker-base` đã được cài và đo, rồi không được chọn. Mã của ba thứ đầu vẫn nằm trong repo dưới các biến môi trường tắt mặc định.

Bộ đo truy hồi (25 câu, không gọi LLM):

| Chỉ số | Vector thuần | Cấu hình cuối | Mục tiêu | Kết quả |
|---|---|---|---|---|
| any@5 | 48% | 64% | 70% | chưa đạt, thiếu 2 câu |
| all@5 | 28% | 36% | 40% | chưa đạt, thiếu 1 câu |
| MRR | 0,371 | 0,376 | không đặt | gần như không đổi |
| độ trễ p50 mỗi lần truy hồi | 12 ms | khoảng 134 đến 149 ms | thêm không quá 3 giây | đạt |

RAGAS đầy đủ trên cùng 25 câu, ba lần chạy: `ragas_baseline_dense` (baseline đã chú giải, vector thuần), `ragas_baseline_rerun` (chạy lại vector thuần) và `ragas_retrieval_v2` (cấu hình cuối).

| Chỉ số | baseline_dense | baseline_rerun | retrieval_v2 |
|---|---|---|---|
| faithfulness | 0,939 | 0,961 | 0,939 |
| context precision | 0,309 | 0,309 | 0,342 |
| context recall | 0,387 | 0,420 | 0,400 |
| answer relevancy | 0,687 | 0,665 | 0,766 |

Cách đọc đúng các số này. Precision tăng và relevancy tăng, phần lớn relevancy đến từ việc ít câu bị từ chối hơn. Faithfulness và recall không đổi trong phạm vi nhiễu: hai lần chạy vector thuần lệch nhau 0,02 đến 0,03, nên các chênh lệch ở RAGAS chưa đủ để kết luận. Về từng câu, q03 và q21 nay lấy được bằng chứng và trả lời đúng (209,586 và 416,161), q04 tệ đi (recall từ 1,0 xuống 0,0), còn q05, q04 và q24 vẫn thiếu bằng chứng trong ngữ cảnh. Phần lợi chắc chắn nhất là bộ đo truy hồi: +16 điểm any@5 trên 25 câu là 4 câu, một con số mỏng.

Chín câu còn trượt ở top-5 trong cấu hình cuối, kèm hạng của từng nhóm bằng chứng: q04 [7], q05 [12], q10 [13], q11 [None, None], q15 [7, None], q16 [15, 13, 20], q22 [None, None], q24 [10, 12], q25 [None, None]. `None` nghĩa là không có chunk bằng chứng nào trong 30 ứng viên.

## Các hướng còn lại

Mỗi hướng dưới đây bám vào một quan sát cụ thể. Cách đo luôn giống nhau: chạy lại `uv run python -m eval.retrieval_eval --label "..."` rồi một lần RAGAS đầy đủ (`uv run python -m eval.run_ragas --tag <tên>`), và so với bảng ở trên. Khi so RAGAS, chênh lệch dưới 0,03 ở faithfulness hoặc recall nằm trong nhiễu đã đo, nên cần chạy mỗi cấu hình ít nhất hai lần.

| Hướng | Quan sát dẫn tới nó | Mục tiêu đo được |
|---|---|---|
| Bỏ phần giá khỏi truy vấn truy hồi của câu `both` (ví dụ mở rộng `search_query` của bộ phân loại để nó chỉ chứa phần nói về báo cáo) | Nhóm combined có any@5 40% và recall RAGAS 0,267. Ở vector thuần nhóm này 0%, vì từ ngữ về giá kéo vector truy vấn về các chunk định giá. q22 và q25 không có chunk bằng chứng nào trong 30 ứng viên | any@5 nhóm combined từ 40% lên ít nhất 60%, recall RAGAS nhóm combined cao hơn 0,433 (lần chạy lại vector thuần), các nhóm khác không giảm |
| Truy vấn gần nội dung bảng hơn cho câu số liệu (mở rộng truy vấn hoặc HyDE) | Chunk bằng chứng duy nhất của q10 nằm ở hạng 13 sau khi chấm lại. Ở kho theo hàng bảng nó rơi hẳn ra ngoài 30 ứng viên của vector, nên việc thay đổi định dạng chunk không giúp được. Tăng ứng viên lên 50 làm any@5 giảm từ 64% xuống 60% | q10 vào top-5 mà không làm hỏng câu nào đang đạt; any@5 không dưới 64% |
| Giảm tác hại của reranker lên bằng chứng thứ hai | MiniLM làm all@5 của nhóm risk giảm từ 10% xuống 0% vì đẩy chunk bằng chứng thứ hai ra khỏi top-5, và ở q04 chunk `MSFT-2026-7-020`, có mặt trong top-5 của vector thuần ở lần chạy RAGAS baseline, bị thay bằng `7-017`. Chunk `7-020` chứa `Total Revenue $ 331,839` và theo bộ đo bằng chứng của q04 nay ở hạng 7. Hướng thử: giữ đa dạng sau khi chấm lại (ví dụ MMR) hoặc trộn thứ hạng của vector và của reranker | all@5 nhóm risk không dưới 10%, q04 có bằng chứng trong top-5, any@5 chung không dưới 64% |
| Reranker hợp với bảng hơn, hoặc một reranker tinh chỉnh trên 10-K | `bge-reranker-base` có MRR cao nhất (0,444) nhưng any@5 chỉ 56% và p50 809 ms. Cross-encoder `ms-marco` được huấn luyện trên văn bản thường, còn chunk của ta phần lớn là con số | any@5 từ 70% trở lên với độ trễ thêm không quá 3 giây ở p50. Cần bộ đo lớn hơn 25 câu trước khi tin kết quả |
| Gửi nhiều hơn 5 chunk mỗi ticker cho LLM | any@10 là 76% và any@20 là 88%, so với any@5 là 64%. Biến `FINASSIST_TOP_K` đã có sẵn nên thử không cần sửa mã | RAGAS recall tăng quá nhiễu 0,03, faithfulness không giảm quá 0,03, và thời gian cùng chi phí token được ghi lại |
| Kiểm tra ngữ nghĩa của con số (grounding) | `ground_numbers` chỉ kiểm tra chuỗi số có trong chunk. Ở baseline q10 và q20 điền `revenue = 139297` từ một bảng segment không có tiêu đề cột (doanh thu thật 215,938), và faithfulness vẫn chấm 1,0 và 0,941. Ở lần chạy `retrieval_v2` hai câu này không lặp lại lỗi (q10 hedge, q20 không lấy chunk đó), nhưng điều đó chỉ do truy hồi đổi, phép kiểm tra vẫn như cũ | Một test cho chunk bảng không tiêu đề: số lấy từ cột không được nhận là `revenue` khi nhãn dòng hoặc cột không khớp. Chạy lại q10 và q20 trên kho cũ cho `revenue = null` hoặc 215,938 |
| Bộ đo lớn hơn | 25 câu, mỗi câu là 4 điểm phần trăm; các câu này cũng dùng chung với RAGAS nên có nguy cơ overfit. Mọi kết luận ở trên chỉ nói về 25 câu này | Thêm ít nhất 25 câu mới viết tay, đo lại cấu hình cuối và vector thuần trên cả hai bộ |

## Hạng mục tồn đọng cũ còn áp dụng

- MSFT diluted EPS: lần chạy live đầu tiên cho `eps = null` vì retrieval miss. Chưa đo lại riêng câu này với cấu hình mới.
- Numeric sanity là 7/10 ở `retrieval_v2`, bằng baseline.
- `MARKER_RE` chỉ nhận một chunk id mỗi cặp ngoặc vuông: `[A, B]` cho 0 citation, dẫn tới thử lại rồi `INSUFFICIENT_TEXT`. Cần cho phép nhiều id hoặc nhắc rõ trong prompt, kèm test.
- Câu hỏi tiếng Việt: đã giảm thiểu bằng `search_query` tiếng Anh, nhưng embedding và reranker đều chỉ có tiếng Anh, và câu về bảng Item 8 ("Lợi nhuận ròng của Microsoft") vẫn dễ retrieve hụt. Hướng thử: embedding đa ngôn ngữ.
- `answer_relevancy = 0.0` với câu từ chối hoặc hedge là artifact của RAGAS. Số câu như vậy đã giảm ở `retrieval_v2`, nhưng chưa có cách đo khác, nên một phần mức tăng relevancy là do ít từ chối hơn chứ chưa chắc câu trả lời hay hơn.
- Judge cùng model với app (tự chấm): thử judge khác model nếu có.
- Lời gọi yfinance chưa đặt timeout, nên mạng chậm có thể làm `price_tool` đứng lâu.
- LLM chạy ở `temperature=0` (`src/finassist/chains.py:47`), nên lần thử lại trong `generate_answer` có thể cho đúng câu trả lời cũ. Chưa có phép đo cho thấy lần thử thứ hai cải thiện được gì.
- Test cho các cách hỏi lời khuyên đầu tư khác nhau để kiểm tra classifier (hiện chỉ có một smoke test, `tests/test_llm_smoke.py:20`).
- Chroma rỗng (quên chạy `ingest`) hiện trả `INSUFFICIENT_TEXT` mà không báo kho trống.
- Ảnh chụp trace LangSmith (`docs/evidence/trace_tree.png`) chưa có, vì UI cần đăng nhập. Người có quyền truy cập có thể mở URL trong `docs/evidence/links.md` và chụp.
