# Thí nghiệm truy hồi

Mỗi mục dưới đây là một lần chạy `eval.retrieval_eval` trên 25 câu có `evidence` trong `eval/golden_set.json`. Không có lệnh gọi LLM. any@K là tỷ lệ câu có ít nhất một nhóm bằng chứng nằm trong top-K của đúng ticker; all@K là tỷ lệ câu có mọi nhóm bằng chứng nằm trong top-K.

Quy tắc giữ hay bỏ một tầng nằm trong `docs/superpowers/specs/2026-10-29-retrieval-improvement-design.md` mục 3.4.

Hai lưu ý khi đọc số. Với câu hỏi số liệu, một "hit" nghĩa là có một chunk nêu cùng sự kiện được truy hồi, không phải đúng chunk mà câu trả lời trích dẫn: có 6 đến 7 chunk thỏa bằng chứng của q01, q05, q07 và q08, nên any@K của nhóm numeric dễ đạt hơn so với việc đòi đúng chunk được trích. Bộ đo chỉ có 25 câu và dùng chung các câu này với RAGAS, nên có nguy cơ overfit: kết quả chỉ nói về 25 câu này.

## giai đoạn 0: vector thuần (baseline)

Cấu hình: mode=dense, reranker=none, candidates=30, chroma_dir=chroma

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 48% | 76% | 88% | 28% | 48% | 72% | 0.371 |
| combined | 5 | 0% | 40% | 60% | 0% | 40% | 60% | 0.075 |
| numeric | 10 | 60% | 80% | 100% | 60% | 80% | 100% | 0.398 |
| risk | 10 | 60% | 90% | 90% | 10% | 20% | 50% | 0.492 |

Độ trễ mỗi lần truy hồi (ms): p50 12, p95 17

Đọc baseline và so với phép đo cũ. Bộ đo mới cho any@5 là 48% và all@5 là 28%, cao hơn 24% và 8% của phép đo trước. Lý do gọi tên được là phép đo trước đòi đúng một chunk id vàng, còn bộ đo này chấp nhận mọi chunk chứa chuỗi bằng chứng, và nhóm numeric có nhiều chunk thỏa (60% ở any@5). Nhóm combined là 0% ở top-5 và q11, q22, q25 không có chunk bằng chứng nào trong top-20, đó là chỗ các giai đoạn sau có nhiều dư địa nhất. Độ trễ p50 của vector thuần chỉ 12 ms, nên ngưỡng thêm 3 giây ở p50 vẫn rộng rãi.

Kết luận: các mục tiêu mục 6 (any@5 từ 70%, all@5 từ 40%, thêm không quá 3 giây ở p50) vẫn hợp lý và được giữ nguyên. Hai mục tiêu đòi tăng thêm khoảng 22 và 12 điểm so với baseline này, và any@10 đã đạt 76% nên mục tiêu any@5 là tham vọng nhưng khả thi. Mục 6 của spec chỉ được sửa để ghi số baseline đo được.
## giai đoạn 1a: vector top-30 + MiniLM

Cấu hình: mode=dense, reranker=cross-encoder/ms-marco-MiniLM-L-6-v2, candidates=30, chroma_dir=chroma

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 64% | 76% | 88% | 36% | 44% | 68% | 0.376 |
| combined | 5 | 40% | 60% | 60% | 40% | 40% | 60% | 0.127 |
| numeric | 10 | 70% | 80% | 100% | 70% | 80% | 100% | 0.422 |
| risk | 10 | 70% | 80% | 90% | 0% | 10% | 40% | 0.455 |

Độ trễ mỗi lần truy hồi (ms): p50 149, p95 178

## giai đoạn 1b: vector top-30 + bge-reranker-base

Cấu hình: mode=dense, reranker=BAAI/bge-reranker-base, candidates=30, chroma_dir=chroma

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 56% | 72% | 84% | 36% | 52% | 64% | 0.444 |
| combined | 5 | 40% | 60% | 60% | 40% | 60% | 60% | 0.425 |
| numeric | 10 | 70% | 90% | 100% | 70% | 90% | 100% | 0.536 |
| risk | 10 | 50% | 60% | 80% | 0% | 10% | 30% | 0.363 |

Độ trễ mỗi lần truy hồi (ms): p50 809, p95 1089

## chẩn đoán 1a: MiniLM với candidates=50 (không dùng cho quyết định)

Cấu hình: mode=dense, reranker=cross-encoder/ms-marco-MiniLM-L-6-v2, candidates=50, chroma_dir=chroma

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 60% | 80% | 92% | 36% | 44% | 64% | 0.379 |
| combined | 5 | 40% | 80% | 80% | 40% | 40% | 60% | 0.160 |
| numeric | 10 | 70% | 80% | 100% | 70% | 80% | 100% | 0.420 |
| risk | 10 | 60% | 80% | 90% | 0% | 10% | 30% | 0.447 |

Độ trễ mỗi lần truy hồi (ms): p50 229, p95 352


### Quyết định giai đoạn 1

Giữ MiniLM (cross-encoder/ms-marco-MiniLM-L-6-v2), bỏ bge-reranker-base. Với MiniLM, any@5 từ 48% lên 64% (tăng 16 điểm, qua ngưỡng 15 điểm), all@5 từ 28% lên 36%, MRR từ 0.371 lên 0.376, p50 149 ms và p95 178 ms so với 12 ms và 17 ms của vector thuần, tức thêm khoảng 137 ms ở p50, rất xa ngưỡng 3 giây. bge-reranker-base chỉ đạt any@5 56% (tăng 8 điểm, không qua ngưỡng) dù all@5 cũng 36% và MRR cao nhất (0.444), và chậm hơn: p50 809 ms, p95 1089 ms. Theo từng nhóm, any@5 của MiniLM là combined 0% lên 40%, numeric 60% lên 70%, risk 60% lên 70%. Với bge, combined cũng lên 40%, numeric lên 70%, nhưng risk giảm từ 60% xuống 50%.

Theo --show-misses, MiniLM còn trượt ở top-5 các câu q04, q05, q10 (numeric), q11, q15, q16 (risk), q22, q24, q25 (combined). Các câu combined q21 và q23 nay có bằng chứng trong top-5, còn q22 và q25 vẫn không có chunk bằng chứng nào trong 30 ứng viên (ranks None), và q24 nằm ở hạng 10 và 12, gần top-5 nhưng chưa vào. Giai đoạn trước đã nêu q11, q22, q25 không có chunk nào trong top-20 của vector, và reranker không cứu được vì nó chỉ xếp lại 30 ứng viên đã có. Riêng nhóm risk, all@5 giảm từ 10% xuống 0% với cả hai reranker và any@10 của MiniLM từ 90% xuống 80%, nên reranker đẩy một số chunk bằng chứng thứ hai ra khỏi top-5 và top-10. Số 64% thấp hơn mục tiêu any@5 70%, và mức tăng 16 điểm chỉ hơn ngưỡng 1 điểm trên 25 câu (4 câu), nên kết quả mỏng.

Chạy chẩn đoán MiniLM với candidates=50 (không dùng để quyết định): any@5 60%, any@10 80%, any@20 92%, all@5 36%, MRR 0.379, p50 229 ms, p95 352 ms. Mở rộng tập ứng viên làm any@5 giảm 4 điểm nhưng any@10 và any@20 tăng, và q25 có một nhóm bằng chứng vào hạng 6, nên 30 vẫn là mức hợp lý cho top-5. Giai đoạn 2 tiếp tục với MiniLM làm reranker.
## giai đoạn 2a: hybrid BM25 + vector, không reranker

Cấu hình: mode=hybrid, reranker=none, candidates=30, chroma_dir=chroma

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 56% | 72% | 80% | 32% | 44% | 52% | 0.451 |
| combined | 5 | 0% | 60% | 60% | 0% | 40% | 40% | 0.077 |
| numeric | 10 | 80% | 80% | 80% | 80% | 80% | 80% | 0.528 |
| risk | 10 | 60% | 70% | 90% | 0% | 10% | 30% | 0.561 |

Độ trễ mỗi lần truy hồi (ms): p50 12, p95 13

## giai đoạn 2b: hybrid + reranker đã chọn

Cấu hình: mode=hybrid, reranker=cross-encoder/ms-marco-MiniLM-L-6-v2, candidates=30, chroma_dir=chroma

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 64% | 76% | 92% | 36% | 48% | 68% | 0.379 |
| combined | 5 | 60% | 80% | 80% | 40% | 60% | 80% | 0.177 |
| numeric | 10 | 70% | 80% | 100% | 70% | 80% | 100% | 0.421 |
| risk | 10 | 60% | 70% | 90% | 0% | 10% | 30% | 0.437 |

Độ trễ mỗi lần truy hồi (ms): p50 223, p95 275



### Quyết định giai đoạn 2

Bỏ hybrid BM25, giữ cấu hình của giai đoạn 1 (vector top-30 + MiniLM). Ngưỡng ở spec 3.4 đòi any@5 hoặc all@5 tăng ít nhất 5 điểm so với kết quả tốt nhất của giai đoạn 1 (any@5 64%, all@5 36%). Biến thể tốt hơn trong hai biến thể hybrid là 2b (hybrid + MiniLM): any@5 64% và all@5 36%, tức tăng 0 điểm ở cả hai chỉ số, nên không qua ngưỡng. Biến thể 2a (hybrid, không reranker) có any@5 56% và all@5 32%, thấp hơn giai đoạn 1 lần lượt 8 và 4 điểm. Mỗi câu là 4 điểm, nên 2b hơn giai đoạn 1 đúng 0 câu và 2a kém 2 câu ở any@5.

Số liệu trước và sau (giai đoạn 1a so với 2b), overall: any@5 64% và 64%, all@5 36% và 36%, any@10 76% và 76%, any@20 88% và 92%, MRR 0.376 và 0.379. Độ trễ p50 149 ms và 223 ms, p95 178 ms và 275 ms; thêm khoảng 74 ms ở p50, vẫn rất xa ngưỡng 3 giây. Theo nhóm, any@5 của 2b là combined 60% (giai đoạn 1a: 40%), numeric 70% (70%), risk 60% (70%). Nhóm combined tăng 1 câu và nhóm risk giảm 1 câu, tổng bằng 0. Hybrid không reranker (2a) có MRR cao nhất (0.451), p50 12 ms và numeric 80%, nhưng combined tụt về 0% ở any@5 và any@20 chung chỉ 80%, thấp hơn 88% của vector thuần.

Tác động lên các câu đã theo dõi. Ở 2b, q25 vào top-5 (ở giai đoạn 1a ranks là None, None) nên combined tăng; q14 (risk) rời top-5 (ranks None, 8, None) nên risk giảm. q21 và q23 vẫn có bằng chứng trong top-5 ở 2b nhưng ở 2a cả hai nằm hạng 7. q22 vẫn không có chunk bằng chứng nào ở cả 2a và 2b; q24 vẫn ở hạng 10 và 12. Trong năm câu khó đã biết, q03 và q06 nằm trong top-5 ở mọi cấu hình, còn q05 (hạng 13 ở 2b) và q10 (hạng 13 ở 2b) vẫn trượt, và q11 vẫn không có chunk nào ở cả hai biến thể. Với các câu risk trượt, BM25 không cứu được q11, q15 và q16 (hạng 11, 16, 19 trở lên), và còn làm mất q14. Kết luận cho dấu hiệu từ khóa: các nhóm bằng chứng còn thiếu thường dùng từ diễn đạt khác với câu hỏi, nên khớp từ vựng không bù được.

Khoảng cách tới mục tiêu 70% sau quyết định: cấu hình cuối (vector top-30 + MiniLM) có any@5 64%, thiếu 6 điểm, tương đương 2 câu nữa (18 trên 25 là 72%; 17 trên 25 là 68% vẫn chưa đạt). all@5 36%, thiếu 4 điểm so với mục tiêu 40%, tức 1 câu. Chạy lại cấu hình cuối cho thấy các câu còn trượt ở top-5 và ranks của từng nhóm bằng chứng: q04 [7], q05 [12], q10 [13] (numeric), q11 [None, None], q15 [7, None], q16 [15, 13, 20] (risk), q22 [None, None], q24 [10, 12], q25 [None, None] (combined). Lần chạy lại này cho p50 143 ms và p95 185 ms, gần số 149 ms đã ghi ở mục 1a. Giai đoạn 3 tiếp tục từ cấu hình này; q04, q15, q24 gần top-5 nhất, còn q11, q22, q25 chưa có chunk nào trong 30 ứng viên.
## giai đoạn 3a: chunk bảng + tiền tố embed, vector top-30 + MiniLM (cấu hình tốt nhất)

Cấu hình: mode=dense, reranker=cross-encoder/ms-marco-MiniLM-L-6-v2, candidates=30, chroma_dir=chroma_tables

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 56% | 64% | 84% | 24% | 32% | 60% | 0.327 |
| combined | 5 | 40% | 40% | 60% | 20% | 20% | 40% | 0.157 |
| numeric | 10 | 50% | 60% | 90% | 50% | 60% | 90% | 0.282 |
| risk | 10 | 70% | 80% | 90% | 0% | 10% | 40% | 0.458 |

Độ trễ mỗi lần truy hồi (ms): p50 152, p95 171

## giai đoạn 3b: chunk bảng + tiền tố embed, vector thuần không reranker

Cấu hình: mode=dense, reranker=none, candidates=30, chroma_dir=chroma_tables

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 56% | 72% | 84% | 32% | 36% | 64% | 0.459 |
| combined | 5 | 20% | 40% | 60% | 20% | 20% | 60% | 0.238 |
| numeric | 10 | 60% | 70% | 90% | 60% | 70% | 90% | 0.455 |
| risk | 10 | 70% | 90% | 90% | 10% | 10% | 40% | 0.573 |

Độ trễ mỗi lần truy hồi (ms): p50 8, p95 10

## giai đoạn 3c: chunk bảng, không tiền tố embed, vector top-30 + MiniLM

Cấu hình: mode=dense, reranker=cross-encoder/ms-marco-MiniLM-L-6-v2, candidates=30, chroma_dir=chroma_tables_noprefix

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 52% | 60% | 72% | 24% | 32% | 52% | 0.313 |
| combined | 5 | 20% | 20% | 40% | 20% | 20% | 40% | 0.117 |
| numeric | 10 | 50% | 60% | 70% | 50% | 60% | 70% | 0.269 |
| risk | 10 | 70% | 80% | 90% | 0% | 10% | 40% | 0.455 |

Độ trễ mỗi lần truy hồi (ms): p50 143, p95 161



### Quyết định giai đoạn 3

Bỏ giai đoạn 3 (chunk theo hàng bảng và tiền tố embed) và giữ cấu hình của giai đoạn 1 (vector top-30 + MiniLM). Mã nguồn ở lại vì nó là tùy chọn và có test; các biến FINASSIST_TABLE_ROWS và FINASSIST_EMBED_PREFIX mặc định tắt. Đã xóa data/chroma_tables và data/chroma_tables_noprefix; data/chroma (baseline) không bị đụng tới.

Ngưỡng ở spec 3.4 đòi any@5 tăng ít nhất 5 điểm so với kết quả tốt nhất trước đó (64%), hoặc sửa được ít nhất một câu trong q05, q04, q21, q24, q03 mà không làm hỏng câu khác. Biến thể tốt nhất của giai đoạn 3 là 3a (bảng + tiền tố, vector top-30 + MiniLM): any@5 56%, thấp hơn 64% đúng 8 điểm, tức 2 câu. Trong năm câu khó, q03 và q21 vẫn có bằng chứng trong top-5 như trước, còn q04 [8], q05 [15] và q24 [12, 13] vẫn trượt, nên không câu nào được sửa. Ba câu bị hỏng so với giai đoạn 1: q07 (hạng 20), q09 (hạng 12) và q23 (không có chunk nào trong 30 ứng viên). Một câu mới vào top-5 là q25, nhưng nó không nằm trong danh sách năm câu của ngưỡng.

Số liệu trước và sau (giai đoạn 1a so với 3a), overall: any@5 64% và 56%, all@5 36% và 24%, any@10 76% và 64%, any@20 88% và 84%, MRR 0.376 và 0.327. Theo nhóm, any@5 là combined 40% và 40%, numeric 70% và 50%, risk 70% và 70%. Độ trễ p50 149 ms và 152 ms, p95 178 ms và 171 ms. Số chunk: kho chuẩn 1162 (AAPL 256, MSFT 451, NVDA 455), kho theo hàng bảng 1168 (AAPL 261, MSFT 452, NVDA 455), tức thêm 6 chunk; việc đổi dòng ngắn thành dòng bảng gần như không đổi ranh giới chunk, nhưng nội dung chunk thay đổi nhiều, và nhóm numeric là nhóm chịu thiệt nhất (giảm 2 câu).

Hai biến thể còn lại để tách tác động. 3b (bảng + tiền tố, vector thuần không reranker) có any@5 56%, all@5 32%, MRR 0.459, p50 8 ms; so với giai đoạn 0 (48%, 28%, 0.371) đó là tăng 8 điểm any@5, nhưng baseline đúng để so là 64% của giai đoạn 1 nên không đạt. Nhờ chunk mới, q04 vào top-5 ở 3b mà không cần reranker, nhưng q06, q18 và q23 rời top-5. 3c (bảng, không tiền tố, vector top-30 + MiniLM) có any@5 52%, all@5 24%, MRR 0.313, p50 143 ms, kém 3a 4 điểm ở any@5 và any@20 (72% so với 84%). Như vậy tiền tố embed có đóng góp dương nhỏ khi đã có chunk bảng (thêm 1 câu ở any@5, 3 câu ở any@20), còn bản thân các chunk theo hàng bảng làm giảm kết quả so với văn bản phẳng ở cấu hình tốt nhất. Nếu giữ lại phần nào của giai đoạn 3 cho thử nghiệm sau thì đó là tiền tố, không phải hàng bảng; phép đo này chưa có biến thể chỉ có tiền tố trên văn bản phẳng nên chưa thể kết luận riêng cho tiền tố.

q10 không được cứu. Chunk duy nhất thỏa bằng chứng của q10 là NVDA-2026-15-142, nay chứa dòng "Data Center | 193,737 | 115,186 | 47,525" (hàng và con số cùng một dòng), nhưng nó không vào 30 ứng viên của vector ở cả ba biến thể (ranks [None]), nên reranker không thể xếp lại nó. Ở baseline chunk đó nằm ở hạng 13. Nguyên nhân gần nhất là biểu diễn embedding của một bảng nhiều số vẫn ở xa câu hỏi bằng lời, và định dạng hàng không thay đổi điều đó. q20 (nhiều nhóm bằng chứng, trong đó có "215,938" và "Up 65%") vẫn có bằng chứng trong top-5 ở cả ba biến thể.

Các câu còn trượt ở top-5 trong cấu hình cuối (giai đoạn 1, vector top-30 + MiniLM): q04, q05, q10, q11, q15, q16, q22, q24, q25. Khoảng cách tới mục tiêu: any@5 64% so với 70%, thiếu 6 điểm (2 câu; 18 trên 25 mới đạt); all@5 36% so với 40%, thiếu 4 điểm (1 câu); p50 149 ms, vượt xa ngưỡng độ trễ. Hai mục tiêu chất lượng chưa đạt sau ba giai đoạn.
