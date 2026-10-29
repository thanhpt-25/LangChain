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

Các câu còn trượt ở top-5 trong cấu hình cuối (giai đoạn 1, vector top-30 + MiniLM): q04, q05, q10, q11, q15, q16, q22, q24, q25. Khoảng cách tới mục tiêu: any@5 64% so với 70%, thiếu 6 điểm (2 câu; 18 trên 25 mới đạt); all@5 36% so với 40%, thiếu 4 điểm (1 câu); p50 149 ms, còn xa mới chạm ngưỡng 3 giây. Hai mục tiêu chất lượng chưa đạt sau ba giai đoạn.
## cấu hình mặc định cuối cùng

Cấu hình: mode=dense, reranker=cross-encoder/ms-marco-MiniLM-L-6-v2, candidates=30, chroma_dir=chroma

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 64% | 76% | 88% | 36% | 44% | 68% | 0.376 |
| combined | 5 | 40% | 60% | 60% | 40% | 40% | 60% | 0.127 |
| numeric | 10 | 70% | 80% | 100% | 70% | 80% | 100% | 0.422 |
| risk | 10 | 70% | 80% | 90% | 0% | 10% | 40% | 0.455 |

Độ trễ mỗi lần truy hồi (ms): p50 134, p95 177



## So sánh RAGAS đầy đủ: baseline vector, chạy lại, cấu hình cuối

Ba lần chạy cùng 25 câu có bằng chứng, cùng judge `kr/claude-sonnet-5.5` (trùng với LLM của ứng dụng, nên faithfulness có thể bị thổi cao). `ragas_baseline_dense` là baseline đã chú giải, `ragas_baseline_rerun` là lần chạy lại hệ thống vector thuần, `ragas_retrieval_v2` là cấu hình cuối (vector top-30 + MiniLM). Hai lần chạy vector thuần cho biết nhiễu giữa các lần chạy: faithfulness lệch 0.022, context recall lệch 0.033, answer relevancy lệch 0.022, còn context precision trùng nhau (0.309).

| chỉ số | baseline_dense | baseline_rerun | retrieval_v2 |
|---|---|---|---|
| faithfulness | 0.939 | 0.961 | 0.939 |
| context precision | 0.309 | 0.309 | 0.342 |
| context recall | 0.387 | 0.420 | 0.400 |
| answer relevancy | 0.687 | 0.665 | 0.766 |
| intent accuracy (30 câu) | 30/30 | 30/30 | 30/30 |
| numeric sanity | 7/10 | 6/10 | 7/10 |
| thời gian chạy (giây) | 814 | 953 | 1134 |

Theo nhóm (faithfulness / context precision / context recall / answer relevancy):

| nhóm | baseline_dense | baseline_rerun | retrieval_v2 |
|---|---|---|---|
| combined | 0.903 / 0.000 / 0.333 / 0.561 | 0.966 / 0.000 / 0.433 / 0.585 | 0.922 / 0.107 / 0.267 / 0.768 |
| numeric | 0.926 / 0.339 / 0.600 / 0.615 | 0.967 / 0.339 / 0.600 / 0.532 | 0.911 / 0.409 / 0.600 / 0.609 |
| risk | 0.971 / 0.434 / 0.200 / 0.822 | 0.954 / 0.434 / 0.233 / 0.837 | 0.977 / 0.392 / 0.267 / 0.921 |

Thời gian chạy dài hơn vì reranker (khoảng 140 ms mỗi lần truy hồi) và vì lần chạy khác nhau về tải của router; chênh lệch giữa hai lần chạy vector thuần (139 giây) đã lớn bằng nửa chênh lệch tới v2, nên không đọc nó thành chi phí của reranker.

### Đối chiếu tiêu chí mục 6 của spec

Context precision ở nhóm numeric: đạt. 0.409 so với 0.339 ở cả hai lần chạy trước. Ở nhóm combined: đạt về hướng, 0.107 so với 0.000 ở cả hai lần, nhưng giá trị tuyệt đối thấp và là 1 câu (q21) có điểm khác 0.

Context recall ở nhóm numeric: không đạt. 0.600 bằng cả hai lần chạy trước, không tăng. Ở nhóm combined: không đạt, và còn giảm: 0.267 so với 0.333 (dense) và 0.433 (rerun). Nhóm chỉ có 5 câu nên chênh lệch này có thể là nhiễu, nhưng không có bằng chứng recall tăng. Recall tổng thể 0.400 nằm giữa 0.387 và 0.420, tức không phân biệt được với nhiễu.

Faithfulness không giảm quá 0.03: đạt ở mức tổng thể. 0.939 so với 0.939 và 0.961 (thấp hơn lần chạy lại 0.022). Riêng nhóm numeric faithfulness giảm 0.056 so với lần chạy lại (0.911 so với 0.967) và 0.015 so với dense, nên tiêu chí chỉ đạt khi xét tổng thể; q04 (0.875) và q07 (0.600) là hai câu numeric kéo nhóm xuống.

Năm câu cần có bằng chứng trong ngữ cảnh. q03: đạt, câu trả lời đúng 209,586 và bằng chứng nằm trong top-5. q21: đạt, câu trả lời đúng 416,161. q05: không đạt, bằng chứng ở hạng 12, câu trả lời từ chối nêu số. q04: không đạt và còn tệ hơn baseline, bằng chứng ở hạng 7, câu trả lời nói ngữ cảnh không có tổng doanh thu, trong khi ở lần chạy dense có recall 1.0. q24: không đạt, hai nhóm bằng chứng ở hạng 10 và 12, câu trả lời không so sánh được doanh thu (recall 0.5 chỉ từ phía Microsoft).

Kết quả đo được là câu trả lời liên quan hơn (answer relevancy 0.766 so với 0.687 và 0.665, phần lớn do q18, q21 và nhóm combined không còn bị trả lời từ chối) và precision tốt hơn, nhưng recall không tăng và q04 bị hỏng. Hai lần chạy vector thuần cho thấy nhiễu cỡ 0.02 đến 0.03 ở faithfulness và recall, nên các chênh lệch recall và faithfulness ở trên không đủ để kết luận theo hướng nào.

## Kết luận

Cấu hình cuối cùng là vector top-30 từ Chroma, sắp xếp lại bằng cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, lấy 5 chunk mỗi mã cổ phiếu. Các biến mặc định: `FINASSIST_RETRIEVAL=dense`, `FINASSIST_RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2`, `FINASSIST_CANDIDATES=30`, `FINASSIST_EMBED_PREFIX=0`, `FINASSIST_TABLE_ROWS=0`, kho `data/chroma` không đổi (id chunk giữ nguyên). Chạy lại bộ đo với mặc định mới cho đúng số của giai đoạn 1a: any@5 64%, all@5 36%, MRR 0.376, p50 134 ms.

So với mục tiêu ở spec mục 6: any@5 đạt 64% so với 70%, thiếu 6 điểm (2 câu); all@5 đạt 36% so với 40%, thiếu 4 điểm (1 câu). Hai mục tiêu chất lượng chưa đạt. Mục tiêu độ trễ (thêm không quá 3 giây ở p50) đạt, thêm khoảng 120 đến 140 ms. So với baseline vector thuần (48% và 28%), any@5 tăng 16 điểm và all@5 tăng 8 điểm.

Các tầng đã bỏ và số liệu. bge-reranker-base: any@5 56% (tăng 8 điểm), all@5 36%, p50 809 ms, nhóm risk giảm từ 60% xuống 50%. Hybrid BM25 với RRF: tốt nhất là hybrid + MiniLM với any@5 64% và all@5 36%, tức tăng 0 điểm so với giai đoạn 1; hybrid không reranker có any@5 56% và nhóm combined về 0%. Chunk theo hàng bảng và tiền tố embed: tốt nhất 56% any@5 và 24% all@5, thấp hơn 8 và 12 điểm, làm hỏng q07, q09, q23 và không cứu q05, q10. Mã của hybrid, bảng và tiền tố vẫn nằm trong repo dưới các biến môi trường tắt mặc định.

Các câu còn trượt ở top-5 (any@5): q04 [7], q05 [12], q10 [13], q11 [None, None], q15 [7, None], q16 [15, 13, 20], q22 [None, None], q24 [10, 12], q25 [None, None]. Với RAGAS, q04 còn tệ đi so với baseline vector (recall 1.0 xuống 0.0).

Ý tưởng chưa thử, bám vào những gì các thí nghiệm cho thấy.
1. Tách câu hỏi kết hợp (giá cổ phiếu và doanh thu) trước khi truy hồi, hoặc chỉ truy hồi bằng phần nói về báo cáo. Nhóm combined có any@5 40% và recall RAGAS giảm, và q22, q25 không có chunk bằng chứng nào trong 30 ứng viên, nhiều khả năng vì từ ngữ về giá làm lệch vector truy vấn.
2. Mở rộng truy vấn hoặc dùng HyDE cho các câu số liệu. Chunk bằng chứng của q10 (có nhãn hàng của bảng) nằm ngoài 30 ứng viên ở cả ba biến thể, nên cần một truy vấn gần với nội dung bảng hơn, không phải reranker mạnh hơn.
3. Chạy tiền tố embed riêng trên văn bản phẳng (chưa có biến thể này), và đo lại reranker theo từng nhóm: MiniLM làm all@5 của nhóm risk giảm từ 10% xuống 0% vì đẩy chunk bằng chứng thứ hai ra khỏi top-5, nên có thể thử đảm bảo độ đa dạng giữa các chunk (ví dụ MMR) sau khi sắp xếp lại.
Tất cả đều cần kiểm tra trên bộ đo lớn hơn 25 câu trước khi tin kết quả.
## both-rewrite: control, câu hỏi gốc (cấu hình cuối)

Cấu hình: mode=dense, reranker=cross-encoder/ms-marco-MiniLM-L-6-v2, candidates=30, chroma_dir=chroma, queries=original questions

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 64% | 76% | 88% | 36% | 44% | 68% | 0.376 |
| combined | 5 | 40% | 60% | 60% | 40% | 40% | 60% | 0.127 |
| numeric | 10 | 70% | 80% | 100% | 70% | 80% | 100% | 0.422 |
| risk | 10 | 70% | 80% | 90% | 0% | 10% | 40% | 0.455 |

Độ trễ mỗi lần truy hồi (ms): p50 179, p95 282

## both-rewrite: truy vấn do bộ phân loại viết lại (chạy thật)

Cấu hình: mode=dense, reranker=cross-encoder/ms-marco-MiniLM-L-6-v2, candidates=30, chroma_dir=chroma, queries=classifier rewrite (live)

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 72% | 80% | 92% | 40% | 52% | 72% | 0.404 |
| combined | 5 | 80% | 80% | 80% | 60% | 80% | 80% | 0.267 |
| numeric | 10 | 70% | 80% | 100% | 70% | 80% | 100% | 0.422 |
| risk | 10 | 70% | 80% | 90% | 0% | 10% | 40% | 0.455 |

Độ trễ mỗi lần truy hồi (ms): p50 171, p95 238

## both-rewrite: phát lại bảng đã lưu

Cấu hình: mode=dense, reranker=cross-encoder/ms-marco-MiniLM-L-6-v2, candidates=30, chroma_dir=chroma, queries=classifier rewrite (replay of query_rewrites.json)

| nhóm | n | any@5 | any@10 | any@20 | all@5 | all@10 | all@20 | mrr |
|---|---|---|---|---|---|---|---|---|
| overall | 25 | 72% | 80% | 92% | 40% | 52% | 72% | 0.404 |
| combined | 5 | 80% | 80% | 80% | 60% | 80% | 80% | 0.267 |
| numeric | 10 | 70% | 80% | 100% | 70% | 80% | 100% | 0.422 |
| risk | 10 | 70% | 80% | 90% | 0% | 10% | 40% | 0.455 |

Độ trễ mỗi lần truy hồi (ms): p50 193, p95 249


### Quyết định cho viết lại truy vấn câu both

Giữ. Cổng ở spec 4.4 đạt ở cả hai vế, và lần phát lại bảng đã lưu cho đúng các con số của lần chạy thật.

Số liệu trước và sau (control với câu hỏi gốc, so với bản viết lại do bộ phân loại tạo ra), overall: any@5 64% và 72%, all@5 36% và 40%, any@10 76% và 80%, any@20 88% và 92%, MRR 0.376 và 0.404. Nhóm combined: any@5 40% và 80%, all@5 40% và 60%, any@10 60% và 80%, any@20 60% và 80%, MRR 0.127 và 0.267. Nhóm numeric (any@5 70% và 70%) và risk (70% và 70%) có mọi cột và mọi hạng trùng với control. Độ trễ p50 179 ms và 171 ms, p95 282 ms và 238 ms; hai con số này dao động theo tải máy giữa các lần chạy, nên không đọc ra thành chênh lệch thật.

Cổng: combined any@5 80% so với ngưỡng 60%, đạt; overall any@5 72% so với ngưỡng 64%, đạt; numeric 70% và risk 70% không thấp hơn 70% và 70% của control, đạt. Hai nhóm này gồm 20 câu không phải both nên không đổi theo thiết kế, và các hạng của từng câu trong hai nhóm trùng khít với control, nên số liệu xác nhận điều đó. Bảng intent mà bộ phân loại trả về cho 25 câu có bằng chứng cũng khớp: q01 đến q20 đều là filing, tiếng Anh, và q21 đến q25 đều là both, tiếng Anh, nên không câu nào đổi nhãn.

Hạng của từng nhóm bằng chứng ở q21 đến q25, trước rồi sau.
- q21 (AAPL): hạng 5 lên hạng 2. Đã nằm trong top-5, được xếp cao hơn.
- q22 (MSFT, hai nhóm): [None, None] vẫn [None, None]. Không có chunk bằng chứng nào trong 30 ứng viên, cả trước lẫn sau.
- q23 (NVDA): hạng 3, không đổi.
- q24 (AAPL và MSFT): [10, 12] lên [4, 4]. Đây là câu chuyển từ trượt sang đạt cả any@5 và all@5.
- q25 (NVDA, hai nhóm): [None, None] lên [9, 4]. Nhóm thứ hai vào top-5 nên any@5 đạt, nhóm thứ nhất ở hạng 9 nên all@5 chưa đạt.
Vậy hai câu mới có bằng chứng trong top-5 (q24, q25), một câu cải thiện hạng trong top-5 (q21), một câu không đổi (q23) và một câu vẫn trượt hoàn toàn (q22). Cách tính điểm: any@5 tăng 2 câu và all@5 tăng 1 câu (q24).

Bản viết lại của năm câu, lấy từ reports/query_rewrites.json.
- q21: "What were Apple's total net sales in its most recent fiscal year?"
- q22: "What is Microsoft's biggest competitive risk disclosed in its 10-K?"
- q23: "What was NVIDIA's total revenue in its most recent fiscal year?"
- q24: "Compare Apple's and Microsoft's total revenue in their most recent fiscal year."
- q25: "How do export controls threaten NVIDIA's business according to its 10-K?"
Không bản viết lại nào còn từ về giá (price, P/E, market cap) và không bản nào bỏ mất công ty: q24 giữ cả Apple và Microsoft, ba câu còn lại giữ đúng một công ty của câu gốc. Cũng không bản nào trông sai nghĩa. Chỗ đáng chú ý duy nhất là "last fiscal year" thành "most recent fiscal year", đổi nghĩa không đáng kể. Các câu chỉ nói tới rủi ro (q22, q25) chỉ còn mệnh đề rủi ro nhưng vẫn thêm "10-K" và "disclosed" hoặc "according to its 10-K", và q22 vẫn không tìm được chunk bằng chứng nào, nên lý do của q22 chưa phải từ giá mà nhiều khả năng là vector không đưa được hai đoạn về rào cản gia nhập và đối thủ siêu quy mô vào 30 ứng viên với câu hỏi chung chung như vậy.

Bảng cũng có bản viết lại không rỗng ở 7 câu khác (q07, q12, q13, q15, q16, q17, q18), tất cả là câu filing tiếng Anh. Quy tắc retrieval_query không dùng chúng, các câu đó vẫn truy hồi bằng câu hỏi gốc, và hạng của chúng trùng control, nên đây không phải là lỗi, chỉ là bộ phân loại điền search_query khi không cần.

Khoảng cách tới mục tiêu ở spec mục 6 sau thay đổi: any@5 72% so với 70%, vượt 2 điểm (18 trên 25 là 72%, mục tiêu cần 17,5 câu); all@5 40% so với 40%, vừa đúng (10 trên 25). Hai mục tiêu chất lượng đạt, nhưng chỉ vừa sát ngưỡng, và all@5 sẽ rơi xuống 36% nếu mất một câu.

Mức tin cậy. Mỗi câu combined chiếm 20 điểm trong nhóm combined và 4 điểm trong số liệu chung. Quyết định này dựa trên 5 câu, trong đó 2 câu đổi hạng đủ để đổi any@5 (q24 và q25) và 1 câu đổi all@5 (q24). Đó là mẫu mỏng: một câu lệch là 20 điểm ở combined và 4 điểm ở overall, và việc vượt mục tiêu any@5 là nhờ đúng hai câu đó. Bản viết lại do LLM tạo cũng có thể khác ở lần chạy khác; bảng đã lưu giúp phát lại đúng số liệu này nhưng không chứng minh rằng bộ phân loại sẽ viết lại giống thế mỗi lần. Cần bộ đo lớn hơn 25 câu, nhiều câu both hơn, trước khi coi kết quả là ổn định.
