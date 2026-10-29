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
