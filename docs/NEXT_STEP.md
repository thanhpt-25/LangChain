# NEXT STEP (tuần 2): cải thiện retrieval cho chunk báo cáo tài chính

## Vấn đề (từ baseline thật)

`reports/ragas_baseline.md`: context precision 0.309 và context recall 0.387 là hai metric thấp nhất, trong khi faithfulness là 0.939 (có thể cao vì judge cùng model với app, xem Caveats). Ở cả 5 case tệ nhất (q05, q04, q21, q24, q03), chunk golden không được retrieve. Ba lỗi numeric sanity (q05, q06, q10) cùng nguyên nhân. Ví dụ q05: `MSFT-2026-8-001` (chứa net income 133,749) không nằm trong top-5; hệ thống hedge thay vì bịa số.

Chunk bảng tài chính gồm chủ yếu chữ số, mỗi số một dòng, mất tiêu đề cột (ví dụ `MSFT-2026-8-000` bắt đầu bằng "ITEM 8. FINANCIAL STATE\nMENTS" rồi các dòng số). Đó là giả thuyết về nguyên nhân; thí nghiệm dưới đây sẽ kiểm chứng nó.

## Cải tiến duy nhất được chọn

**Hybrid retrieval (BM25 + vector) kèm reranker**, và/hoặc **chunk theo bảng giữ tiêu đề cột và nhãn dòng** (đưa cả tên công ty, năm tài chính, tên báo cáo vào đầu chunk). Tại sao chọn cái này trước: nó đánh vào nguyên nhân chung của 5/5 case tệ nhất, không phải vá từng câu.

Điểm xuất phát trong code: `retrieve` ở `src/finassist/chains.py:118-119` gọi `FilingStore.search` (`src/finassist/vectorstore.py:41-52`, chỉ vector); chunking ở `src/finassist/ingest.py:65-74`.

## Mục tiêu đo được (cùng golden set, cùng metric)

Baseline cần vượt (copy từ `reports/ragas_baseline.md`):

| category | context precision | context recall |
|---|---|---|
| numeric | 0.339 | 0.600 |
| risk | 0.434 | 0.200 |
| combined | 0.000 | 0.333 |
| overall | 0.309 | 0.387 |

Đo lại bằng đúng `uv run python -m eval.run_ragas` trên `eval/golden_set.json` (30 câu, ground truth viết tay), so từng category và 5 case tệ nhất. Tiêu chí thành công đề xuất: cả hai metric tăng ở category `numeric` và `combined`, không giảm faithfulness 0.939 một cách đáng kể, và các câu q05, q04, q21, q24, q03 retrieve được chunk golden của chúng. Lưu ý đo bằng judge cùng model với app nên cần giữ nguyên cấu hình để so sánh công bằng; cân nhắc thêm kiểm tra hit-rate trực tiếp của chunk id golden (không cần judge).

## Backlog (đã biết, chưa làm)

- MSFT diluted EPS: retrieval miss khiến `eps = null` khi chạy live (ghi nhận ở Task 9).
- Numeric sanity 7/10: q05, q06, q10 (MSFT net income/EPS, NVDA data center) là retrieval miss.
- `answer_relevancy = 0.0` ở 6 dòng (q05, q06, q10, q18, q21, q24) là artifact của RAGAS với câu hedge, cần cách đo khác hoặc bỏ qua có chú thích.
- Judge cùng model với app (self-grading): thử judge khác model nếu có.
- Test cho các cách hỏi lời khuyên đầu tư khác nhau để kiểm tra classifier (hiện chỉ có một smoke test, `tests/test_llm_smoke.py:20`).
- Ảnh chụp trace LangSmith (`docs/evidence/trace_tree.png`) chưa có vì UI cần đăng nhập.
