# Bước tiếp theo (tuần 2): cải thiện truy hồi các đoạn báo cáo tài chính

Tài liệu này mô tả cải tiến ưu tiên và các việc còn tồn đọng. Danh sách giới hạn của hệ thống nằm ở [`ARCHITECTURE.md`](ARCHITECTURE.md) mục 8.

## Vấn đề quan sát từ baseline

`reports/ragas_baseline.md`: context precision 0.309 và context recall 0.387 là hai metric thấp nhất, trong khi faithfulness là 0.939 (có thể cao vì judge cùng model với app, xem Caveats). Ở cả 5 case tệ nhất (q05, q04, q21, q24, q03), chunk golden không được retrieve. Ba lỗi numeric sanity (q05, q06, q10) đều bắt đầu từ retrieval miss, nhưng chỉ q05 và q06 kết thúc bằng hedge. Ở q05, `MSFT-2026-8-001` (chứa net income 133,749) không nằm trong top-5 và hệ thống hedge. Còn q10 và q20 (NVDA) trả "total revenue $139,297" từ chunk `NVDA-2026-7-029`, một bảng segment không có tiêu đề cột (doanh thu thật 215,938); số này vẫn qua `ground_numbers` và judge chấm faithfulness 1.0/0.941.

Chunk bảng tài chính gồm chủ yếu chữ số, mỗi số một dòng, mất tiêu đề cột (ví dụ `MSFT-2026-8-000` bắt đầu bằng "ITEM 8. FINANCIAL STATE\nMENTS" rồi các dòng số). Đây mới là giả thuyết về nguyên nhân, và thí nghiệm dưới đây sẽ kiểm chứng nó.

## Cải tiến ưu tiên được chọn

Hybrid retrieval (BM25 cộng vector) kèm reranker, và/hoặc chunk theo bảng giữ tiêu đề cột và nhãn dòng (đưa cả tên công ty, năm tài chính, tên báo cáo vào đầu chunk). Lý do chọn trước: hướng này xử lý nguyên nhân chung của cả 5 case tệ nhất, thay vì vá từng câu.

Điểm xuất phát trong code: `retrieve` ở `src/finassist/chains.py:121-122` gọi `FilingStore.search` (`src/finassist/vectorstore.py:41-52`, chỉ vector); chunking ở `src/finassist/ingest.py:65-74`.

## Mục tiêu có thể đo lường (cùng golden set, cùng metric)

Baseline cần vượt (copy từ `reports/ragas_baseline.md`):

| category | context precision | context recall |
|---|---|---|
| numeric | 0.339 | 0.600 |
| risk | 0.434 | 0.200 |
| combined | 0.000 | 0.333 |
| overall | 0.309 | 0.387 |

Đo lại bằng đúng `uv run python -m eval.run_ragas` trên `eval/golden_set.json` (30 câu, ground truth viết tay), so từng category và 5 case tệ nhất. Tiêu chí thành công đề xuất: cả hai metric tăng ở category `numeric` và `combined`, không giảm faithfulness 0.939 một cách đáng kể, và các câu q05, q04, q21, q24, q03 retrieve được chunk golden của chúng. Vì judge dùng cùng model với app, cần giữ nguyên cấu hình để so sánh công bằng. Có thể thêm kiểm tra hit-rate trực tiếp của chunk id golden, không cần judge.

## Hạng mục tồn đọng đã biết

- MSFT diluted EPS: retrieval miss khiến `eps = null` khi chạy live (ghi nhận khi chạy thử CLI live đầu tiên).
- Numeric sanity 7/10: q05, q06, q10 (MSFT net income/EPS, NVDA data center) bắt đầu từ retrieval miss; q05, q06 hedge, còn q10 trả số sai.
- Giới hạn grounding: `ground_numbers` chỉ kiểm tra chuỗi số có trong chunk đã retrieve, không kiểm tra nghĩa (q10/q20 điền `revenue = 139297` từ bảng segment không tiêu đề, thật là 215,938). Cần kiểm tra ngữ nghĩa (nhãn dòng và cột của số) và/hoặc chunk bảng giữ tiêu đề. Faithfulness của judge cùng model không bắt được lỗi này.
- `MARKER_RE` chỉ nhận một chunk id mỗi cặp ngoặc vuông: `[A, B]` cho 0 citation, dẫn tới thử lại rồi `INSUFFICIENT_TEXT`. Cần cho phép nhiều id hoặc nhắc rõ trong prompt, kèm test.
- Câu hỏi tiếng Việt: đã giảm thiểu bằng `search_query` tiếng Anh, nhưng embedding vẫn chỉ tiếng Anh; câu về bảng Item 8 ("Lợi nhuận ròng của Microsoft") vẫn retrieve hụt. Hướng thử: embedding đa ngôn ngữ.
- `answer_relevancy = 0.0` ở 6 dòng (q05, q06, q10, q18, q21, q24) là artifact của RAGAS với câu hedge. Cần cách đo khác, hoặc bỏ qua kèm chú thích.
- Judge cùng model với app (tự chấm): thử judge khác model nếu có.
- Lời gọi yfinance chưa đặt timeout, nên mạng chậm có thể làm `price_tool` đứng lâu.
- LLM chạy ở `temperature=0` (`src/finassist/chains.py:46`), nên lần thử lại trong `generate_answer` có thể cho đúng câu trả lời cũ. Chưa có phép đo cho thấy lần thử thứ hai cải thiện được gì.
- Test cho các cách hỏi lời khuyên đầu tư khác nhau để kiểm tra classifier (hiện chỉ có một smoke test, `tests/test_llm_smoke.py:20`).
- Ảnh chụp trace LangSmith (`docs/evidence/trace_tree.png`) chưa có, vì UI cần đăng nhập. Người có quyền truy cập có thể mở URL trong `docs/evidence/links.md` và chụp.
