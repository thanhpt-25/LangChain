# Kịch bản demo (8 phút trình bày + 4 phút hỏi đáp + 3 phút phản hồi)

Tổng cộng 15 phút. Phần demo 8 phút chia làm 4 phần. Trainer có thể hỏi một trong 4 loại: Explain, Predict, Debug, Justify (đáp án sẵn trong `docs/KNOWLEDGE_CHECK.md`).

## 1. Lịch trình trình bày 8 phút

| Thời gian | Phần | Nói và làm gì | Mở sẵn |
|---|---|---|---|
| 0:00-1:30 | 1. Bài toán và bối cảnh tài chính | Nhà đầu tư/analyst muốn trả lời nhanh từ 10-K và giá cổ phiếu, nhưng LLM trần dễ bịa số. Phạm vi: AAPL, MSFT, NVDA. Hệ thống phải (a) có trích dẫn `chunk_id`, (b) lấy giá bằng tool tất định, (c) từ chối câu ngoài phạm vi và tư vấn đầu tư | `README.md` (phần mở đầu) |
| 1:30-3:30 | 2. Kiến trúc và data flow | Chỉ sơ đồ graph trong `docs/ARCHITECTURE.md` mục 3: `classify_intent` rẽ 4 nhánh. Đi qua data flow thật của câu `both` (mục 4.2): `price_tool` 1.67 s, `retrieve` 0.74 s ra 5 chunk `NVDA-2026-1A-126` ... `NVDA-2026-15-020` (trace này ghi trước khi thêm reranker; nay `retrieve` lấy 30 ứng viên từ Chroma rồi cross-encoder chọn 5, thêm khoảng 140 ms, xem mục 4.5). Nhấn 3 quyết định trong mục 7: citation tất định (`postprocess.py:10-22`), `ground_numbers` (`postprocess.py:36`), giá từ tool | `docs/ARCHITECTURE.md` |
| 3:30-6:00 | 3. Chạy trực tiếp | (a) Câu `both`: "What is NVIDIA's current stock price and how do export controls threaten it?" qua Streamlit: câu trả lời, bảng citation, JSON, chunk retrieve. Mở trace LangSmith (URL trong `docs/evidence/links.md`) và chỉ các node `classify_intent`, `price_tool`, `retrieve`, `generate_answer`, `format_output`. (b) Câu ngoài phạm vi: "Should I buy NVIDIA stock now?" cho ra từ chối, trace chỉ có `classify_intent` và `refuse` | Streamlit, tab LangSmith |
| 6:00-8:00 | 4. Đánh giá, một lỗi, bước tiếp | Bộ đo truy hồi 25 câu (`reports/retrieval_experiments.md`): thêm reranker đưa any@5 từ 48% lên 64% và all@5 từ 28% lên 36%, độ trễ p50 từ 12 ms lên khoảng 140 ms. Mục tiêu 70% và 40% chưa đạt, thiếu 2 và 1 câu; hybrid BM25 và chunk theo hàng bảng đã được thử và bỏ vì không giúp. RAGAS (`reports/ragas_retrieval_v2.md`, so với `reports/ragas_baseline.md`): faithfulness 0.939 so với 0.939, precision 0.342 so với 0.309, recall 0.400 so với 0.387, relevancy 0.766 so với 0.687; intent 30/30, numeric sanity 7/10. Hai lần chạy vector thuần lệch nhau 0.02 đến 0.03, nên phần RAGAS chưa kết luận được; hãy nói thẳng điều này và nêu ngay Caveats (judge cùng model với app). Một lỗi: q05 MSFT net income vẫn hụt, chunk có 133,749 ở hạng 12 nên câu trả lời hedge đúng. Nếu được hỏi tiếp: q04 (MSFT total revenue) là ca tệ đi do reranker, recall từ 1.0 xuống 0.0; và lỗi nặng hơn từ baseline, q10/q20 trả "total revenue $139,297" (sai; thật là 215,938) từ một bảng không có tiêu đề cột nhưng vẫn qua `ground_numbers` vì chỉ kiểm tra văn bản. Lần chạy v2 không lặp lại lỗi đó, nhưng phép kiểm tra không đổi (xem `docs/ARCHITECTURE.md` mục 8). Bước tiếp: `docs/NEXT_STEP.md` | `reports/retrieval_experiments.md`, `reports/ragas_retrieval_v2.md` |

Ghi chú khi demo:
- Ở chế độ Replay, Streamlit luôn phát lại `runs/demo.json` (câu Apple net sales, intent `filing`), bất kể câu nhập ở ô Question; app hiển thị dòng "Replaying a recorded run" kèm câu đã ghi. Hãy nói rõ với khán giả đây là bản ghi.
- Giá thay đổi mỗi lần chạy (trace thật: 215.38 USD lúc 2026-10-29 01:31 UTC), đừng hứa số cụ thể trước.
- Một lần chạy live `both` mất khoảng 13.65 s (trace thật, trước khi có reranker; nay thêm khoảng 140 ms ở `retrieve`); vừa chờ vừa giải thích node đang chạy.
- Câu hỏi filing đầu tiên của một tiến trình mới phải tải và nạp model reranker (khoảng 90 MB ở lần tải đầu tiên trên máy; các lần sau chỉ nạp từ cache), nên chậm hơn hẳn. Hỏi thử một câu filing trước giờ demo để model nằm trong cache và đã nạp. Chế độ Replay không dùng reranker.

## 2. Danh sách 6 thứ cần chuẩn bị

| # | Thứ cần mang | Đường dẫn / lệnh | Sẵn sàng |
|---|---|---|---|
| 1 | Product | `uv run streamlit run app/streamlit_app.py`; CLI `uv run finassist ask "..."`; bản ghi `runs/demo.json` | [ ] |
| 2 | Repository | repo `LangChain`, nhánh `main`; `README.md`, `src/finassist/`, `tests/` (248 passed, 38 skipped) | [ ] |
| 3 | Architecture | `docs/ARCHITECTURE.md` (thành phần, control flow, data flow thật, sự cố, quyết định kiến trúc, giới hạn) | [ ] |
| 4 | Evaluation | `reports/ragas_retrieval_v2.md`, `reports/retrieval_experiments.md`, `reports/ragas_baseline.md` (vector thuần) và các file `.json` kèm theo; `eval/golden_set.json` (30 câu, 25 câu có `evidence`) | [ ] |
| 5 | Team evidence | `docs/TEAM_LOG.md`; `docs/evidence/links.md` và `docs/evidence/trace_tree.md`; `git log` | [ ] |
| 6 | Next step | `docs/NEXT_STEP.md` (các hướng còn lại sau đợt truy hồi: tách phần giá khỏi truy vấn, reranker hợp với bảng, grounding theo ngữ nghĩa) | [ ] |

## 3. Kế hoạch dự phòng

| Sự cố | Làm gì |
|---|---|
| Router LLM hoặc mạng lỗi khi live | Streamlit: tick "Replay recorded run (offline fallback)" ở sidebar rồi bấm Ask. Terminal: `uv run finassist ask x --replay runs/demo.json` (không gọi mạng, không cần LLM; tham số câu hỏi bị bỏ qua) |
| yfinance lỗi hoặc không có mạng | `FINASSIST_PRICE_OFFLINE=1 uv run streamlit run app/streamlit_app.py` (hoặc đặt biến này khi chạy `finassist ask`). Giá lấy từ `data/cache/prices.json` (có NVDA, AAPL, MSFT) và hiển thị `stale` |
| LangSmith không mở được | Dùng `docs/evidence/trace_tree.md` (text rendering của 2 trace thật). Không có ảnh chụp màn hình trong repo |
| Streamlit không lên | Dùng CLI `finassist ask` hoặc `--replay` |
| Hỏi số liệu khác với kỳ vọng | Chỉ vào citation và `reports/retrieval_experiments.md`: lỗi retrieval đã biết (9 trên 25 câu vẫn trượt top-5); thường hệ thống hedge, nhưng ở baseline q10/q20 đã trả một con số sai có nguồn (139,297), nên hãy nói thẳng điều này |
| Không có mạng (model reranker đã có trong cache) | Đặt `HF_HUB_OFFLINE=1` cùng với `FINASSIST_PRICE_OFFLINE=1`. Khi mạng tắt mà thiếu biến này, việc nạp model mất khoảng 29 giây; có biến này chỉ khoảng 6 giây (đo một lần trên một máy, nên chỉ là số tham khảo). Nếu model reranker chưa có trong cache, ứng dụng ghi một cảnh báo và dùng top-5 vector thuần. Câu filing đầu tiên, khi không có cache lẫn mạng, vẫn có thể phải chờ lần tải model hết thời gian, đo được khoảng 50 giây một lần |
| Câu filing đầu tiên rất chậm | Model reranker đang được tải và nạp. Chờ, hoặc chuyển sang Replay; lần sau sẽ nhanh |

Tạo lại bản ghi: `uv run finassist ask "..." --record runs/demo.json` (cần router).

Câu hỏi live: dùng câu tiếng Anh làm chính (ví dụ "What is NVIDIA's current stock price and how do export controls threaten it?"), vì embedding `bge-small-en-v1.5` chỉ dành cho tiếng Anh. Tiếng Việt chỉ là phần thưởng tùy chọn: câu rủi ro xuất khẩu của NVIDIA đã được giảm thiểu bằng `search_query` tiếng Anh do classifier viết lại, nhưng câu về bảng số liệu (ví dụ "Lợi nhuận ròng của Microsoft là bao nhiêu?") vẫn hay retrieve hụt.

Lưu ý: lệnh là `uv run python -m eval.run_ragas --tag <tên>` và `--tag` là bắt buộc. Tag mặc định `baseline` đã bị bỏ, vì nó từng ghi đè `reports/ragas_baseline.{json,md}`, kể cả các ghi chú viết tay ("TO ANALYZE", Caveats, Worst cases). Hãy dùng một tên mới (ví dụ `--tag thu_nghiem`). Chạy lại mất khoảng 14 đến 19 phút. Không cần chạy lại trước giờ demo.

## 4. Danh sách kiểm tra an toàn

- [ ] `.env` không mở trong editor, không `cat .env`, không hiện trong terminal history hay cửa sổ chia sẻ. File này nằm trong `.gitignore`.
- [ ] Khóa API (router, LangSmith) không hiện trên màn hình. Trace URL LangSmith nằm trong `docs/evidence/links.md`; mở trang LangSmith ở tab riêng và chỉ chia sẻ tab đó, không chia sẻ màn hình có thanh Settings/API keys.
- [ ] Đóng các tab trình duyệt không liên quan (mail, chat, tab có thông tin cá nhân) trước khi chia sẻ màn hình.
- [ ] Tắt thông báo hệ thống.
- [ ] Trước giờ demo: `git ls-files | grep -E '^\.env$'` phải rỗng và `git grep -nE 'sk-[A-Za-z0-9-]{12,}|lsv2_pt_' -- . ':!docs/DEMO_SCRIPT.md'` không có kết quả (loại trừ chính file này vì nó chứa mẫu regex).
- [ ] Chạy thử `uv run finassist ask x --replay runs/demo.json` một lần trước giờ demo.
- [ ] Chạy thử một câu filing live (ví dụ "What was Apple's total net sales in its most recent fiscal year?") để model reranker đã được tải và nạp.
- [ ] Cỡ chữ terminal và trình duyệt đủ lớn để cả phòng đọc được.

## 5. Chuẩn bị hỏi đáp: mẫu và 2 ví dụ đã điền

Mẫu gồm bốn phần theo thứ tự: Context (đang làm gì, môi trường), Attempt (đã thử gì), Evidence (log, test, commit) và Specific question (một câu hỏi hẹp).

### Ví dụ 1: `ragas` không import được

- Context: chạy `eval/run_ragas.py` để chấm baseline bằng RAGAS 0.4.3 trên Python 3.12, `langchain-core` 1.x.
- Attempt: `import ragas` lỗi vì ragas import `ChatVertexAI` và `VertexAI` từ `langchain_community`, mà `langchain-community` 0.4.x không còn cung cấp. Không hạ được `langchain-core` về bản 0.x vì các gói LangChain khác trong dự án cần bản 1.x. Cách đang dùng: `eval/_ragas_compat.py` đăng ký module giả rỗng (`_stub`, dòng 22-32) chỉ khi module thật thiếu, và import trước ragas (`eval/run_ragas.py:11`). Đã thu hẹp `except` thành `ModuleNotFoundError` đúng tên module để không che lỗi import khác (commit `b0ad410`).
- Evidence: `eval/_ragas_compat.py`, `tests/test_ragas_compat.py`, baseline chạy được 25 câu, 0 lỗi app (`reports/ragas_baseline.md`).
- Specific question: shim stub này có chấp nhận được cho một lần đo baseline không, hay nên ghim phiên bản `ragas` cũ hơn và chấp nhận phải hạ cả bộ LangChain?

### Ví dụ 2: biến môi trường trong shell ghi đè `.env`

- Context: `finassist` đọc `ANTHROPIC_BASE_URL` và `ANTHROPIC_API_KEY` từ `.env` qua `python-dotenv`.
- Attempt: mặc định `load_dotenv` không ghi đè biến đã có trong shell, nên giá trị `ANTHROPIC_*` đã export sẵn trong shell thắng `.env` và app gọi nhầm endpoint. Đã đổi thành `load_dotenv(ROOT / ".env", override=True)` (`src/finassist/config.py:39`), và thêm fixture trong `tests/conftest.py` vô hiệu hóa việc nạp `.env` khi chạy unit test (trừ khi `RUN_INTEGRATION=1`) để test không phụ thuộc máy (commit `e091608`).
- Evidence: commit `e091608` "let project .env override shell env; isolate unit tests from .env"; `tests/conftest.py` (fixture `_isolate_from_dotenv`); `README.md` mục Xử lý sự cố.
- Specific question: với một app demo, nên để `.env` của dự án luôn thắng như hiện tại, hay để biến shell thắng (chuẩn 12-factor) và in cảnh báo khi hai giá trị khác nhau?
