# Viết lại truy vấn truy hồi cho câu `both`: đặc tả thiết kế

Ngày: 2026-10-29 · Trạng thái: chờ duyệt · Tiếp nối [`docs/NEXT_STEP.md`](../../NEXT_STEP.md) và [`reports/retrieval_experiments.md`](../../../reports/retrieval_experiments.md)

## 1. Vấn đề

Câu `both` hỏi cùng lúc về giá và về 10-K, ví dụ "What is NVIDIA's market capitalization now, and what was its total revenue last fiscal year?". Hiện tại cả câu được dùng làm truy vấn cho vector search. Các từ về giá ("market capitalization", "current stock price", "P/E") kéo vector truy vấn về các chunk định giá và cổ phiếu, nên chunk chứa đáp án của phần 10-K bị đẩy xuống thấp hoặc ra khỏi 30 ứng viên.

Số đo của đợt truy hồi trước cho thấy nhóm combined (q21 đến q25, đều là câu `both`) chỉ đạt any@5 40% với cấu hình cuối (vector top-30 cộng reranker MiniLM). q22 và q25 không có chunk bằng chứng nào trong 30 ứng viên.

## 2. Số đo ban đầu về mức cải thiện có thể đạt

Tôi viết tay lại 5 câu combined, bỏ phần hỏi về giá, rồi chạy qua đúng bộ truy hồi hiện tại. Hạng của chunk bằng chứng (nhỏ hơn là tốt hơn, `-` là không có trong 30 ứng viên):

| Câu | Câu gốc | Sau khi bỏ phần giá |
|---|---|---|
| q21 (AAPL) | 5 | 2 |
| q22 (MSFT, hai nhóm bằng chứng) | -, - | 9, 20 |
| q23 (NVDA) | 3 | 3 |
| q24 (AAPL và MSFT) | 10, 12 | 5, 7 |
| q25 (NVDA, hai nhóm) | -, - | 7, 5 |

Nhóm combined có any@5 từ 2/5 lên 4/5, và any@5 chung từ 64% lên khoảng 72%, vượt mục tiêu 70% mà đợt trước thiếu 2 câu. Đây là giới hạn trên, vì các câu ở đây do tôi viết tay. Bản viết lại do LLM tạo có thể kém hơn, nên spec yêu cầu đo lại với bản viết lại thật.

## 3. Mục tiêu và phạm vi

Mục tiêu: đưa chunk bằng chứng của phần 10-K trong câu `both` vào top-5.

Trong phạm vi:

- bộ phân loại trả `search_query` tiếng Anh chỉ gồm phần hỏi về 10-K cho câu `both`;
- đồ thị dùng `search_query` đó để truy hồi;
- bộ đo truy hồi đo được tác dụng với bản viết lại thật, có thể phát lại mà không gọi LLM;
- test, đo lại và cập nhật tài liệu.

Ngoài phạm vi: viết lại truy vấn cho câu `filing` tiếng Anh (giữ nguyên để 20 câu khác của bộ đo không đổi), thay đổi reranker, bộ đo lớn hơn, kiểm tra ngữ nghĩa của số.

## 4. Thiết kế

### 4.1 Quy tắc chọn truy vấn truy hồi

Một hàm thuần `retrieval_query(question: str, intent: Intent, result: IntentResult) -> str` trong `src/finassist/graph.py`. `intent` là intent sau khi đồ thị lọc ticker, còn `result` là kết quả thô của bộ phân loại:

- nếu (`intent == "both"` hoặc `result.language == "other"`) và `result.search_query` không rỗng sau khi cắt khoảng trắng, trả `search_query` đã cắt;
- trường hợp còn lại trả `question` gốc.

`classify_intent` dùng hàm này để đặt `search_query` trong state. Hôm nay điều kiện chỉ là `language == "other"`, nên đây là một thay đổi nhỏ: thêm `intent == "both"`. `generate_answer` vẫn nhận câu hỏi gốc nên câu trả lời giữ nguyên ngôn ngữ và nội dung hỏi, kể cả phần giá.

Dùng intent sau khi lọc ticker vì một câu `both` mất hết ticker hợp lệ thành `out_of_scope` và không truy hồi.

### 4.2 Prompt của bộ phân loại

`CLASSIFY_SYSTEM` mở rộng mô tả của `search_query`:

- với câu `both`, luôn trả một câu hỏi tiếng Anh, đầy đủ và tự đứng được, chỉ nói về nội dung báo cáo 10-K (doanh thu, EPS, rủi ro, MD&A), giữ tên công ty và thuật ngữ tài chính, và bỏ mọi từ về giá, vốn hóa, P/E, thị trường hay cổ phiếu hiện tại;
- với câu `both` hỏi nhiều công ty, giữ tên tất cả công ty (ví dụ q24);
- với câu không phải `both` và tiếng Anh, `search_query` là null như hiện nay; với câu không phải tiếng Anh, giữ quy tắc hiện tại (bản viết lại tiếng Anh).

Có hai ví dụ trong prompt (một câu tiếng Anh, một tiếng Việt). Schema `IntentResult` không đổi, chỉ cập nhật mô tả trường.

### 4.3 Đo bằng bộ đo truy hồi, có thể phát lại

`eval/retrieval_eval.py` thêm hai tuỳ chọn:

- `--classify`: với mỗi câu hỏi, gọi bộ phân loại thật, tính truy vấn bằng `retrieval_query` và dùng nó cho truy hồi. `--save-rewrites PATH` ghi bảng `{id: {"intent", "language", "search_query"}}` ra file JSON.
- `--rewrites PATH`: đọc bảng đã lưu, không gọi LLM. Dùng để phát lại chính xác và để chạy trong test.

Không có hai tuỳ chọn này, bộ đo chạy như hiện nay (câu hỏi gốc). Bảng bản viết lại của lần đo chính thức được lưu vào `reports/query_rewrites.json` để người khác kiểm lại mà không cần router.

### 4.4 Cổng quyết định

Giữ thay đổi nếu đo với bản viết lại thật (chạy `--classify` rồi phát lại bằng `--rewrites`) thỏa cả hai:

- any@5 của nhóm combined từ 40% lên ít nhất 60%;
- any@5 chung không thấp hơn 64% và any@5 của hai nhóm numeric và risk không giảm (hai nhóm này gồm 20 câu không phải `both` nên không thay đổi theo thiết kế; đo để xác nhận).

Nếu cổng không đạt, giữ mã hàm `retrieval_query` nhưng không bật quy tắc `both` (điều kiện về lại chỉ `language == "other"`), ghi kết quả vào báo cáo thí nghiệm.

### 4.5 Kiểm tra trên chất lượng câu trả lời

Chạy RAGAS đầy đủ hai lần với cấu hình mới (`--tag both_rewrite_1`, `--tag both_rewrite_2`) và so với `retrieval_v2` và `demo_check`, hai lần chạy của cấu hình trước (nhiễu giữa hai lần đó dưới 0,02). Tiêu chí: recall của nhóm combined cao hơn mức cao nhất của hai lần chạy trước (0,267 và 0,200, nên ít nhất 0,30 là có ý nghĩa), faithfulness chung không giảm quá 0,03, và q21 đến q25 không có câu nào mất bằng chứng đang có. Báo cáo trung thực nếu không đạt.

## 5. Thay đổi trong repo

- Sửa: `src/finassist/graph.py` (`retrieval_query` và `classify_intent`), `src/finassist/chains.py` (`CLASSIFY_SYSTEM`), `src/finassist/schemas.py` (mô tả `search_query`), `eval/retrieval_eval.py` (hai tuỳ chọn), tài liệu `ARCHITECTURE.md`, `NEXT_STEP.md`, `README.md`, `KNOWLEDGE_CHECK.md`, `DEMO_SCRIPT.md`.
- Thêm: `reports/query_rewrites.json`, thí nghiệm ghi vào `reports/retrieval_experiments.md`, `reports/ragas_both_rewrite_*.{md,json}`.
- Test mới: unit test cho `retrieval_query`, test đồ thị chứng minh `deps.retrieve` nhận `search_query` cho câu `both` còn câu `filing` tiếng Anh nhận câu gốc, test bộ đo với `--rewrites` trên dữ liệu nhỏ, integration test với router thật kiểm `search_query` của một số câu `both` không còn từ về giá.

Không đổi: reranker, `Deps` (chữ ký `retrieve(question, tickers)` giữ nguyên; giá trị truyền vào là `search_query`), `postprocess.py`, kho Chroma.

## 6. Kiểm thử

- Unit test không gọi LLM: `retrieval_query` với các tổ hợp (intent, language, search_query rỗng hoặc có) cho đúng kết quả; đồ thị truyền đúng truy vấn cho `retrieve`; bộ đo với `--rewrites` tính đúng từ bảng cho trước.
- Integration test (`RUN_INTEGRATION=1`): với khoảng 8 câu `both` tiếng Anh và tiếng Việt, `search_query` do bộ phân loại trả không chứa các từ về giá (`price`, `market cap`, `P/E`, `giá`, `vốn hóa`) và vẫn chứa tên công ty; các câu `filing` và `price` tiếng Anh có `search_query` bị bỏ qua đúng như thiết kế.
- Kiểm tra lại bộ phân loại sau khi sửa prompt: 30/30 intent trên golden set (script phân loại, không chạy RAGAS), toàn bộ `tests/test_advice_refusal.py` (14 câu từ chối và 9 câu đối chứng) và các test phân loại cũ trong `tests/test_llm_smoke.py`.

## 7. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Thay đổi prompt làm lệch intent của các câu khác | Kiểm lại 30/30 golden set và các test advice trước khi tiếp tục; nếu lệch thì giữ prompt cũ và đặt yêu cầu về `search_query` vào chỉ dẫn phụ |
| Bản viết lại làm mất tên công ty hoặc mất phần hỏi của một câu nhiều công ty (q24) | Test integration kiểm tên công ty; xem bảng bản viết lại thật trong báo cáo |
| LLM viết lại tốt hơn hoặc kém hơn bản viết tay ở mục 2 | Đo bằng bản viết lại thật và lưu bảng, quyết định theo cổng 4.4 |
| Đo chỉ trên 5 câu combined, mỗi câu là 20 điểm phần trăm trong nhóm | Ghi rõ trong báo cáo; mục tiêu chung (any@5 từ 70% trở lên) chỉ là chỉ báo, không là bằng chứng chắc chắn |
| Truy vấn viết lại thiếu từ khóa của phần giá nên câu trả lời thiếu ngữ cảnh giá | Không xảy ra: giá lấy từ công cụ và được đưa vào ngữ cảnh riêng; `generate_answer` nhận câu hỏi gốc |
