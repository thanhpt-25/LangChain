# FinAssist (`finassist`)

A LangGraph assistant for AAPL, MSFT and NVDA: it routes a question to a stock price tool, a Chroma retriever over 10-K filings, or a refusal, and returns a validated `FinanceAnswer` with citations.

FinAssist là trợ lý AI tài chính cho ba công ty Apple (AAPL), Microsoft (MSFT) và NVIDIA (NVDA). Câu hỏi về nội dung báo cáo 10-K được trả lời bằng RAG trên kho Chroma cục bộ, kèm trích dẫn `chunk_id`. Câu hỏi về giá, vốn hóa và P/E được trả lời bằng một công cụ tất định gọi yfinance. Kết quả là một đối tượng JSON có cấu trúc (`FinanceAnswer`). Câu hỏi ngoài phạm vi, kể cả xin tư vấn đầu tư, bị từ chối. Dự án được làm cho buổi demo W1 (miền: trợ lý chat AI).

## Kiến trúc tóm tắt

Một câu hỏi đi vào `classify_intent`, node này chọn một trong bốn đường đi. Sơ đồ đầy đủ, state, từng node, data flow thật, các kiểu sự cố và bảng quyết định nằm trong [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```mermaid
flowchart TD
    Start(["question"]) --> Classify["classify_intent"]
    Classify -->|"price"| Price["price_tool"]
    Classify -->|"both"| Price
    Classify -->|"filing"| Retrieve["retrieve"]
    Classify -->|"out_of_scope"| Refuse["refuse"]

    Price -->|"intent là both"| Retrieve
    Price -->|"intent là price"| Generate["generate_answer"]
    Retrieve --> Generate
    Generate --> Format["format_output"]

    Format --> End(["FinanceAnswer"])
    Refuse --> End
```

Node `retrieve` lấy 30 chunk ứng viên từ Chroma cho mỗi ticker, một cross-encoder cục bộ chấm lại, và 5 chunk đứng đầu được gửi cho LLM (xem [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) mục 4.5). Truy vấn của `retrieve` là `search_query` tiếng Anh do bộ phân loại viết lại khi câu hỏi là `both` (bản viết lại chỉ còn phần 10-K) hoặc không phải tiếng Anh. Mọi câu khác tìm bằng câu hỏi gốc, và `generate_answer` luôn nhận câu hỏi gốc (mục 4.4).

LLM hiểu câu hỏi, viết câu trả lời và trích số liệu. Mã tất định tạo citation, lấy giá từ công cụ và loại các con số không có trong chunk đã retrieve (`ground_numbers`).

## Yêu cầu

- `uv`. Dự án ghim Python 3.12 (`pyproject.toml:4`), đừng dùng Python của hệ thống.
- Một endpoint LLM tương thích Anthropic. Dự án được xây với router cục bộ (`ANTHROPIC_BASE_URL=http://localhost:20128/v1`, model `kr/claude-sonnet-5.5`).
- Khóa LangSmith (tùy chọn, chỉ cần để xem trace).
- Cần mạng một lần để tải filing từ SEC EDGAR, tải model embedding `BAAI/bge-small-en-v1.5`, tải model reranker và lấy giá từ `yfinance`.

## Cài đặt

```bash
uv sync
cp .env.example .env            # then edit .env: router key, LangSmith key, SEC_USER_AGENT
uv run python scripts/fetch_filings.py   # downloads latest 10-K of AAPL/MSFT/NVDA into data/raw/
uv run finassist ingest                  # chunk by Item, embed locally, store in data/chroma/
```

`.env` và `data/` nằm trong `.gitignore`. Không commit hai thứ này.

Lần chạy đầu tiên cần reranker sẽ tải model `cross-encoder/ms-marco-MiniLM-L-6-v2` (khoảng 90 MB) từ Hugging Face. Việc này xảy ra một lần, ở câu hỏi đầu tiên cần truy hồi, nên câu đó chậm hơn hẳn. Từ lần sau model nằm trong cache của máy.

### Biến môi trường tùy chọn

Tất cả đều tùy chọn. Giá trị mặc định là cấu hình đã được đo và chọn (xem [`reports/retrieval_experiments.md`](reports/retrieval_experiments.md)).

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `FINASSIST_RETRIEVAL` | `dense` | `dense` hoặc `hybrid` (thêm BM25 và RRF, tùy chọn đã đo và không được chọn) |
| `FINASSIST_RERANKER` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | tên cross-encoder, hoặc `none` để quay về vector top-5 |
| `FINASSIST_CANDIDATES` | `30` | số ứng viên lấy trước khi chấm lại |
| `FINASSIST_EMBED_PREFIX` | `0` | `1` thì embed thêm tiền tố ticker, năm, Item (tùy chọn đã đo và không được chọn) |
| `FINASSIST_TABLE_ROWS` | `0` | `1` thì mỗi hàng bảng thành một dòng văn bản (tùy chọn đã đo và không được chọn) |
| `FINASSIST_CHROMA_DIR` | `data/chroma` | thư mục kho Chroma; đặt thư mục khác để thử mà không đụng kho chuẩn |
| `FINASSIST_PRICE_TIMEOUT` | `8` | số giây chờ yfinance cho mỗi ticker; quá hạn thì dùng giá cache (`stale`) hoặc báo không có giá. Giá trị <= 0, nan, inf hoặc không phải số thì dùng mặc định 8 và ghi cảnh báo |

Hai biến `FINASSIST_EMBED_PREFIX` và `FINASSIST_TABLE_ROWS` đổi cách chunk, nên sau khi bật phải nạp lại kho:

```bash
uv run finassist ingest --rebuild   # xóa collection rồi nạp lại từ data/raw
```

## Chạy

```bash
uv run finassist ask "What is NVIDIA's current stock price and how do export controls threaten it?"
uv run finassist ask "..." --record runs/demo.json     # run live and save the result
uv run finassist ask x --replay runs/demo.json         # offline: replay the saved run (question text is ignored)
uv run streamlit run app/streamlit_app.py              # UI; sidebar checkbox "Replay recorded run" = offline fallback
FINASSIST_PRICE_OFFLINE=1 uv run finassist ask "..."   # price tool uses cache only (data/cache/prices.json)
```

Dùng câu hỏi tiếng Anh làm chính, vì model embedding chỉ hỗ trợ tiếng Anh. Câu tiếng Việt vẫn chạy nhờ `search_query` tiếng Anh do bộ phân loại viết lại (xem giới hạn bên dưới).

## Đánh giá

Bộ đo truy hồi chạy trên 25 câu có `evidence` trong `eval/golden_set.json` và không gọi LLM:

```bash
uv run python -m eval.retrieval_eval --label "tên lần chạy"                                  # in bảng any@K, all@K, MRR, độ trễ
uv run python -m eval.retrieval_eval --label "tên lần chạy" --append reports/retrieval_experiments.md   # ghi thêm vào báo cáo
uv run python -m eval.retrieval_eval --label "tên lần chạy" --classify --save-rewrites reports/query_rewrites.json   # truy vấn do bộ phân loại thật viết lại (cần router)
uv run python -m eval.retrieval_eval --label "tên lần chạy" --rewrites reports/query_rewrites.json   # phát lại bảng viết lại đã lưu, không cần router
```

Lệnh nhận thêm `--mode`, `--reranker`, `--candidates` và `--show-misses` để thử một cấu hình mà không sửa `.env`. Không có `--classify` hay `--rewrites` thì mọi câu tìm bằng câu hỏi gốc (control). `--classify` cho từng câu đi qua bộ phân loại thật và dùng quy tắc truy vấn của ứng dụng; `--save-rewrites` lưu bảng kết quả; `--rewrites` phát lại bảng đó. Kết quả hiện tại, với bản viết lại cho câu `both`: any@5 72% và all@5 40% (control không viết lại: 64% và 36%; vector thuần: 48% và 28%), MRR 0,404, nhóm combined any@5 80% (control 40%). Hai mục tiêu 70% và 40% đạt, nhưng chỉ nhờ 5 câu `both` và all@5 vừa đúng ngưỡng. Bản viết lại do LLM tạo, nên một lần `--classify` mới có thể cho số khác; `reports/query_rewrites.json` phát lại một lần chạy.

RAGAS đầy đủ chấm 25 câu bằng bốn metric và kiểm tra intent trên cả 30 câu:

```bash
uv run python -m eval.run_ragas --tag retrieval_v2   # ghi reports/ragas_retrieval_v2.{json,md}
```

Lệnh này mất khoảng 15 đến 19 phút. `--tag <tên>` là bắt buộc. Tag mặc định `baseline` đã bị bỏ, vì nó từng ghi đè `reports/ragas_baseline.{json,md}` cùng các ghi chú viết tay; hãy dùng một tên mới. Các báo cáo hiện có: `reports/ragas_baseline.md` (vector thuần, đã chú giải), `reports/ragas_baseline_rerun.md` (chạy lại vector thuần, để đo nhiễu), `reports/ragas_retrieval_v2.md` và `reports/ragas_demo_check.md` (reranker, chưa viết lại), `reports/ragas_both_rewrite_1.md` và `reports/ragas_both_rewrite_2.md` (cấu hình hiện tại).

Điểm RAGAS (faithfulness, context precision, context recall, answer relevancy):

| Lần chạy | faithfulness | precision | recall | relevancy |
|---|---|---|---|---|
| vector thuần (`ragas_baseline_dense`) | 0,939 | 0,309 | 0,387 | 0,687 |
| vector thuần chạy lại (`ragas_baseline_rerun`) | 0,961 | 0,309 | 0,420 | 0,665 |
| reranker, chưa viết lại (`ragas_retrieval_v2`) | 0,939 | 0,342 | 0,400 | 0,766 |
| reranker, chạy lại (`ragas_demo_check`) | 0,933 | 0,342 | 0,413 | 0,770 |
| cấu hình hiện tại, lần 1 (`ragas_both_rewrite_1`) | 0,952 | 0,377 | 0,420 | 0,772 |
| cấu hình hiện tại, lần 2 (`ragas_both_rewrite_2`) | 0,929 | 0,372 | 0,460 | 0,799 |

Cách đọc: nhiễu giữa hai lần chạy của cấu hình hiện tại là 0,04 ở recall và 0,023 ở faithfulness, lớn hơn nhiễu của cấu hình cũ. Chỉ precision (cao hơn khoảng 0,03) là thay đổi chắc chắn; recall (+0,03) và relevancy (+0,02) nằm trong nhiễu. Ở nhóm combined, precision từ 0,107 lên 0,282 và 0,257, và q24 từ "không so sánh được" sang nêu cả hai doanh thu. Với bản chạy vector thuần, precision và relevancy tăng (relevancy phần lớn vì ít câu bị từ chối hơn), còn faithfulness và recall không đổi trong phạm vi nhiễu 0,02 đến 0,03 giữa hai lần chạy vector thuần. Hãy đọc mục "Caveats" trong các báo cáo trước khi trích số, vì judge dùng cùng model với ứng dụng.

## Kiểm thử

```bash
uv run pytest -q                                       # 266 passed, 46 skipped (integration tests skipped)
RUN_INTEGRATION=1 uv run pytest tests/test_both_rewrite.py tests/test_advice_refusal.py tests/test_llm_smoke.py tests/test_retrieval_integration.py -q   # needs the router up and a populated data/chroma
```

46 test bị bỏ qua là các integration test cần router, model thật và `data/chroma`. Trong đó `tests/test_retrieval_integration.py` có ba test chỉ cần model cục bộ: q03 và q21 phải có chunk bằng chứng trong top-5, còn q05 là `xfail` ở chế độ strict (bằng chứng của q05 nằm ở hạng 12, nên test sẽ báo lỗi ngay khi một thay đổi làm nó vào top-5 và nhắc cập nhật tài liệu).

`tests/test_advice_refusal.py` chạy bộ phân loại thật trên 14 cách hỏi xin lời khuyên đầu tư hoặc dự đoán giá và 9 câu đối chứng (có 3 câu ghép khó hơn). Lần chạy gần nhất: cả 14 câu bị từ chối và 9 câu đối chứng giữ đúng intent. `tests/test_both_rewrite.py` chạy bộ phân loại thật trên 8 câu `both` (có hai câu tiếng Việt) và đòi `search_query` là tiếng Anh, không còn từ về giá và giữ tên mọi công ty. Với router bật, bốn file integration trên cho 59 passed, 1 xfailed.

## Giới hạn đã biết

Danh sách đầy đủ nằm ở [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) mục 8 (Giới hạn đã biết). Các giới hạn cần biết trước khi trích số liệu:

- Truy hồi còn bỏ lỡ bằng chứng ở 7 trên 25 câu của bộ đo (q04, q05, q10, q11, q15, q16, q22). Reranker chỉ sắp xếp lại 30 ứng viên của vector, nên chunk nằm ngoài 30 ứng viên không cứu được. Ví dụ: lợi nhuận ròng của Microsoft (q05) vẫn bị trả lời là không có số liệu.
- Model embedding chỉ có tiếng Anh. Câu hỏi không phải tiếng Anh được retrieve bằng `search_query` tiếng Anh do bộ phân loại viết lại, còn câu `filing` tiếng Anh giữ nguyên. Câu hỏi tiếng Việt về bảng số liệu vẫn có thể trượt chunk.
- Bản viết lại cho câu `both` phụ thuộc vào bộ phân loại và được đo trên 5 câu. Một bản viết lại làm mất tên công ty hoặc đổi nghĩa sẽ kéo truy hồi tới chunk sai, và chưa có bước nào kiểm tra tự động ngoài `tests/test_both_rewrite.py`.
- `ground_numbers` chỉ kiểm tra một con số có xuất hiện trong chunk đã retrieve, không kiểm tra nghĩa của nó. Ở baseline vector thuần, q10 và q20 trả "total revenue $139,297" cho NVIDIA từ một bảng segment không có tiêu đề cột, trong khi giá trị đúng là 215,938.

## Bản đồ repo

```
pyproject.toml  uv.lock  .env.example  .gitignore  .python-version
src/finassist/
  config.py       Settings, normalize_base_url, TICKERS
  schemas.py      pydantic models (Chunk, FinanceAnswer, PriceInfo, Intent...)
  ingest.py       html_to_text, split_items, chunk_filing, load_filing
  vectorstore.py  get_embeddings, FilingStore (Chroma, filter by ticker, reset, chunks_for)
  retrieval.py    Retriever: vector top-N, optional BM25 + RRF, cross-encoder rerank
  tools.py        get_stock_price (deterministic, cache fallback, stale flag)
  postprocess.py  extract_citations, ground_numbers
  graph.py        GraphState, Deps, resolve_intent, retrieval_query, build_graph (6 nodes)
  chains.py       real LLM implementations of Deps (classify, generate, structure)
  runtime.py      build_app, ask
  cli.py          finassist ingest [--rebuild] | ask [--record|--replay]
app/streamlit_app.py
scripts/          fetch_filings.py  find_in_filing.py  trace_link.py  trace_evidence.py
eval/             golden_set.json  run_ragas.py  retrieval_eval.py  report.py  _ragas_compat.py
reports/          ragas_*.{json,md}  retrieval_experiments.md  query_rewrites.json
runs/             demo.json  (recorded answer for offline replay)
docs/  tests/
data/             (git-ignored) raw/ chroma/ cache/
```

## Bản đồ tài liệu

| File | Nội dung |
|---|---|
| [`docs/README.md`](docs/README.md) | mục lục tài liệu, đọc theo vai trò |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | thành phần, control flow, data flow, sự cố, quyết định kiến trúc, giới hạn |
| [`docs/KNOWLEDGE_CHECK.md`](docs/KNOWLEDGE_CHECK.md) | câu trả lời sẵn cho Giải thích, Dự đoán, Gỡ lỗi, Bảo vệ lựa chọn và các câu hỏi gợi ý |
| [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) | kịch bản demo 8 phút, danh sách 6 thứ cần chuẩn bị, phương án dự phòng, kiểm tra an toàn, mẫu hỏi đáp |
| [`docs/TEAM_LOG.md`](docs/TEAM_LOG.md) | ai đã làm gì, kèm commit |
| [`docs/NEXT_STEP.md`](docs/NEXT_STEP.md) | kết quả đợt cải thiện truy hồi và các hướng còn lại |
| [`docs/evidence/`](docs/evidence/) | link trace LangSmith và bản text của cây trace |
| `docs/superpowers/specs/`, `docs/superpowers/plans/` | spec thiết kế và kế hoạch triển khai |

## Xử lý sự cố

- `ANTHROPIC_BASE_URL` kết thúc bằng `/v1` (kiểu curl). SDK Anthropic tự thêm `/v1/messages`, nên `normalize_base_url` cắt phần `/v1` ở cuối (`src/finassist/config.py:18-20`).
- Các model `auto/*` của router báo lỗi giới hạn tần suất từ upstream, nên model được ghim là `kr/claude-sonnet-5.5` (`src/finassist/config.py:63`).
- Biến shell thắng `.env` theo mặc định của `python-dotenv`. `get_settings` gọi `load_dotenv(..., override=True)` (`src/finassist/config.py:58`) để `.env` của dự án thắng các biến `ANTHROPIC_*` đã export sẵn. Unit test vô hiệu hóa việc nạp `.env` trong `tests/conftest.py`.
- SEC từ chối tải về: đặt `SEC_USER_AGENT="Your Name you@email"` trong `.env`. SEC yêu cầu thông tin liên hệ (`src/finassist/config.py:69`).
- `ImportError` cho `ChatVertexAI` khi import ragas: ragas 0.4.3 cần các module mà `langchain-community` không còn cung cấp. `eval/_ragas_compat.py` đăng ký module giả rỗng và phải được import trước ragas (đã làm ở `eval/run_ragas.py:11`).
- Router LLM lỗi lúc demo: chạy `uv run finassist ask x --replay runs/demo.json`, hoặc tick "Replay recorded run" trong Streamlit.
- Báo "The filings store is empty": kho Chroma chưa có chunk nào. Chạy `uv run finassist ingest` rồi hỏi lại. Câu hỏi chỉ về giá vẫn chạy khi kho rỗng.
- Lấy giá thất bại hoặc quá `FINASSIST_PRICE_TIMEOUT` giây: công cụ dùng `data/cache/prices.json` và đặt `stale=true` (`src/finassist/tools.py:111-113`).
- Sửa `CLASSIFY_SYSTEM` mà kết quả phân loại không đổi: router lưu cache phản hồi của bộ phân loại theo tin nhắn người dùng, không theo prompt hệ thống, nên một thay đổi prompt bị bỏ qua mà không có cảnh báo. Đổi hậu tố của tin nhắn người dùng trong `classify` (hiện là `QUESTION (v2)`, `src/finassist/chains.py:123`) lên `QUESTION (v3)` và các số tiếp theo mỗi lần sửa prompt, rồi chạy lại `tests/test_both_rewrite.py` với `RUN_INTEGRATION=1`.
