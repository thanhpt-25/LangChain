# Kiến trúc ứng dụng FinAssist

FinAssist là trợ lý tài chính cho ba công ty: Apple (AAPL), Microsoft (MSFT) và NVIDIA (NVDA). Ứng dụng nhận hai loại câu hỏi và xử lý chúng bằng hai cơ chế khác nhau.

- Câu hỏi về nội dung báo cáo 10-K (doanh thu, EPS, rủi ro, MD&A) được trả lời bằng RAG trên một kho Chroma chạy cục bộ, kèm trích dẫn đến từng đoạn nguồn.
- Câu hỏi về giá, vốn hóa và P/E được trả lời bằng một công cụ gọi yfinance. Những con số này đổi theo từng phút nên LLM không được tự đưa ra.

Câu hỏi nằm ngoài hai loại trên bị từ chối, gồm tư vấn mua bán, dự đoán giá và các công ty khác.

Tài liệu mô tả ứng dụng đúng như mã trong `src/finassist/`. Các mục đi theo thứ tự một người mới thường hỏi: hệ thống có những phần nào (mục 1 và 2), một câu hỏi đi qua đâu (mục 3), dữ liệu đến từ đâu và ai tạo ra nó (mục 4 và 5), khi có sự cố thì chuyện gì xảy ra (mục 6), vì sao thiết kế như vậy (mục 7) và những gì còn chưa tốt (mục 8).

## 1. Hệ thống nhìn từ bên ngoài

Toàn bộ ứng dụng chạy trong một tiến trình Python. Embedding và mô hình reranker cũng chạy ngay trong tiến trình đó, nên chỉ có ba dịch vụ bên ngoài: router LLM, yfinance và SEC EDGAR. Hai mô hình cục bộ được tải từ Hugging Face ở lần chạy đầu tiên rồi nằm trong cache của máy. SEC EDGAR chỉ được gọi lúc chuẩn bị dữ liệu, còn LangSmith chỉ nhận trace khi được bật.

```mermaid
flowchart LR
    subgraph prep["Chuẩn bị dữ liệu, chạy một lần"]
        direction LR
        SEC["SEC EDGAR"] --> Fetch["fetch_filings.py"]
        Fetch --> Raw[("data/raw")]
        Raw --> Ingest["finassist ingest"]
    end

    subgraph run["Khi có câu hỏi"]
        direction LR
        User["Người dùng"] --> Entry["CLI hoặc Streamlit"]
        Entry --> App["FinAssist<br/>một tiến trình Python"]
    end

    Ingest --> Chroma[("data/chroma")]
    App -->|"tìm đoạn 10-K"| Chroma
    App -->|"phân loại, trả lời, trích field"| LLM["LLM qua router<br/>kr/claude-sonnet-5.5"]
    App -->|"giá, vốn hóa, P/E"| YF["yfinance"]
    App <-->|"đọc và ghi"| PriceCache[("data/cache/prices.json")]
    App -.->|"trace, nếu bật"| LS["LangSmith"]
```

Hai nhóm ở trên chạy ở hai thời điểm khác nhau. Nhóm chuẩn bị chạy một lần trước khi dùng và để lại kho Chroma. Nhóm thứ hai chạy mỗi khi có câu hỏi và đọc kho đó.

## 2. Các thành phần

### 2.1 Ai gọi ai

```mermaid
flowchart TB
    subgraph entry["Điểm vào"]
        cli["cli.py"]
        ui["app/streamlit_app.py"]
        ev["eval/run_ragas.py"]
    end

    rt["runtime.py<br/>build_app, ask"]

    subgraph core["Điều phối"]
        gr["graph.py<br/>StateGraph và giao diện Deps"]
        chains["chains.py<br/>prompt, gọi LLM, ghép Deps"]
    end

    subgraph support["Adapter và kiểm chứng"]
        rv["retrieval.py<br/>Retriever"]
        vs["vectorstore.py"]
        tools["tools.py"]
        post["postprocess.py"]
    end

    cli --> rt
    ui --> rt
    ev --> rt
    rt --> gr
    rt --> chains
    chains -->|"hiện thực Deps"| gr
    chains --> rv
    rv --> vs
    chains --> tools
    chains --> post
    gr --> post
```

`schemas.py` (các kiểu dữ liệu) và `config.py` (cấu hình) được hầu hết các module dùng nên không vẽ vào sơ đồ. `retrieval.py` đứng giữa `chains.py` và `vectorstore.py`: `chains.py` hỏi `Retriever`, còn `Retriever` mới hỏi kho Chroma.

Điểm đáng nhìn kỹ là mũi tên giữa `chains.py` và `graph.py`. `graph.py` không biết gì về LLM, Chroma hay yfinance. Nó chỉ biết một dataclass `Deps` gồm năm hàm: `classify`, `retrieve`, `price`, `generate`, `structure`. `chains.py` hiện thực năm hàm đó bằng các dịch vụ thật, còn test thay chúng bằng hàm giả. Nhờ vậy toàn bộ logic điều hướng và kiểm chứng chạy được trong unit test mà không cần mạng.

### 2.2 Từng module làm gì

| Module | Nhận vào | Trả ra | Gọi dịch vụ ngoài |
|---|---|---|---|
| `cli.py` | dòng lệnh `ingest` hoặc `ask` | văn bản và JSON in ra terminal, file record | không (qua `runtime`) |
| `app/streamlit_app.py` | câu hỏi trong ô nhập | trang web: câu trả lời, citation, chunk, JSON | không (qua `runtime`) |
| `runtime.py` | cấu hình | `build_app()` dựng đồ thị hoàn chỉnh, `ask()` chạy một câu hỏi kèm tag và metadata trace | không |
| `graph.py` | `Deps` | đồ thị LangGraph biên dịch sẵn | không |
| `chains.py` | `Settings`, kho Chroma, client LLM | một `Deps` thật (hàm `retrieve` đi qua `Retriever`); chứa ba prompt (phân loại, trả lời, trích field) | router LLM |
| `retrieval.py` | câu truy vấn, danh sách ticker, `Settings` | `Retriever.retrieve` trả `Chunk` (5 chunk mỗi ticker): vector top-30 rồi cross-encoder chấm lại; có tùy chọn BM25 và RRF | mô hình reranker cục bộ |
| `vectorstore.py` | danh sách `Chunk`, câu truy vấn | `search(query, ticker, k)` trả `Chunk` kèm khoảng cách; `chunks_for(ticker)`; `reset()` | Chroma, mô hình embedding cục bộ |
| `tools.py` | mã ticker | `PriceInfo` (giá, vốn hóa, P/E, `as_of`, `stale`) hoặc lỗi `PriceUnavailable` | yfinance, file cache |
| `postprocess.py` | câu trả lời, các chunk đã retrieve | danh sách `Citation` hợp lệ; `FinanceAnswer` đã bỏ số không có trong chunk | không |
| `ingest.py` | HTML của 10-K | các `Chunk` có metadata; tùy chọn đổi mỗi hàng bảng thành một dòng văn bản | không |
| `schemas.py` | không | các kiểu Pydantic dùng chung | không |
| `config.py` | file `.env` và biến môi trường | `Settings` | không |

### 2.3 Các kiểu dữ liệu chính

| Kiểu | Chứa gì | Ai tạo |
|---|---|---|
| `Chunk` | một đoạn 10-K: `chunk_id`, text, ticker, năm tài chính, Item, `score` (khoảng cách Chroma, càng thấp càng gần; chunk chỉ do BM25 tìm thấy thì là `None`) | `ingest.py`, rồi `vectorstore.search` |
| `PriceInfo` | giá, vốn hóa, P/E, thời điểm `as_of`, cờ `stale` | `tools.get_stock_price` |
| `IntentResult` | intent, danh sách ticker, `language`, `search_query` | LLM phân loại |
| `LLMExtraction` | `ticker`, `fiscal_year`, `revenue`, `eps`, `risk_factors` | LLM trích field |
| `Citation` | nguồn, Item, `chunk_id`, 200 ký tự đầu của chunk | `postprocess.extract_citations` |
| `FinanceAnswer` | kết quả cuối cùng của một câu hỏi | `graph.py`, ghép từ LLM và dữ liệu tất định |

`revenue` luôn tính bằng triệu USD, đúng như số in trong 10-K.

## 3. Control flow: một câu hỏi đi qua những node nào

Đồ thị có sáu node và hai điểm rẽ nhánh. Điểm rẽ đầu tiên đọc `intent` do `classify_intent` ghi ra. Điểm rẽ thứ hai nằm sau `price_tool` và quyết định có cần tra thêm 10-K hay không.

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

### 3.1 Bốn đường đi

| Intent | Khi nào | Đường đi | Số lần gọi LLM |
|---|---|---|---|
| `price` | chỉ cần giá, vốn hóa hoặc P/E | classify → price_tool → generate → format | 2 |
| `filing` | chỉ cần thông tin trong 10-K | classify → retrieve → generate → format | 3 |
| `both` | cần cả giá và 10-K | classify → price_tool → retrieve → generate → format | 3 |
| `out_of_scope` | công ty khác, tư vấn đầu tư, dự đoán giá, chủ đề không liên quan | classify → refuse | 1 |

Số lần gọi LLM là số lần tối thiểu. `generate_answer` có thể gọi thêm một lần khi câu trả lời về filing chưa có citation hợp lệ. Câu `price` chỉ gọi LLM hai lần vì không có chunk nào để trích field, nên `structure` bỏ qua bước gọi LLM.

Câu "Should I buy NVIDIA stock now?" dừng ở `refuse`. Trace thật của câu này chỉ có `classify_intent` và `refuse`, mất khoảng 0,25 giây, không đụng tới giá, Chroma hay bước sinh câu trả lời.

### 3.2 Mỗi node làm gì

State của đồ thị là một `TypedDict` tên `GraphState`. Mỗi node đọc một vài khóa và ghi lại một vài khóa.

| Node | Đọc | Ghi | Hành vi |
|---|---|---|---|
| `classify_intent` | `question` | `intent`, `tickers`, `search_query`, và khởi tạo `retrieved_chunks=[]`, `price_data=[]`, `error=None`, `short_circuit=False` | Chuẩn hóa ticker (cắt khoảng trắng, viết hoa), bỏ ticker ngoài AAPL/MSFT/NVDA. Nếu intent cần dữ liệu mà không còn ticker hợp lệ thì đổi thành `out_of_scope`. |
| `price_tool` | `tickers` | `price_data`, `error` | Lấy giá từng ticker. Ticker nào lỗi thì ghi lý do vào `error`, các ticker còn lại vẫn chạy bình thường. |
| `retrieve` | `search_query`, `tickers` | `retrieved_chunks` | Với mỗi ticker, `Retriever` lấy 30 chunk ứng viên từ Chroma (lọc theo ticker ngay trong truy vấn), cross-encoder `ms-marco-MiniLM-L-6-v2` chấm lại từng cặp câu hỏi và chunk, rồi giữ 5 chunk điểm cao nhất (k = 5 vẫn là số chunk gửi cho LLM). Đặt `FINASSIST_RERANKER=none` thì bỏ bước chấm lại và trả top-5 của vector, đúng như trước đây. Chi tiết ở mục 4.5. |
| `generate_answer` | câu hỏi gốc, chunk, giá | `answer`, và có thể `error`, `short_circuit` | Gọi LLM với context gồm chunk và khối `PRICE DATA`. Với câu cần filing, kiểm tra citation và thử lại nếu thiếu (xem mục 5.2). |
| `format_output` | toàn bộ state | `output` (`FinanceAnswer`) | Gọi LLM trích field, rồi ghi đè citation và giá bằng dữ liệu tất định. |
| `refuse` | không đọc gì | `answer`, `output` | Trả mẫu từ chối song ngữ cố định. |

`search_query` luôn có giá trị sau `classify_intent`: bằng câu hỏi gốc, trừ khi người dùng hỏi bằng ngôn ngữ khác tiếng Anh (xem mục 4.4).

## 4. Data flow: dữ liệu đến từ đâu và đi đâu

### 4.1 Nạp 10-K, chạy trước khi hỏi

```mermaid
flowchart TD
    SEC["SEC EDGAR<br/>10-K mới nhất"] --> Fetch["fetch_filings.py"]
    Fetch --> Raw["data/raw/TICKER.html<br/>và TICKER.json"]
    Raw --> Text["html_to_text<br/>bỏ script, style, ix:header"]
    Text --> Split["split_items<br/>giữ Item 1, 1A, 7, 7A, 8, 15"]
    Split --> Chunk["chunk_filing<br/>1.000 ký tự, chồng 150"]
    Chunk --> Embed["embedding bge-small-en<br/>chạy cục bộ"]
    Embed --> Chroma[("data/chroma")]
```

Mỗi chunk có `chunk_id` ổn định, ví dụ `AAPL-2025-1A-012` (ticker, năm tài chính, Item, số thứ tự trong Item). Chroma dùng chính `chunk_id` làm khóa nên chạy `finassist ingest` lần hai không tạo thêm bản sao: kho có 1.162 chunk (AAPL 256, MSFT 451, NVDA 455) và con số này không đổi sau mỗi lần nạp.

Hai bước của ingestion có công tắc riêng, cả hai tắt mặc định vì phép đo không ủng hộ chúng (mục 7.6).

- `FINASSIST_TABLE_ROWS=1` làm `html_to_text` đổi mỗi hàng `<tr>` thành một dòng, các ô nối bằng ` | `, ví dụ `Data Center | 193,737 | 115,186 | 47,525`. Ở dạng mặc định mỗi ô là một dòng riêng và nhãn hàng tách khỏi con số.
- `FINASSIST_EMBED_PREFIX=1` embed thêm tiền tố kiểu `AAPL FY2025 Item 8` trước nội dung chunk. Chroma vẫn chỉ lưu nội dung gốc, nên LLM không thấy tiền tố.

Đổi chunking làm `chunk_id` đổi theo, nên sau khi bật một trong hai biến phải chạy `uv run finassist ingest --rebuild`. Lệnh này xóa collection rồi nạp lại. Chạy `ingest` thường sau khi đổi sẽ để lại chunk cũ trong kho. Khi thử nghiệm, kho khác được chọn bằng `FINASSIST_CHROMA_DIR`, vì vậy `data/chroma` không bị đụng tới. Kho theo hàng bảng có 1.168 chunk (AAPL 261, MSFT 452, NVDA 455), nhiều hơn kho chuẩn 6 chunk.

Ba chi tiết của 10-K thật đã ảnh hưởng tới parser:

- 10-K liệt kê mỗi Item hai lần, một lần trong mục lục và một lần ở phần thân. `split_items` giữ đoạn dài nhất của mỗi Item, tức là phần thân. Dòng bắt đầu bằng "Item 1A" nhưng nằm giữa một câu tham chiếu (ví dụ "see Item 1A of this Form") bị loại bằng `_is_heading`.
- Microsoft in lại tiêu đề "PART II / Item N" ở đầu mỗi trang, làm một Item bị cắt thành nhiều phần. Parser gộp các tiêu đề lặp liên tiếp của cùng một Item.
- NVIDIA để phần lớn báo cáo tài chính ở Item 15, còn Item 8 chỉ có một đoạn ngắn. Vì vậy Item 15 nằm trong danh sách giữ lại.

### 4.2 Trả lời một câu hỏi `both`

Ví dụ lấy từ một trace LangSmith thật: "What is NVIDIA's current stock price and how do export controls threaten it?"

```mermaid
sequenceDiagram
    participant U as Người dùng
    participant G as Đồ thị LangGraph
    participant L as LLM
    participant P as price_tool
    participant R as Retriever
    participant V as Chroma
    participant X as postprocess

    U->>G: question
    G->>L: phân loại (prompt hệ thống + câu hỏi)
    L-->>G: intent=both, tickers=[NVDA]
    G->>P: get_stock_price(NVDA)
    P-->>G: PriceInfo (giá, vốn hóa, P/E, stale=false)
    G->>R: retrieve(search_query, [NVDA])
    R->>V: search(search_query, NVDA, k=30)
    V-->>R: 30 Chunk ứng viên kèm khoảng cách
    R->>R: cross-encoder chấm lại 30 cặp, giữ 5
    R-->>G: 5 Chunk (score vẫn là khoảng cách Chroma)
    G->>L: context = 5 chunk + khối PRICE DATA, kèm câu hỏi gốc
    L-->>G: answer có marker [chunk_id]
    G->>X: extract_citations(answer, chunks)
    X-->>G: citation hợp lệ (đã đối chiếu với chunk vừa retrieve)
    G->>L: trích field từ answer và các chunk được trích dẫn
    L-->>G: LLMExtraction
    G->>X: ground_numbers(output, chunks)
    X-->>G: FinanceAnswer đã bỏ số không có trong chunk
    G-->>U: FinanceAnswer, giá và citation gắn bằng mã tất định
```

Thời gian từng bước trong trace đó:

| Bước | Thời gian | Dữ liệu ra |
|---|---|---|
| `classify_intent` | 0,30 giây | `intent=both` |
| `price_tool` | 1,67 giây | giá NVDA 215,38 USD, vốn hóa khoảng 5,2 nghìn tỷ USD, P/E 27,54 |
| `retrieve` | 0,74 giây | 5 chunk: `NVDA-2026-1A-126`, `1A-106`, `15-085`, `15-025`, `15-020` |
| `generate_answer` | 5,50 giây | câu trả lời có marker citation |
| `format_output` | 5,27 giây | `FinanceAnswer` hoàn chỉnh |
| Tổng | 13,65 giây | |

Trace này được ghi trước khi thêm reranker, nên bước `retrieve` 0,74 giây là top-5 của vector. Trong bộ đo truy hồi, reranker thêm khoảng 120 đến 140 ms mỗi lần truy hồi (p50 từ 12 ms lên 134 đến 149 ms), nhỏ so với hai bước còn lại. Hai bước chiếm gần hết thời gian đều là lời gọi LLM. URL trace và bản text của cây trace nằm ở [`evidence/links.md`](evidence/links.md) và [`evidence/trace_tree.md`](evidence/trace_tree.md). Giá trong trace là ảnh chụp tại thời điểm đó, vì vậy khi demo hãy đọc `as_of` của lần chạy hiện tại.

### 4.3 LLM nhìn thấy gì

Ba lời gọi LLM nhận ba loại đầu vào khác nhau.

- Phân loại: prompt hệ thống mô tả bốn intent và danh sách ba ticker, cùng một tin nhắn dạng `QUESTION` rồi đến câu hỏi. Kết quả trả về theo schema `IntentResult`.
- Trả lời: prompt hệ thống ra luật (chỉ dùng context, mỗi khẳng định về filing phải kết thúc bằng `[chunk_id]`, không tư vấn đầu tư, trả lời bằng ngôn ngữ của người hỏi), tiếp theo là context và câu hỏi gốc. Context gồm từng chunk có tiêu đề `[chunk_id] (ticker năm Item)` và, nếu có, một khối `PRICE DATA` ghi rõ `stale`.
- Trích field: câu hỏi, câu trả lời và các chunk mà câu trả lời đã trích dẫn. Nếu câu trả lời không trích chunk nào thì dùng toàn bộ chunk. Kết quả theo schema `LLMExtraction`.

### 4.4 Câu hỏi tiếng Việt

Embedding `bge-small-en-v1.5` và 10-K đều là tiếng Anh. Nếu tìm bằng nguyên văn câu tiếng Việt, khoảng cách tới các đoạn đúng lớn tới mức chúng không lọt vào top-k. Khi đo thử với câu rủi ro kiểm soát xuất khẩu của NVIDIA, top-5 không có chunk Item 1A nào (khoảng cách 0,71 đến 0,74), trong khi câu tiếng Anh tương đương cho khoảng cách 0,26.

Vì vậy bộ phân loại trả thêm hai trường:

- `language`: `en` hoặc `other`.
- `search_query`: bản viết lại bằng tiếng Anh, chỉ có khi `language` là `other`.

```mermaid
flowchart LR
    Q["Câu hỏi tiếng Việt"] --> C["classify_intent"]
    C -->|"language = other"| SQ["search_query tiếng Anh"]
    SQ --> R["retrieve"]
    Q --> G["generate_answer"]
    R --> G
    G --> A["Câu trả lời tiếng Việt"]
```

`retrieve` dùng `search_query`, còn `generate_answer` vẫn nhận câu hỏi gốc nên câu trả lời giữ ngôn ngữ của người hỏi. Với câu tiếng Anh, đồ thị bỏ qua `search_query` và tìm bằng câu gốc. Nhờ vậy điểm RAGAS baseline, vốn đo bằng câu tiếng Anh, không bị ảnh hưởng. Sau khi sửa, câu NVIDIA ở trên lấy được 5/5 chunk Item 1A với khoảng cách 0,27 đến 0,47.

### 4.5 Retriever bên trong

`Retriever` (`src/finassist/retrieval.py`) có hai tầng tùy chọn trên đầu truy vấn vector, và ba biến môi trường điều khiển chúng.

```mermaid
flowchart TD
    Q["search_query và một ticker"] --> Mode{"FINASSIST_RETRIEVAL"}
    Mode -->|"dense, mặc định"| D["Chroma: top-N theo vector<br/>N = FINASSIST_CANDIDATES, mặc định 30"]
    Mode -->|"hybrid"| H["top-N vector và top-N BM25<br/>gộp bằng RRF"]
    D --> Rr{"FINASSIST_RERANKER"}
    H --> Rr
    Rr -->|"cross-encoder, mặc định"| X["chấm lại từng cặp<br/>câu hỏi và chunk"]
    Rr -->|"none"| T["giữ 5 chunk đầu"]
    X --> T
    T --> Out["5 chunk cho ticker này"]
```

Ba công tắc và giá trị mặc định:

- `FINASSIST_RETRIEVAL`: `dense` hoặc `hybrid`. Mặc định `dense`. Giá trị khác bị từ chối khi `Retriever` được dựng, tức ở câu hỏi đầu tiên cần truy hồi.
- `FINASSIST_RERANKER`: tên một cross-encoder của `sentence-transformers`, hoặc `none`. Mặc định `cross-encoder/ms-marco-MiniLM-L-6-v2` (khoảng 90 MB, tải một lần).
- `FINASSIST_CANDIDATES`: số ứng viên lấy trước khi chấm lại. Mặc định 30.

Cấu hình `dense` kèm `FINASSIST_RERANKER=none` trả đúng kết quả của `FilingStore.search(query, ticker, 5)`. Cross-encoder chỉ sắp xếp lại các ứng viên đã có, nên một chunk nằm ngoài 30 ứng viên của vector không bao giờ lên được top-5. `Chunk.score` giữ nghĩa khoảng cách Chroma và reranker không ghi đè nó, nên citation và bảng chunk trong giao diện vẫn đọc như trước. `Retriever` dựng ở lần truy hồi đầu tiên, vì thế mô hình reranker cũng được tải và nạp ở lần đầu đó.

## 5. Ai quyết định điều gì

LLM được dùng cho phần ngôn ngữ: hiểu câu hỏi, viết câu trả lời, đọc văn bản để trích số. Những phần có thể kiểm bằng mã thì mã làm.

```mermaid
flowchart LR
    subgraph llm["Do LLM tạo"]
        I["intent, tickers,<br/>search_query"]
        A["answer"]
        E["revenue, eps,<br/>risk_factors"]
    end

    subgraph code["Do mã tất định tạo"]
        C["citations"]
        S["stock_price,<br/>stock_prices"]
        T["ticker, fiscal_year"]
        N["ground_numbers"]
    end

    I -->|"graph lọc ticker"| Route["đường đi trong đồ thị"]
    A --> Out["FinanceAnswer"]
    E --> N
    N -->|"giữ hoặc đặt null"| Out
    C --> Out
    S --> Out
    T --> Out
```

| Trường của `FinanceAnswer` | Nguồn | Ghi chú |
|---|---|---|
| `intent` | LLM, sau khi graph lọc | Intent cần dữ liệu mà không còn ticker hợp lệ thì thành `out_of_scope` |
| `answer` | LLM, hoặc mẫu cố định | Mẫu cố định gồm từ chối, không đủ thông tin, không có giá |
| `citations` | mã | Chỉ tạo từ marker `[chunk_id]` có trong các chunk vừa retrieve của lần chạy này |
| `stock_price`, `stock_prices` | mã (`price_tool`) | `stock_price` chỉ có khi đúng một ticker; nhiều ticker thì dùng `stock_prices` |
| `ticker`, `fiscal_year` | mã | Nếu mọi chunk thuộc một ticker thì lấy ticker và năm lớn nhất của chunk. Nhiều ticker thì `fiscal_year` là null. Câu chỉ có một ticker mà output chưa có ticker thì điền từ state. |
| `revenue`, `eps`, `risk_factors` | LLM trích | `revenue` và `eps` còn phải qua `ground_numbers` |

### 5.1 Citation

Câu trả lời của LLM chứa marker dạng `[AAPL-2025-1A-012]`. `extract_citations` đọc các marker này và chỉ giữ lại marker có `chunk_id` nằm trong danh sách chunk đã retrieve của chính lần chạy đó, loại trùng và giữ thứ tự xuất hiện. LLM có thể viết ra một ID nghe hợp lý nhưng không được retrieve; ID đó không bao giờ thành citation.

### 5.2 Vòng kiểm tra trong `generate_answer`

```mermaid
flowchart TD
    Start["generate_answer"] --> NeedF{"Intent cần filing?"}
    NeedF -->|"có, nhưng không có chunk"| Insuff["INSUFFICIENT_TEXT<br/>short_circuit"]
    NeedF -->|"intent là price, không có giá"| NoPrice["PRICE_UNAVAILABLE_TEXT<br/>short_circuit"]
    NeedF -->|"các trường hợp còn lại"| Gen["gọi LLM sinh câu trả lời"]
    Gen --> Check{"Cần filing và<br/>chưa có citation hợp lệ?"}
    Check -->|"không"| Done["answer, kèm ghi chú thiếu giá nếu có"]
    Check -->|"có, lần thử đầu"| Gen
    Check -->|"có, lần thử thứ hai"| Fail["INSUFFICIENT_TEXT<br/>error = missing_citation<br/>short_circuit"]
```

Khi `short_circuit` là true, `format_output` bỏ qua lời gọi LLM trích field và trả ngay một `FinanceAnswer` chỉ có `answer`, `intent`, `ticker` (nếu chỉ có một ticker) và các giá đã lấy được. Cờ này là một khóa riêng trong state, không suy ra từ nội dung câu trả lời, vì một câu trả lời thật hoàn toàn có thể bắt đầu bằng đúng cụm "Không đủ thông tin".

Câu hỏi có hai ticker mà một ticker thiếu giá vẫn được trả lời. Câu trả lời được nối thêm một ghi chú `[Giá hiện không khả dụng / price unavailable: MSFT]` với tên các ticker thiếu.

### 5.3 Kiểm tra số liệu

`ground_numbers` nhận `revenue` và `eps` do LLM trích, rồi kiểm xem chuỗi số tương ứng (ví dụ `416,161` hoặc `4.90`) có xuất hiện trong các chunk đã retrieve hay không. Phép kiểm chỉ chấp nhận một số đứng riêng, không phải một phần của số dài hơn: `6,161` không được tính là có mặt trong `416,161`. Nếu số không có, trường đó bị đặt về null.

Phép kiểm này trả lời câu hỏi "con số có nằm trong context không", không trả lời "con số có đúng nghĩa không". Mục 8 có một trường hợp thật cho thấy khác biệt đó.

## 6. Khi có sự cố

| Sự cố | Hành vi | Nơi xử lý |
|---|---|---|
| Không nạp được model reranker (chưa có trong cache và không có mạng, hoặc cache hỏng) | Ghi cảnh báo `reranker ... unavailable`, retriever dùng top-5 vector thuần. Retriever được cache nên các câu sau không thử lại. | `retrieval.make_retriever` |
| Router LLM không phản hồi | Lỗi nổi lên khỏi đồ thị. CLI in một dòng `error:` kèm gợi ý dùng `--replay`; Streamlit hiện `st.error`. | `cli.py`, `streamlit_app.py` |
| yfinance lỗi, không có mạng hoặc quá `FINASSIST_PRICE_TIMEOUT` giây (mặc định 8) | Dùng giá trong cache, đánh dấu `stale=true`. Lời gọi chạy trong thread daemon, hết giờ thì ném `TimeoutError` và ghi cảnh báo như mọi lỗi khác. Timeout áp dụng cho từng ticker, nên câu hỏi giá ba ticker có thể chờ tới ba lần timeout. Giá trị `FINASSIST_PRICE_TIMEOUT` <= 0, nan, inf hoặc không phải số thì dùng mặc định 8 và ghi cảnh báo. | `tools.get_stock_price`, `tools._call_with_timeout` |
| Không có cả mạng lẫn cache | Ném `PriceUnavailable`. Câu `price` trả `PRICE_UNAVAILABLE_TEXT`; câu `both` vẫn trả phần filing kèm ghi chú thiếu giá. | `price_tool`, `generate_answer` |
| File `prices.json` hỏng hoặc sai định dạng | Bỏ qua kèm cảnh báo trong log; lần lấy giá live kế tiếp ghi đè lại file | `tools._read` |
| Ghi cache bị gián đoạn | Ghi qua file tạm rồi `os.replace`, nên file cũ không bao giờ ở trạng thái ghi dở | `tools._write_atomic` |
| Không retrieve được chunk nào | `INSUFFICIENT_TEXT`, không gọi LLM | `generate_answer` |
| LLM trả lời về filing mà không có citation hợp lệ sau hai lần | `INSUFFICIENT_TEXT`, `error=missing_citation` | `generate_answer` |
| Trích field có cấu trúc thất bại | Đọc lại từ tham số tool call thô, từng field một; nếu vẫn không được thì dùng extraction rỗng và ghi cảnh báo | `chains.coerce_extraction` |
| Số trích ra không có trong chunk | Đặt `revenue` hoặc `eps` về null | `ground_numbers` |
| Chroma rỗng (quên chạy `ingest`) | Câu filing đầu tiên ném `EmptyStoreError` ("The filings store is empty. Build it first: uv run finassist ingest"). CLI in một dòng không kèm gợi ý router; Streamlit hiện lỗi mà không nhắc ô Replay. Câu hỏi chỉ về giá vẫn chạy vì retriever dựng ở lần truy hồi đầu. | `retrieval.make_retriever`, `cli.py`, `streamlit_app.py` |
| Cần demo khi không có mạng | `finassist ask x --replay runs/demo.json`, hoặc ô Replay trong Streamlit. Replay không dựng app và không gọi dịch vụ nào. | `cli.py`, `streamlit_app.py` |

## 7. Các quyết định kiến trúc

### 7.1 Đồ thị có nhánh thay vì agent tự chọn công cụ

Bối cảnh: câu hỏi thuộc bốn loại với chi phí và rủi ro khác nhau. Câu xin tư vấn đầu tư không được chạm tới dữ liệu nào. Câu hỏi về giá không cần 10-K.

Quyết định: dùng LangGraph `StateGraph` với hai cạnh điều kiện, intent do một lời gọi LLM phân loại ra, còn đường đi do mã quyết định.

Đã loại: agent kiểu ReAct, nơi LLM tự chọn công cụ ở mỗi bước. Đường đi khi đó khó đoán và khó đánh giá, và một demo có hỏi "câu này sẽ gọi công cụ nào" không có câu trả lời chắc chắn. Pipeline tuyến tính cũng bị loại vì nó không biểu diễn được nhánh "từ chối mà không truy hồi".

Cái giá: thêm một loại câu hỏi mới nghĩa là thêm node và cạnh.

### 7.2 Đồ thị chỉ biết `Deps`, không biết dịch vụ thật

Quyết định: `graph.py` nhận một dataclass `Deps` gồm năm hàm. `chains.py` hiện thực chúng bằng LLM, Chroma và yfinance thật.

Lý do: logic điều hướng, thử lại, chặn câu trả lời thiếu citation đều là mã thường, và test chúng bằng hàm giả nhanh hơn nhiều so với gọi LLM. Bộ unit test của dự án không gọi mạng hay LLM; các test cần router nằm sau cờ `RUN_INTEGRATION=1`.

Cái giá: có một lớp gián tiếp, và chữ ký của năm hàm phải khớp ở cả hai phía.

### 7.3 Citation do mã tạo, LLM chỉ viết marker

Đã loại: để LLM trả về danh sách đối tượng citation. LLM có thể bịa ID, chép sai Item hoặc bịa đoạn trích mà không có gì phát hiện.

Quyết định: LLM viết `[chunk_id]` vào câu trả lời, mã đối chiếu từng ID với các chunk vừa retrieve rồi mới dựng `Citation` (nguồn, Item, 200 ký tự đầu của chunk) từ chính chunk đó.

Cái giá: phụ thuộc LLM chịu viết marker đúng định dạng. `postprocess.marker_ids` nhận một hoặc nhiều ID trong mỗi cặp ngoặc (`[A]`, `[A, B]`, `[A; B]`, `[A][B]`), miễn là nội dung cặp ngoặc chỉ gồm ID và dấu phân cách. Cặp ngoặc có chữ khác, như `[see A]` hay ghi chú "price unavailable", không được tính, nên ghi chú giá không thể giả làm citation.

### 7.4 Giá đến từ công cụ, không đến từ LLM

LLM không biết giá hôm nay. Đồ thị gọi `get_stock_price`, đưa kết quả vào context cho LLM diễn đạt, và khi dựng output thì ghi đè `stock_price` bằng dữ liệu từ công cụ, bất kể LLM trích ra gì. Mỗi giá mang theo `as_of` và `stale` để người đọc biết nó cũ đến đâu.

### 7.5 Từ chối dựa vào bộ phân loại

Việc từ chối câu ngoài phạm vi, kể cả xin tư vấn đầu tư, dựa hoàn toàn vào intent `out_of_scope` của bộ phân loại, cộng với luật "không tư vấn đầu tư" trong prompt trả lời. Không có danh sách từ khóa chặn hay lớp kiểm tra thứ hai. Lý do là câu hỏi tự nhiên có quá nhiều cách diễn đạt, và một danh sách từ khóa vừa chặn nhầm vừa bỏ sót. Cái giá là nếu bộ phân loại xếp nhầm một câu xin tư vấn thành `price`, không còn lớp nào chặn nó. `tests/test_advice_refusal.py` chạy bộ phân loại thật trên 14 cách hỏi xin lời khuyên hoặc dự đoán (tiếng Anh và tiếng Việt): cả 14 đều bị từ chối, và 9 câu đối chứng (gồm 3 câu ghép khó hơn) giữ đúng intent. Kết quả này là một lần đo trên 14 câu và không bảo đảm cho các cách diễn đạt khác.

### 7.6 Các quyết định còn lại

| Quyết định | Đã loại | Lý do | Cái giá |
|---|---|---|---|
| Chia chunk theo Item rồi mới theo độ dài | Cắt đều toàn bộ filing | Đoạn rủi ro và bảng tài chính không bị trộn vào một chunk; mỗi chunk biết mình thuộc Item nào | Parser phải xử lý định dạng riêng của từng công ty (mục 4.1) |
| Chroma với bộ lọc ticker trong truy vấn | FAISS | Chroma lọc ngay trong truy vấn. FAISS của LangChain lọc sau khi tìm `fetch_k` ứng viên, nên với kho ba công ty phải tăng `fetch_k` để chắc chắn đủ k kết quả cho một ticker | Thêm một dịch vụ lưu trữ cục bộ |
| Embedding `bge-small-en-v1.5` chạy cục bộ | Embedding của OpenAI | Không cần thêm khóa API, filing không rời khỏi máy. Anthropic không có API embedding | Chỉ tiếng Anh, nên cần `search_query` cho câu tiếng Việt |
| Top-k tính riêng cho từng ticker | Một truy vấn chung cho cả kho | Câu so sánh hai công ty có đủ context của cả hai | Prompt dài hơn khi hỏi nhiều ticker |
| `search_query` chỉ dùng khi `language = other` | Viết lại mọi câu hỏi | Giữ nguyên hành vi, nên điểm RAGAS baseline vẫn có giá trị | Câu tiếng Anh diễn đạt vụng không được viết lại |
| Ghim model `kr/claude-sonnet-5.5` | Các model `auto/*` của router | `auto/*` thường trả lỗi giới hạn tần suất từ upstream | Phụ thuộc một model cụ thể |
| Record và replay | Chỉ chạy live | Demo vẫn chạy khi mất mạng hoặc router lỗi | Replay luôn hiện câu đã ghi, không phải câu mới gõ; Streamlit ghi rõ điều này |
| Reranker cục bộ `ms-marco-MiniLM-L-6-v2` trên 30 ứng viên của vector (đã dùng) | Chỉ vector top-5 | Trên bộ đo 25 câu, any@5 từ 48% lên 64% (4 câu), all@5 từ 28% lên 36%, nhóm combined từ 0% lên 40%. Vượt ngưỡng 15 điểm đặt trước, nhưng chỉ hơn ngưỡng 1 điểm | Thêm khoảng 120 đến 140 ms mỗi lần truy hồi và một lần tải mô hình khoảng 90 MB. MRR gần như không đổi (0,371 lên 0,376), all@5 của nhóm risk giảm từ 10% xuống 0%, và q04 kém đi ở RAGAS (mục 8) |
| Hybrid BM25 và RRF (đã đo, không dùng; `FINASSIST_RETRIEVAL=hybrid` còn trong mã) | Hybrid làm mặc định | Hybrid kèm MiniLM cho any@5 64% và all@5 36%, bằng đúng vector kèm MiniLM (tăng 0 điểm, ngưỡng là 5). Hybrid không reranker chỉ 56% và 32%, nhóm combined về 0%. BM25 không cứu q11, q15, q16 vì bằng chứng dùng từ khác với câu hỏi | Giữ thêm mã, thư viện `rank-bm25` và một tùy chọn không dùng |
| Chunk theo hàng bảng và tiền tố embed (đã đo, không dùng; `FINASSIST_TABLE_ROWS`, `FINASSIST_EMBED_PREFIX` còn trong mã) | Bật hai tùy chọn này mặc định | Cấu hình tốt nhất của giai đoạn này (hàng bảng, tiền tố và MiniLM) chỉ đạt any@5 56% và all@5 24%, thấp hơn 8 và 12 điểm. Làm hỏng q07, q09, q23 và không cứu q05, q10, q24. Chưa có phép đo tiền tố riêng trên văn bản phẳng | Cần `ingest --rebuild` và một kho riêng khi thử lại |
| `bge-reranker-base` (đã đo, không dùng) | Reranker lớn hơn thay MiniLM | any@5 56% (tăng 8 điểm, dưới ngưỡng 15), all@5 36%, p50 809 ms. MRR cao nhất (0,444) và nhóm risk giảm từ 60% xuống 50% ở any@5 | Chậm hơn khoảng năm lần mà không thêm câu nào vào top-5 |

## 8. Giới hạn đã biết

1. Retrieval còn bỏ lỡ bằng chứng, và mục tiêu đặt ra chưa đạt. Trên bộ đo 25 câu (không gọi LLM), cấu hình hiện tại đạt any@5 64% và all@5 36%, so với 48% và 28% của vector thuần. Mục tiêu là 70% và 40%, thiếu 2 và 1 câu. Chín câu vẫn trượt ở top-5: q04, q05, q10 (số liệu), q11, q15, q16 (rủi ro), q22, q24, q25 (kết hợp). Ba câu q11, q22, q25 không có chunk bằng chứng nào trong 30 ứng viên, nên reranker không cứu được. Ví dụ điển hình là lợi nhuận ròng của Microsoft (q05): chunk chứa `133,749` nằm ở hạng 12 và hệ thống trả lời rằng nó không thấy số liệu. Phần RAGAS cho kết quả lẫn lộn. Context precision tăng từ 0,309 lên 0,342 và answer relevancy từ 0,687 lên 0,766 (phần lớn vì ít câu bị từ chối hơn). Faithfulness (0,939) và context recall (0,387 lên 0,400) không đổi trong phạm vi nhiễu 0,02 đến 0,03 giữa hai lần chạy vector thuần. q21 và q03 nay trả lời được; q04 tệ đi (recall từ 1,0 xuống 0,0); q05, q04, q24 vẫn thiếu bằng chứng trong ngữ cảnh. Vì vậy mức cải thiện nên đọc là nhỏ và chưa kết luận được ở phần RAGAS. Số liệu đầy đủ nằm ở [`../reports/retrieval_experiments.md`](../reports/retrieval_experiments.md).
2. `ground_numbers` không hiểu ý nghĩa của con số. Ở q10 và q20 của baseline, hệ thống trả lời "total revenue 139,297" cho NVIDIA. Con số đó đến từ chunk `NVDA-2026-7-029`, một bảng theo mảng không có tiêu đề cột, và nó có thật trong chunk nên được giữ lại. Doanh thu thật là 215,938.
3. Embedding chỉ có tiếng Anh. `search_query` giúp câu tiếng Việt, nhưng câu hỏi về bảng số liệu bằng tiếng Việt vẫn có thể trượt chunk.
4. Từ chối chỉ có một lớp, như mục 7.5.
5. LLM chạy ở `temperature=0` nên lần thử lại trong `generate_answer` có thể cho đúng câu trả lời cũ. Chưa có phép đo cho thấy lần thử thứ hai cải thiện được gì.
6. Judge của RAGAS dùng cùng model với ứng dụng, nên điểm faithfulness có thể cao hơn thực tế.

Các hướng còn lại, rút ra từ những gì đã đo, nằm trong [`NEXT_STEP.md`](NEXT_STEP.md).

## 9. Chạy và quan sát

```bash
uv run finassist ingest                    # nạp 10-K vào Chroma, chạy một lần
uv run finassist ingest --rebuild          # xóa kho rồi nạp lại, cần khi đổi cách chunk
uv run finassist ask "..."                 # chạy live
uv run finassist ask "..." --record runs/demo.json   # chạy live và lưu kết quả
uv run finassist ask x --replay runs/demo.json       # phát lại offline, câu hỏi truyền vào bị bỏ qua
uv run streamlit run app/streamlit_app.py  # giao diện; ô Replay dùng runs/demo.json
uv run python -m eval.retrieval_eval --label "tên lần chạy" [--append reports/retrieval_experiments.md]   # bộ đo truy hồi, không gọi LLM
```

Biến môi trường của phần truy hồi (đặt trong `.env` hoặc shell):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `FINASSIST_RETRIEVAL` | `dense` | `dense` hoặc `hybrid` (thêm BM25 và RRF) |
| `FINASSIST_RERANKER` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | tên cross-encoder, hoặc `none` để bỏ bước chấm lại |
| `FINASSIST_CANDIDATES` | `30` | số ứng viên lấy trước khi chấm lại |
| `FINASSIST_TOP_K` | `5` | số chunk mỗi ticker gửi cho LLM |
| `FINASSIST_EMBED_PREFIX` | `0` | `1` thì embed thêm tiền tố ticker, năm, Item (cần `ingest --rebuild`) |
| `FINASSIST_TABLE_ROWS` | `0` | `1` thì mỗi hàng bảng thành một dòng (cần `ingest --rebuild`) |
| `FINASSIST_CHROMA_DIR` | `data/chroma` | thư mục kho Chroma, dùng để thử một kho khác mà không đụng kho chuẩn |

Bộ đo đọc 25 câu có `evidence` trong `eval/golden_set.json`, in any@K, all@K, MRR và độ trễ theo nhóm, và nhận thêm `--mode`, `--reranker`, `--candidates`, `--show-misses` để thử một cấu hình mà không đổi `.env`. Lần chạy đầu với reranker tải mô hình về máy, sau đó chạy offline.

`runtime.ask` gọi đồ thị với `run_name=finassist`, tag mặc định `demo` và metadata `app_version`. Khi các biến `LANGSMITH_*` có trong `.env`, LangChain tự gửi trace gồm từng node, từng lời gọi LLM, đầu vào và đầu ra của `price_tool` và danh sách chunk của `retrieve`. Lệnh `uv run python scripts/trace_link.py` in URL của trace mới nhất.

`eval/run_ragas.py` (chạy bằng `uv run python -m eval.run_ragas --tag <tên>`, `--tag` là bắt buộc và không còn tag mặc định) dùng cùng `build_app` và gắn tag `eval`, nên các lần chạy đánh giá tách được khỏi các lần demo trong LangSmith.

## 10. Tài liệu liên quan

- [`evidence/links.md`](evidence/links.md) và [`evidence/trace_tree.md`](evidence/trace_tree.md): URL LangSmith và bản text của hai trace thật.
- [`../reports/ragas_baseline.md`](../reports/ragas_baseline.md): điểm baseline của hệ thống vector thuần, các ca thất bại và phần caveats.
- [`../reports/retrieval_experiments.md`](../reports/retrieval_experiments.md): từng giai đoạn thí nghiệm truy hồi, quyết định giữ hoặc bỏ, và so sánh RAGAS.
- [`../reports/ragas_retrieval_v2.md`](../reports/ragas_retrieval_v2.md): điểm RAGAS của cấu hình hiện tại.
- [`KNOWLEDGE_CHECK.md`](KNOWLEDGE_CHECK.md): câu trả lời mẫu cho Giải thích, Dự đoán, Gỡ lỗi và Bảo vệ lựa chọn, có tham chiếu `file:line` vào mã.
- [`NEXT_STEP.md`](NEXT_STEP.md): các hướng cải thiện còn lại sau thí nghiệm truy hồi.
