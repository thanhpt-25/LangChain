# Nhật ký đóng góp

## Ai đã làm gì

Repo này có một tác giả git duy nhất (`thanh.pt`): 89 commit trên nhánh `feat/both-query-rewrite`, từ 2026-10-28 đến 2026-10-29 (theo `git log`, tính cả hai commit tài liệu của đợt viết lại truy vấn câu `both`). Dự án do một người thực hiện, với sự hỗ trợ của AI, theo phương pháp spec-driven development. Người làm viết spec (`docs/superpowers/specs/2026-10-28-finance-rag-assistant-design.md`) và plan (`docs/superpowers/plans/2026-10-28-finance-rag-assistant.md`), rồi giao từng task cho subagent AI thực hiện và review, mỗi task một vài commit. Repo không có bằng chứng về đóng góp của thành viên khác, nên bảng dưới không ghi ai ngoài tác giả này.

## Bảng commit

Lấy từ `git log --format='%h %ad %an %s' --date=short`. Nhánh `feat/finance-assistant` đã được merge vào `main` ở `f7a31c8`. Các nhánh `feat/retrieval-improvement` và `feat/hardening-group1` đã được merge vào `main` ở `8502fdb` và `7ab03e1`. Các dòng cuối bảng (viết lại truy vấn câu `both`) nằm trên nhánh `feat/both-query-rewrite`, chưa merge vào `main` tại thời điểm viết.

| date | who | work | evidence (commit) |
|---|---|---|---|
| 2026-10-28 | thanh.pt | Spec và plan | `2a355b9`, `dcb0edc`, `3e33b38` |
| 2026-10-28 | thanh.pt | Scaffold, config, schemas | `4d0de4d` |
| 2026-10-28 | thanh.pt | `.env` ghi đè biến shell; tách test khỏi `.env` | `e091608` |
| 2026-10-28 | thanh.pt | Parse 10-K thành chunk theo Item; bỏ cross-reference; gộp running header; thêm Item 15 | `a027cb1`, `8c52565`, `b24e307`, `676e1bf` |
| 2026-10-28 | thanh.pt | Chroma store có filter ticker | `d6180cf` |
| 2026-10-28 | thanh.pt | Tool giá tất định, cache an toàn | `72a75f2`, `381823b` |
| 2026-10-28 | thanh.pt | Citation tất định, `ground_numbers` | `c81f92f` |
| 2026-10-28 | thanh.pt | LangGraph workflow; ép citation hợp lệ; giá chỉ từ tool | `2f41cce`, `a668069` |
| 2026-10-28 | thanh.pt | Chain Claude, runtime, structured extraction bền vững | `9e0b923`, `473da92` |
| 2026-10-28 | thanh.pt | CLI `ingest`/`ask`/replay | `50cc770` |
| 2026-10-28 | thanh.pt | Golden set 30 câu viết tay | `9d41fb1` (`eval/golden_set.json`) |
| 2026-10-29 | thanh.pt | RAGAS baseline và báo cáo | `7830774` (`reports/ragas_baseline.md`) |
| 2026-10-29 | thanh.pt | UI Streamlit có Replay | `21b0561`, `f94bbcb`, `fd8ab01` |
| 2026-10-29 | thanh.pt | Sửa lỗi sau review: postprocess, CLI, schemas, graph, eval shim, scripts, golden q26 | `308d83b`, `5723efe`, `caf7c58`, `3603e20`, `b0ad410`, `6e26a90`, `a8a38e0` |
| 2026-10-29 | thanh.pt | Caveats trong báo cáo RAGAS | `be58788` |
| 2026-10-29 | thanh.pt | Script và bằng chứng LangSmith trace | `b78d35b` (`docs/evidence/`) |
| 2026-10-29 | thanh.pt | README và spec cập nhật; bộ tài liệu demo (kiến trúc, knowledge check, demo script, team log, next step) | `bd35dcc`, `9d006de`, `55a46ea` |
| 2026-10-29 | thanh.pt | Câu hỏi tiếng Việt: classifier trả `language` và `search_query`, retrieve bằng câu tiếng Anh; integration test cho Item 1A | `1a190f5`, `dcfeff9` |
| 2026-10-29 | thanh.pt | Sửa cuối: `to_record` công khai, đọc replay bằng utf-8 trong Streamlit | `5b80964` |
| 2026-10-29 | thanh.pt | Ghi nhận q10/q20 (số 139,297 sai nhưng có trong chunk) trong báo cáo và tài liệu | `75390aa`, `ba171e3` |
| 2026-10-29 | thanh.pt | Mục lục tài liệu tiếng Việt, dịch heading của spec và evidence, plan hồi cứu; cập nhật `file:line` | `f4e8131`, `af455bd` |
| 2026-10-29 | thanh.pt | Merge nhánh `feat/finance-assistant` vào `main` | `f7a31c8` |
| 2026-10-29 | thanh.pt | Viết lại `docs/ARCHITECTURE.md` | `ad9ba10` |
| 2026-10-29 | thanh.pt | README tiếng Việt, đồng bộ demo script và knowledge check với đánh số mới, làm mới team log và next step | `7cc731a`, `46e1d1c`, `3d69af6` |
| 2026-10-29 | thanh.pt | Dọn `.gitkeep`; tắt hai cảnh báo deprecation đã biết của ragas và langchain-community | `376a0e2`, `f17b7ef` |
| 2026-10-29 | thanh.pt | Spec và kế hoạch cải thiện truy hồi; giữ lần chạy lại RAGAS bên cạnh baseline đã chú giải | `d51e15f`, `becd346`, `d662d52` |
| 2026-10-29 | thanh.pt | `Retriever` (BM25, RRF, cross-encoder), cấu hình truy hồi, `make_deps` đi qua `Retriever` | `db6c9b6`, `b02ecb9`, `69b9de8` |
| 2026-10-29 | thanh.pt | Bộ đo truy hồi: nhóm bằng chứng viết tay, bộ đo, giai đoạn 0, test ranh giới | `429a51a`, `8e94797`, `dccc959` |
| 2026-10-29 | thanh.pt | Thí nghiệm giai đoạn 1 (reranker), 2 (hybrid), 3 (hàng bảng và tiền tố embed) và báo cáo | `c4b40e3`, `c4e2fdb`, `0ae3424`, `aa11e30` |
| 2026-10-29 | thanh.pt | Mặc định MiniLM, `--tag` cho RAGAS, integration test truy hồi | `3427dab` |
| 2026-10-29 | thanh.pt | Báo cáo cấu hình cuối và so sánh RAGAS | `0d4b248` |
| 2026-10-29 | thanh.pt | Tài liệu cho phần truy hồi: ARCHITECTURE và NEXT_STEP | `5be9242` |
| 2026-10-29 | thanh.pt | Làm mới README, demo script, knowledge check, team log, mục lục tài liệu cho phần truy hồi | `432f23e` |
| 2026-10-29 | thanh.pt | Sửa sau review của đợt truy hồi: fallback khi không nạp được reranker, kiểm tra `Retriever` mode, `--tag` bắt buộc; sửa ghi chú q04 và độ trễ; ghi rõ offline và nhiễu; lần chạy RAGAS thứ hai của cấu hình cuối; merge vào `main` | `218b0f5`, `894a357`, `0aefa36`, `33a28a9`, `8502fdb` |
| 2026-10-29 | thanh.pt | Đợt gia cố nhóm 1: citation nhiều id trong một cặp ngoặc (sửa regex tuyến tính), timeout cho yfinance, lỗi rõ khi kho Chroma rỗng, test thêm cách hỏi xin tư vấn đầu tư, cập nhật tài liệu và số test; merge vào `main` | `34aba2b`, `afeed27`, `063610f`, `caf091e`, `89c1afe`, `8a5db29`, `1817418`, `34b1b83`, `7ab03e1` |
| 2026-10-29 | thanh.pt | Spec và kế hoạch viết lại truy vấn câu `both` | `bc53083`, `441fa27` |
| 2026-10-29 | thanh.pt | `retrieval_query` và `resolve_intent`; bộ phân loại trả `search_query` tiếng Anh chỉ phần 10-K cho câu `both`; bộ đo truy hồi có `--classify`, `--save-rewrites`, `--rewrites` | `03020c2`, `bb4cd9d`, `461375b` |
| 2026-10-29 | thanh.pt | Thí nghiệm viết lại truy vấn và quyết định giữ (any@5 64% lên 72%, all@5 36% lên 40%); so sánh RAGAS bốn lần chạy | `cff93d8`, `1839b2a` (`reports/query_rewrites.json`, `reports/ragas_both_rewrite_*.json`) |
| 2026-10-29 | thanh.pt | Tài liệu cho phần viết lại truy vấn: ARCHITECTURE và NEXT_STEP; README, KNOWLEDGE_CHECK, DEMO_SCRIPT, TEAM_LOG, mục lục tài liệu | `7b4861d` và commit tài liệu cuối của nhánh (xem `git log`) |

## Bài học rút ra

- Retrieval bỏ sót số liệu trong bảng tài chính nhiều nhất. Ở baseline, cả 5 case tệ nhất đều không retrieve được chunk golden (`reports/ragas_baseline.md`, phần "Common pattern"). Phần lớn các lần hụt dẫn tới hedge thay vì bịa số. Ngoại lệ là q10 và q20: hai câu này trả một số sai (139,297 thay vì 215,938) lấy từ bảng không có tiêu đề cột, và số đó vẫn qua `ground_numbers`. Ở điểm này hệ thống chưa đủ hữu ích và chưa đủ an toàn.
- Kiểm tra dữ liệu thật sớm có ích. NVDA đặt financial statements ở Item 15 và Item 8 chỉ là stub, nên phải thêm Item 15 (`676e1bf`). MSFT lặp running header "Item 8", nên phải gộp (`b24e307`).
- Cho LLM sinh citation hoặc con số rồi tin ngay là rủi ro. Phần đó được chuyển sang mã tất định (`postprocess.py`), vừa kiểm được vừa test được.
- Judge dùng cùng model với app, nên faithfulness 0.939 cần đọc cùng với Caveats.
- Đo trước khi tin một cải tiến. Đợt truy hồi thử bốn thứ (reranker MiniLM, reranker bge, hybrid BM25, chunk theo hàng bảng) với ngưỡng đặt trước, và chỉ reranker MiniLM được giữ. Kết quả vẫn dưới mục tiêu (any@5 64% so với 70%, all@5 36% so với 40%), và RAGAS cho thấy faithfulness và recall chênh lệch trong khoảng nhiễu giữa các lần chạy (0.02 đến 0.03), còn answer relevancy tăng 0.08 đến 0.10, lớn hơn nhiễu đó, chủ yếu nhờ ít câu từ chối hơn, nên mức cải thiện nên đọc là nhỏ. Việc chạy lại baseline để đo nhiễu và việc ghi cả những thí nghiệm bị bỏ là phần giúp kết luận này đáng tin.
- Viết lại truy vấn chỉ cho câu `both` đưa hai mục tiêu lên mức đạt (any@5 72%, all@5 40%), nhưng bằng 5 câu và với biên bằng 0 ở all@5. Hai lần chạy RAGAS cho cấu hình mới cho thấy nhiễu của nó (recall 0.04) lớn hơn của cấu hình cũ, nên chỉ precision tăng chắc chắn. Việc đặt ngưỡng cổng trước, giữ bảng viết lại để phát lại và nói thẳng mẫu chỉ có 5 câu là phần giúp kết luận này không bị thổi phồng.
- Cache của router làm một thay đổi prompt bị bỏ qua mà không báo gì cho tới khi tin nhắn người dùng đổi thành `QUESTION (v2)`. Lỗi này lộ ra khi chạy bộ phân loại thật: kết quả không đổi sau khi sửa prompt là dấu hiệu để nghi ngờ lớp cache.

## Bản sửa ưu tiên đã commit

`a668069` "enforce valid citations, surface price failures, ground stock_price in tool data": sau review, một câu filing không có marker `[chunk_id]` hợp lệ được thử lại một lần, rồi trả `INSUFFICIENT_TEXT` kèm `error = missing_citation` (`src/finassist/graph.py:114-118`). Trước đó hệ thống trả câu không có nguồn.

## Thông tin nhóm cần bổ sung

Repository không chứa các thông tin sau, nên nhóm cần điền:

- Tên và vai trò của các thành viên khác trong nhóm, nếu có. Thêm một dòng mỗi người vào bảng trên, kèm commit hoặc PR của họ.
- Việc làm theo cặp hoặc review chéo ngoài git, nếu có: ghi lại ở đây kèm bằng chứng (link PR, ảnh chụp review, ghi chú họp).
- Link PR, nếu nhánh `feat/finance-assistant` từng được mở PR. Repo hiện không có thông tin PR; nhánh được merge vào `main` bằng merge commit `f7a31c8`.
