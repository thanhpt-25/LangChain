# Nhật ký đóng góp

## Ai đã làm gì

Repo này có một tác giả git duy nhất (`thanh.pt`): 44 commit từ 2026-10-28 đến 2026-10-29, tính đến `ad9ba10` (theo `git log`). Dự án do một người thực hiện, với sự hỗ trợ của AI, theo phương pháp spec-driven development. Người làm viết spec (`docs/superpowers/specs/2026-10-28-finance-rag-assistant-design.md`) và plan (`docs/superpowers/plans/2026-10-28-finance-rag-assistant.md`), rồi giao từng task cho subagent AI thực hiện và review, mỗi task một vài commit. Repo không có bằng chứng về đóng góp của thành viên khác, nên bảng dưới không ghi ai ngoài tác giả này.

## Bảng commit

Lấy từ `git log --format='%h %ad %an %s' --date=short`. Nhánh `feat/finance-assistant` đã được merge vào `main` ở `f7a31c8`.

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

Các commit sau `ad9ba10` chỉ chỉnh tài liệu và dọn repo; xem `git log` để biết chi tiết.

## Bài học rút ra

- Retrieval bỏ sót số liệu trong bảng tài chính nhiều nhất. Ở baseline, cả 5 case tệ nhất đều không retrieve được chunk golden (`reports/ragas_baseline.md`, phần "Common pattern"). Phần lớn các lần hụt dẫn tới hedge thay vì bịa số. Ngoại lệ là q10 và q20: hai câu này trả một số sai (139,297 thay vì 215,938) lấy từ bảng không có tiêu đề cột, và số đó vẫn qua `ground_numbers`. Ở điểm này hệ thống chưa đủ hữu ích và chưa đủ an toàn.
- Kiểm tra dữ liệu thật sớm có ích. NVDA đặt financial statements ở Item 15 và Item 8 chỉ là stub, nên phải thêm Item 15 (`676e1bf`). MSFT lặp running header "Item 8", nên phải gộp (`b24e307`).
- Cho LLM sinh citation hoặc con số rồi tin ngay là rủi ro. Phần đó được chuyển sang mã tất định (`postprocess.py`), vừa kiểm được vừa test được.
- Judge dùng cùng model với app, nên faithfulness 0.939 cần đọc cùng với Caveats.

## Bản sửa ưu tiên đã commit

`a668069` "enforce valid citations, surface price failures, ground stock_price in tool data": sau review, một câu filing không có marker `[chunk_id]` hợp lệ được thử lại một lần, rồi trả `INSUFFICIENT_TEXT` kèm `error = missing_citation` (`src/finassist/graph.py:97-101`). Trước đó hệ thống trả câu không có nguồn.

## Thông tin nhóm cần bổ sung

Repository không chứa các thông tin sau, nên nhóm cần điền:

- Tên và vai trò của các thành viên khác trong nhóm, nếu có. Thêm một dòng mỗi người vào bảng trên, kèm commit hoặc PR của họ.
- Việc làm theo cặp hoặc review chéo ngoài git, nếu có: ghi lại ở đây kèm bằng chứng (link PR, ảnh chụp review, ghi chú họp).
- Link PR, nếu nhánh `feat/finance-assistant` từng được mở PR. Repo hiện không có thông tin PR; nhánh được merge vào `main` bằng merge commit `f7a31c8`.
