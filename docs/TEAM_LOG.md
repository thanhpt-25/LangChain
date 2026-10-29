# TEAM LOG

## Ai đã làm gì (nói thẳng)

Repo này có **một tác giả git duy nhất** (`thanh.pt`, 32 commit từ 2026-10-28 đến 2026-10-29 tính đến `b78d35b`, theo `git log`). Dự án do một người thực hiện, với sự hỗ trợ của AI, theo phương pháp spec-driven development: viết spec (`docs/superpowers/specs/2026-10-28-finance-rag-assistant-design.md`), viết plan (`docs/superpowers/plans/2026-10-28-finance-rag-assistant.md`), rồi giao từng task cho subagent AI thực hiện và review, mỗi task một vài commit. Không có bằng chứng trong repo về đóng góp của thành viên khác, nên không ghi đóng góp nào khác ở đây.

## Bảng commit (từ `git log --format='%ad %an %s' --date=short`)

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
| 2026-10-29 | thanh.pt | Dọn lỗi sau review: postprocess, CLI, schemas, graph, eval shim, scripts, golden q26 | `308d83b`, `5723efe`, `caf7c58`, `3603e20`, `b0ad410`, `6e26a90`, `a8a38e0` |
| 2026-10-29 | thanh.pt | Caveats trong báo cáo RAGAS | `be58788` |
| 2026-10-29 | thanh.pt | Script và bằng chứng LangSmith trace | `b78d35b` (`docs/evidence/`) |

Các tài liệu README, `docs/*.md` được thêm ở commit docs của Task 14 (xem `git log` sau commit `b78d35b`).

## Learned (điều đã học)

- Số liệu trong bảng tài chính bị retrieval bỏ sót nhiều nhất: ở baseline, 5 trong 5 case tệ nhất đều không retrieve được chunk golden (`reports/ragas_baseline.md`, "Common pattern"). Hệ thống an toàn ở chỗ hedge thay vì bịa, nhưng chưa đủ hữu ích.
- Kiểm tra dữ liệu thật sớm có ích: NVDA đặt financial statements ở Item 15 và Item 8 chỉ là stub, nên phải thêm Item 15 (`676e1bf`); MSFT lặp running header "Item 8" nên phải gộp (`b24e307`).
- Cho LLM sinh citation hoặc con số rồi tin ngay là rủi ro; chuyển phần đó sang mã tất định (`postprocess.py`) và test được.
- Judge cùng model với app nên faithfulness 0.939 cần đọc cùng với Caveats.

## Priority fix committed

`a668069` "enforce valid citations, surface price failures, ground stock_price in tool data": sau review, một câu filing không có marker `[chunk_id]` hợp lệ giờ bị thử lại một lần rồi trả `INSUFFICIENT_TEXT` kèm `error = missing_citation` (`src/finassist/graph.py:95-99`), thay vì trả câu không có nguồn.

## To fill by the team (không suy ra được từ repo)

- Tên và vai trò của các thành viên khác trong nhóm, nếu có; thêm một dòng mỗi người vào bảng trên với commit/PR của họ.
- Nếu nhóm làm việc theo cặp hoặc review chéo ngoài git, ghi lại ở đây kèm bằng chứng (link PR, ảnh chụp review, ghi chú họp).
- Link PR nếu nhánh `feat/finance-assistant` được mở PR (hiện repo chưa có thông tin PR).
