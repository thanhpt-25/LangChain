# Mục lục tài liệu của FinAssist

Thư mục này là điểm bắt đầu cho người muốn hiểu, chạy thử hoặc trình bày dự án trợ lý AI tài chính `finassist`. Hệ thống chỉ hỗ trợ báo cáo 10-K và dữ liệu giá cổ phiếu của AAPL, MSFT và NVDA, và không đưa ra khuyến nghị đầu tư. Cài đặt và lệnh chạy nằm trong [`../README.md`](../README.md).

## Đọc gì theo nhu cầu

| Bạn cần | Bắt đầu ở | Sau đó đọc |
|---|---|---|
| Hiểu hệ thống trong vài phút | [`ARCHITECTURE.md`](ARCHITECTURE.md) | [`NEXT_STEP.md`](NEXT_STEP.md) để biết giới hạn hiện tại |
| Chạy hoặc trình bày demo | [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | [`evidence/links.md`](evidence/links.md), và [`../README.md`](../README.md) để cài đặt |
| Chuẩn bị trả lời câu hỏi kỹ thuật | [`KNOWLEDGE_CHECK.md`](KNOWLEDGE_CHECK.md) | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Kiểm tra bằng chứng và trace | [`evidence/trace_tree.md`](evidence/trace_tree.md) | [`evidence/links.md`](evidence/links.md) |
| Xem phạm vi, yêu cầu và quyết định ban đầu | [`superpowers/specs/2026-10-28-finance-rag-assistant-design.md`](superpowers/specs/2026-10-28-finance-rag-assistant-design.md) | kế hoạch triển khai ở dòng dưới |
| Xem cách dự án được xây và kiểm chứng | [`superpowers/plans/2026-10-28-finance-rag-assistant.md`](superpowers/plans/2026-10-28-finance-rag-assistant.md) | [`TEAM_LOG.md`](TEAM_LOG.md) |
| Xem thiết kế và kế hoạch của đợt cải thiện truy hồi | [`superpowers/specs/2026-10-29-retrieval-improvement-design.md`](superpowers/specs/2026-10-29-retrieval-improvement-design.md) | [`superpowers/plans/2026-10-29-retrieval-improvement.md`](superpowers/plans/2026-10-29-retrieval-improvement.md) và [`../reports/retrieval_experiments.md`](../reports/retrieval_experiments.md) |

## Các tài liệu chính

- [`ARCHITECTURE.md`](ARCHITECTURE.md): hệ thống nhìn từ bên ngoài, các thành phần và quan hệ giữa chúng, control flow của đồ thị LangGraph, data flow (kể cả một trace thật của câu `both` và cấu trúc bên trong `Retriever`), ai quyết định điều gì, các kiểu sự cố, quyết định kiến trúc kèm phương án đã loại và số đo, và giới hạn đã biết.
- [`KNOWLEDGE_CHECK.md`](KNOWLEDGE_CHECK.md): câu trả lời mẫu cho Giải thích, Dự đoán, Gỡ lỗi và Bảo vệ lựa chọn, dùng khi bảo vệ hoặc demo.
- [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md): kịch bản demo 8 phút, phương án dự phòng, kiểm tra an toàn và mẫu hỏi đáp.
- [`NEXT_STEP.md`](NEXT_STEP.md): kết quả đợt cải thiện truy hồi (chưa đạt mục tiêu) và các hướng còn lại, mỗi hướng có cách đo.
- [`TEAM_LOG.md`](TEAM_LOG.md): lịch sử đóng góp theo Git.
- [`evidence/`](evidence/): link LangSmith và bản text của các trace thật.

## Hai điều cần biết trước khi dùng số liệu

1. Giá cổ phiếu thay đổi theo thời điểm. Hãy đọc trường `as_of` và `stale`, và đừng lặp lại giá từ một trace cũ như giá hiện tại.
2. Các điểm RAGAS là bằng chứng của từng lần chạy, kèm nhiễu 0,02 đến 0,03 giữa hai lần chạy cùng cấu hình. Judge dùng cùng model với ứng dụng, nên hãy đọc phần Caveats trong `../reports/ragas_baseline.md` và `../reports/ragas_retrieval_v2.md` trước khi trích số. Số của bộ đo truy hồi và các thí nghiệm nằm ở `../reports/retrieval_experiments.md`.

## Liên kết với mã nguồn

Các tài liệu nêu tên file và số dòng khi cần đối chiếu với mã. Điều phối nằm ở `src/finassist/graph.py`, các LLM chain ở `src/finassist/chains.py`, truy hồi (vector, tùy chọn BM25, reranker) ở `src/finassist/retrieval.py`, còn citation và kiểm tra số liệu ở `src/finassist/postprocess.py`.
