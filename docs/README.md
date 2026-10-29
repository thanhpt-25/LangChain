# Hướng dẫn tài liệu của FinAssist

Thư mục này là điểm bắt đầu dành cho người muốn hiểu, chạy thử hoặc trình bày dự án trợ lý AI tài chính `finassist`. Hệ thống chỉ hỗ trợ báo cáo 10-K và dữ liệu giá cổ phiếu của AAPL, MSFT và NVDA; hệ thống không đưa ra khuyến nghị đầu tư.

## Nên đọc theo vai trò nào?

| Bạn cần làm gì? | Hãy bắt đầu ở đâu? | Sau đó đọc |
|---|---|---|
| Hiểu hệ thống trong vài phút | [`ARCHITECTURE.md`](ARCHITECTURE.md) | [`NEXT_STEP.md`](NEXT_STEP.md) để biết giới hạn hiện tại |
| Chạy hoặc trình bày demo | [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | [`evidence/links.md`](evidence/links.md), rồi `README.md` ở gốc repo để cài đặt |
| Chuẩn bị trả lời câu hỏi kỹ thuật | [`KNOWLEDGE_CHECK.md`](KNOWLEDGE_CHECK.md) | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Kiểm tra bằng chứng và trace | [`evidence/trace_tree.md`](evidence/trace_tree.md) | [`evidence/links.md`](evidence/links.md) |
| Xem phạm vi, yêu cầu và quyết định ban đầu | [`superpowers/specs/2026-10-28-finance-rag-assistant-design.md`](superpowers/specs/2026-10-28-finance-rag-assistant-design.md) | kế hoạch triển khai bên dưới |
| Hiểu cách dự án được xây dựng và kiểm chứng | [`superpowers/plans/2026-10-28-finance-rag-assistant.md`](superpowers/plans/2026-10-28-finance-rag-assistant.md) | [`TEAM_LOG.md`](TEAM_LOG.md) |

## Các tài liệu chính

- [`ARCHITECTURE.md`](ARCHITECTURE.md): sơ đồ LangGraph, state, từng node, một trace thật và lý do chọn công nghệ.
- [`KNOWLEDGE_CHECK.md`](KNOWLEDGE_CHECK.md): câu trả lời mẫu cho Giải thích, Dự đoán, Gỡ lỗi và Bảo vệ lựa chọn; phù hợp khi bảo vệ hoặc demo.
- [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md): kịch bản demo 8 phút, phương án dự phòng, kiểm tra an toàn và mẫu hỏi đáp.
- [`NEXT_STEP.md`](NEXT_STEP.md): vấn đề retrieval đã quan sát được và tiêu chí cho cải tiến tiếp theo.
- [`TEAM_LOG.md`](TEAM_LOG.md): lịch sử đóng góp có bằng chứng từ Git.
- [`evidence/`](evidence/): link LangSmith và bản ghi dạng văn bản của các trace thật.

## Hai điều cần biết trước khi sử dụng số liệu

1. Giá cổ phiếu thay đổi theo thời điểm. Luôn đọc trường `as_of` và `stale`; không lặp lại giá từ trace cũ như giá hiện tại.
2. Baseline RAGAS là bằng chứng về một lần chạy, không phải tuyên bố chất lượng tuyệt đối. Judge hiện dùng cùng model với ứng dụng; hãy đọc caveat trong `reports/ragas_baseline.md` trước khi trích số.

## Liên kết với mã nguồn

Mỗi tài liệu nêu tên file và dòng mã khi cần đối chiếu. Phần điều phối nằm ở `src/finassist/graph.py`, LLM chain nằm ở `src/finassist/chains.py`, còn các hàng rào citation và số liệu nằm ở `src/finassist/postprocess.py`. Sơ đồ đầy đủ có trong [`ARCHITECTURE.md`](ARCHITECTURE.md).
