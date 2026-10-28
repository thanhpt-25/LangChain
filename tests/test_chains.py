from types import SimpleNamespace

from finassist.chains import coerce_extraction, format_context, message_text
from finassist.schemas import Chunk, PriceInfo


def test_format_context_lists_chunk_ids_and_price():
    c = Chunk(chunk_id="AAPL-2025-1A-001", text="Supply risk", ticker="AAPL", fiscal_year=2025, item="1A")
    p = PriceInfo(ticker="AAPL", price=200.0, as_of="2026-10-28T00:00:00+00:00", stale=True)
    ctx = format_context([c], [p])
    assert "[AAPL-2025-1A-001]" in ctx and "Supply risk" in ctx
    assert "AAPL" in ctx and "200.0" in ctx and "stale" in ctx


def test_message_text_handles_block_lists():
    assert message_text(SimpleNamespace(content="hi")) == "hi"
    blocks = [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert message_text(SimpleNamespace(content=blocks)) == "ab"


def test_coerce_extraction_wraps_bare_revenue():
    e = coerce_extraction({"revenue": 416161, "eps": 7.46, "fiscal_year": 2025})
    assert e.revenue.value == 416161 and e.revenue.period == "FY2025" and e.eps == 7.46


def test_coerce_extraction_passes_money_dict():
    e = coerce_extraction({"revenue": {"value": 5, "period": "FY2024"}})
    assert e.revenue.value == 5 and e.revenue.period == "FY2024"
