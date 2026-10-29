import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from finassist.chains import coerce_extraction, format_context, make_deps, message_text
from finassist.schemas import Chunk, LLMExtraction, PriceInfo


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


def test_coerce_string_revenue_with_commas():
    assert coerce_extraction({"revenue": "416,161"}).revenue.value == 416161


def test_coerce_dict_string_value():
    e = coerce_extraction({"revenue": {"value": "$416,161", "period": "FY2025"}})
    assert e.revenue.value == 416161 and e.revenue.period == "FY2025"


def test_coerce_eps_string():
    assert coerce_extraction({"eps": "$7.46"}).eps == 7.46


def test_coerce_none_and_empty():
    e = coerce_extraction({"revenue": None, "eps": None})
    assert e.revenue is None and e.eps is None


def test_coerce_bool_revenue_degrades_per_field():
    e = coerce_extraction({"revenue": True, "eps": 1.5, "ticker": "AAPL", "risk_factors": ["fx"]})
    assert e.revenue is None and e.eps == 1.5 and e.ticker == "AAPL" and e.risk_factors == ["fx"]


def test_coerce_bad_revenue_string_keeps_rest():
    e = coerce_extraction({"revenue": "n/a", "eps": "7.46", "ticker": "AAPL"})
    assert e.revenue is None and e.eps == 7.46 and e.ticker == "AAPL"


def test_coerce_dict_missing_period_and_no_mutation():
    args = {"revenue": {"value": 5}}
    e = coerce_extraction(args)
    assert e.revenue.value == 5 and e.revenue.period
    assert args == {"revenue": {"value": 5}}


def test_coerce_non_dict_args():
    assert coerce_extraction(None) == LLMExtraction()
    assert coerce_extraction("x") == LLMExtraction()


class _Runnable:
    def __init__(self, result):
        self.result = result

    def invoke(self, _msgs):
        return self.result


class _FakeLLM:
    def __init__(self, extractor_result):
        self.extractor_result = extractor_result

    def with_structured_output(self, schema, include_raw=False):
        return _Runnable(self.extractor_result if include_raw else None)


def _deps(result, tmp_path):
    settings = SimpleNamespace(cache_dir=tmp_path, top_k=3, price_offline=True)
    return make_deps(settings, None, _FakeLLM(result))


def _chunk(cid="AAPL-2025-7-001", ticker="AAPL", fy=2025, text="Total net sales 416,161"):
    return Chunk(chunk_id=cid, text=text, ticker=ticker, fiscal_year=fy, item="7")


def test_structure_fallback_logs_and_returns_empty(tmp_path, caplog):
    raw = SimpleNamespace(tool_calls=[{"args": "garbage"}])
    deps = _deps({"parsed": None, "raw": raw, "parsing_error": ValueError("x")}, tmp_path)
    with caplog.at_level(logging.WARNING, logger="finassist.chains"):
        out = deps.structure("q", "a [AAPL-2025-7-001]", [_chunk()], [], "filing")
    assert out.revenue is None and out.eps is None and out.risk_factors == []
    assert any("x" in r.getMessage() for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_structure_identity_fields_from_chunks_override_llm(tmp_path):
    ext = LLMExtraction(ticker="MSFT", fiscal_year=1999, eps=1.0)
    deps = _deps({"parsed": ext, "raw": None, "parsing_error": None}, tmp_path)
    cs = [_chunk("AAPL-2024-7-001", fy=2024), _chunk("AAPL-2025-7-002", fy=2025)]
    out = deps.structure("q", "a [AAPL-2025-7-002]", cs, [], "filing")
    assert out.ticker == "AAPL" and out.fiscal_year == 2025 and out.eps == 1.0


def test_structure_multi_ticker_keeps_llm_ticker(tmp_path):
    ext = LLMExtraction(ticker="MSFT", eps=1.0)
    deps = _deps({"parsed": ext, "raw": None, "parsing_error": None}, tmp_path)
    cs = [_chunk("AAPL-2025-7-001"), _chunk("MSFT-2025-7-001", ticker="MSFT")]
    assert deps.structure("q", "a", cs, [], "filing").ticker == "MSFT"


def test_structure_warns_on_empty_extraction_with_chunks(tmp_path, caplog):
    deps = _deps({"parsed": LLMExtraction(), "raw": None, "parsing_error": None}, tmp_path)
    with caplog.at_level(logging.WARNING, logger="finassist.chains"):
        deps.structure("q", "a", [_chunk()], [], "filing")
    assert any("empty" in r.getMessage().lower() or "no revenue" in r.getMessage().lower() for r in caplog.records)


def test_structure_context_narrowed_to_cited_chunks(tmp_path):
    seen = {}

    class R:
        def invoke(self, msgs):
            seen["human"] = msgs[1][1]
            return {"parsed": LLMExtraction(eps=1.0), "raw": None, "parsing_error": None}

    llm = SimpleNamespace(with_structured_output=lambda schema, include_raw=False: R())
    deps = make_deps(SimpleNamespace(cache_dir=tmp_path, top_k=3, price_offline=True), None, llm)
    cs = [_chunk("AAPL-2025-7-001", text="AAA"), _chunk("AAPL-2025-7-002", text="BBB")]
    deps.structure("q", "see [AAPL-2025-7-002]", cs, [], "filing")
    assert "BBB" in seen["human"] and "AAA" not in seen["human"]
    deps.structure("q", "no markers", cs, [], "filing")
    assert "AAA" in seen["human"] and "BBB" in seen["human"]


def test_structure_fiscal_year_from_chunks_only_for_single_ticker(tmp_path):
    deps = _deps({"parsed": LLMExtraction(fiscal_year=2030), "raw": None, "parsing_error": None}, tmp_path)
    mixed = [_chunk("AAPL-2025-7-001", fy=2025), _chunk("MSFT-2026-7-001", ticker="MSFT", fy=2026)]
    assert deps.structure("q", "a", mixed, [], "filing").fiscal_year is None
    single = [_chunk("AAPL-2024-7-001", fy=2024), _chunk("AAPL-2025-7-002", fy=2025)]
    assert deps.structure("q", "a", single, [], "filing").fiscal_year == 2025


def test_make_deps_retrieve_delegates_to_the_retriever():
    from finassist.chains import make_deps
    from finassist.config import get_settings
    from finassist.schemas import Chunk

    class FakeLLM:
        def with_structured_output(self, *a, **k):
            return object()

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, tickers):
            self.calls.append((query, tickers))
            return [Chunk(chunk_id="AAPL-2025-8-001", text="t", ticker="AAPL", fiscal_year=2025, item="8")]

    r = FakeRetriever()
    deps = make_deps(get_settings(), store=object(), llm=FakeLLM(), retriever=r)
    got = deps.retrieve("english query", ["AAPL"])
    assert [c.chunk_id for c in got] == ["AAPL-2025-8-001"]
    assert r.calls == [("english query", ["AAPL"])]
