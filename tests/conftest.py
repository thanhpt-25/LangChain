import os

import pytest

from finassist.graph import Deps
from finassist.schemas import Chunk, FinanceAnswer, IntentResult, PriceInfo


@pytest.fixture(autouse=True)
def _isolate_from_dotenv(monkeypatch):
    # get_settings() calls load_dotenv(override=True), which would read the real
    # project .env and permanently mutate os.environ. Unit tests must be hermetic,
    # so neutralize it unless running integration tests (RUN_INTEGRATION=1).
    if os.environ.get("RUN_INTEGRATION") != "1":
        monkeypatch.setattr("finassist.config.load_dotenv", lambda *a, **k: False)


class Calls(list):
    pass


@pytest.fixture
def make_deps():
    def factory(intent="filing", tickers=("AAPL",), chunks=None, generate=None, price_fn=None):
        calls = Calls()
        default_chunk = Chunk(chunk_id="AAPL-2025-7-000", text="Total net sales 416,161", ticker="AAPL",
                              fiscal_year=2025, item="7")
        chunks = [default_chunk] if chunks is None else chunks

        def classify(q):
            calls.append("classify")
            return IntentResult(intent=intent, tickers=list(tickers))

        def retrieve(q, tks):
            calls.append("retrieve")
            return chunks

        def price(t):
            calls.append("price")
            if price_fn:
                return price_fn(t)
            return PriceInfo(ticker=t, price=200.0, as_of="2026-10-28T00:00:00+00:00")

        def gen(q, cs, ps):
            calls.append("generate")
            return "Net sales were 416,161 [AAPL-2025-7-000]."

        def structure(q, ans, cs, ps, it):
            calls.append("structure")
            return FinanceAnswer(intent=it, answer=ans)

        deps = Deps(classify=classify, retrieve=retrieve, price=price,
                    generate=generate or gen, structure=structure)
        return deps, calls

    return factory
