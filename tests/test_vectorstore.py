from langchain_core.embeddings import DeterministicFakeEmbedding

from finassist.schemas import Chunk
from finassist.vectorstore import FilingStore


def mk(ticker, i, text):
    return Chunk(chunk_id=f"{ticker}-2025-7-{i:03d}", text=text, ticker=ticker, fiscal_year=2025, item="7", source_url="u")


def test_upsert_is_idempotent_and_filters_by_ticker(tmp_path):
    store = FilingStore(tmp_path, DeterministicFakeEmbedding(size=32))
    chunks = [mk("AAPL", 0, "iPhone sales"), mk("AAPL", 1, "Services"), mk("MSFT", 0, "Azure growth")]
    store.upsert(chunks)
    store.upsert(chunks)
    assert store.count() == 3
    hits = store.search("anything", "MSFT", k=5)
    assert [h.chunk_id for h in hits] == ["MSFT-2025-7-000"]
    assert hits[0].score is not None and hits[0].item == "7"
