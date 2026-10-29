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


def test_chunks_for_returns_stored_chunks_sorted(tmp_path):
    store = FilingStore(tmp_path, DeterministicFakeEmbedding(size=32))
    store.upsert([mk("AAPL", 1, "Services"), mk("AAPL", 0, "iPhone sales"), mk("MSFT", 0, "Azure growth")])
    got = store.chunks_for("AAPL")
    assert [c.chunk_id for c in got] == ["AAPL-2025-7-000", "AAPL-2025-7-001"]
    assert got[0].text == "iPhone sales" and got[0].ticker == "AAPL" and got[0].fiscal_year == 2025 and got[0].item == "7"


def test_reset_empties_the_collection_and_it_stays_usable(tmp_path):
    store = FilingStore(tmp_path, DeterministicFakeEmbedding(size=32))
    chunks = [mk("AAPL", 0, "iPhone sales"), mk("MSFT", 0, "Azure growth")]
    store.upsert(chunks)
    store.reset()
    assert store.count() == 0
    store.upsert(chunks)
    assert store.count() == 2


def test_embed_prefix_changes_the_vector_but_not_the_stored_text(tmp_path):
    store = FilingStore(tmp_path, DeterministicFakeEmbedding(size=32), embed_prefix=True)
    store.upsert([mk("AAPL", 0, "iPhone sales"), mk("AAPL", 1, "Services")])
    # DeterministicFakeEmbedding is a hash of the text: the exact prefixed text matches with distance ~0.
    hits = store.search("AAPL FY2025 Item 7\niPhone sales", "AAPL", k=2)
    assert hits[0].chunk_id == "AAPL-2025-7-000"
    assert hits[0].score is not None and hits[0].score < 1e-6
    assert hits[0].text == "iPhone sales"
