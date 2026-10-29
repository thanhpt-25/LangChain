from dataclasses import replace

import pytest

from finassist.config import get_settings
from finassist.retrieval import Retriever, make_retriever, make_scorer, rrf, tokenize
from finassist.schemas import Chunk


def ck(i, text, ticker="AAPL"):
    return Chunk(chunk_id=f"{ticker}-2025-8-{i:03d}", text=text, ticker=ticker, fiscal_year=2025,
                 item="8", score=float(i))


class FakeStore:
    """Dense order is insertion order. chunks_for returns every chunk of the ticker."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.search_calls = []

    def search(self, query, ticker, k=5):
        self.search_calls.append((ticker, k))
        return [c for c in self.chunks if c.ticker == ticker][:k]

    def chunks_for(self, ticker):
        return [c for c in self.chunks if c.ticker == ticker]


def test_tokenize_keeps_numbers_with_separators_whole():
    assert tokenize("Net income: $133,749 and EPS 4.90.") == ["net", "income", "133,749", "and", "eps", "4.90"]


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([["a", "b", "c"], ["c", "a"]]) == ["a", "c", "b"]


def test_make_scorer_none_values():
    assert make_scorer("none") is None
    assert make_scorer("") is None


def test_dense_mode_without_scorer_matches_store_search():
    store = FakeStore([ck(0, "a"), ck(1, "b"), ck(2, "c"), ck(0, "m", "MSFT"), ck(1, "n", "MSFT")])
    r = Retriever(store, mode="dense", top_k=2)
    got = r.retrieve("q", ["AAPL", "MSFT"])
    assert [c.chunk_id for c in got] == ["AAPL-2025-8-000", "AAPL-2025-8-001", "MSFT-2025-8-000", "MSFT-2025-8-001"]
    assert store.search_calls == [("AAPL", 2), ("MSFT", 2)]


def test_reranker_reorders_a_wider_candidate_pool():
    store = FakeStore([ck(0, "iPhone"), ck(1, "Services"), ck(2, "Mac"), ck(3, "Net income 133,749"), ck(4, "Other")])
    scorer = lambda q, passages: [10.0 if "income" in p else 0.0 for p in passages]
    r = Retriever(store, mode="dense", scorer=scorer, candidates=5, top_k=2)
    got = r.retrieve("net income", ["AAPL"])
    assert got[0].chunk_id == "AAPL-2025-8-003"
    assert len(got) == 2
    assert store.search_calls == [("AAPL", 5)]


def test_hybrid_surfaces_a_chunk_the_dense_top_n_missed():
    chunks = [ck(0, "iPhone sales"), ck(1, "Services revenue"), ck(2, "Mac and iPad"), ck(9, "Net income 133,749 for the year")]
    store = FakeStore(chunks)
    r = Retriever(store, mode="hybrid", candidates=3, top_k=2)
    got = r.retrieve("net income 133,749", ["AAPL"])
    ids = [c.chunk_id for c in got]
    assert "AAPL-2025-8-009" in ids           # dense top-3 never contained it; BM25 rank 1 did
    assert store.search_calls == [("AAPL", 3)]


def test_hybrid_result_keeps_dense_scores_and_none_for_bm25_only_chunks():
    bm25_only = ck(9, "Net income 133,749")
    bm25_only.score = None   # chunks_for (the BM25 source) has no dense score
    store = FakeStore([ck(0, "iPhone sales"), ck(1, "Services revenue"), bm25_only])
    r = Retriever(store, mode="hybrid", candidates=2, top_k=3)
    by_id = {c.chunk_id: c for c in r.retrieve("net income 133,749", ["AAPL"])}
    assert by_id["AAPL-2025-8-000"].score == 0.0
    assert by_id["AAPL-2025-8-001"].score == 1.0
    assert by_id["AAPL-2025-8-009"].score is None


def test_retriever_rejects_unknown_mode_directly():
    with pytest.raises(ValueError):
        Retriever(FakeStore([]), mode="sparse")


def test_make_retriever_falls_back_to_dense_when_reranker_cannot_load(monkeypatch, caplog):
    import logging

    def boom(name):
        raise OSError("no network and no cache")

    monkeypatch.setattr("finassist.retrieval.make_scorer", boom)
    store = FakeStore([ck(i, f"text {i}") for i in range(8)])
    with caplog.at_level(logging.WARNING, logger="finassist.retrieval"):
        r = make_retriever(replace(get_settings(), retrieval_mode="dense", reranker="some/model", top_k=3), store)
    assert isinstance(r, Retriever) and r.scorer is None
    assert any("falling back to dense" in m for m in caplog.messages)
    assert r.retrieve("q", ["AAPL"]) == store.search("q", "AAPL", 3)


def test_make_retriever_rejects_unknown_mode():
    with pytest.raises(ValueError):
        make_retriever(replace(get_settings(), retrieval_mode="sparse"), FakeStore([]))
