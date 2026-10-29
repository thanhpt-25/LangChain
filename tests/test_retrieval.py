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


class CountingStore(FakeStore):
    def __init__(self, n):
        super().__init__([])
        self._n = n

    def count(self):
        return self._n


def _settings():
    return replace(get_settings(), reranker="none")


def test_make_retriever_raises_on_empty_store():
    from finassist.retrieval import EmptyStoreError

    with pytest.raises(EmptyStoreError, match="uv run finassist ingest") as e:
        make_retriever(_settings(), CountingStore(0))
    assert isinstance(e.value, RuntimeError)
    assert "The filings store is empty. Build it first" in str(e.value)


def test_make_retriever_works_with_non_empty_store_and_without_count():
    assert isinstance(make_retriever(_settings(), CountingStore(3)), Retriever)
    assert isinstance(make_retriever(_settings(), FakeStore([])), Retriever)


def test_empty_store_error_is_raised_before_the_scorer_is_built(monkeypatch):
    from finassist.retrieval import EmptyStoreError

    def no_scorer(name):
        raise AssertionError("scorer must not be built for an empty store")

    monkeypatch.setattr("finassist.retrieval.make_scorer", no_scorer)
    with pytest.raises(EmptyStoreError):
        make_retriever(replace(get_settings(), reranker="cross-encoder/x"), CountingStore(0))


def test_make_deps_retrieve_raises_for_empty_real_store(tmp_path):
    from types import SimpleNamespace

    from langchain_core.embeddings import DeterministicFakeEmbedding

    from finassist.chains import make_deps
    from finassist.retrieval import EmptyStoreError
    from finassist.vectorstore import FilingStore

    store = FilingStore(tmp_path / "chroma", DeterministicFakeEmbedding(size=32))
    settings = SimpleNamespace(cache_dir=tmp_path, price_offline=True, price_timeout=8.0, retrieval_mode="dense",
                               reranker="none", candidates=30, top_k=5)

    class NoLLM:  # never invoked: only retrieval is exercised
        def with_structured_output(self, *a, **k):
            return None

    deps = make_deps(settings, store, NoLLM())
    with pytest.raises(EmptyStoreError, match="finassist ingest"):
        deps.retrieve("q", ["AAPL"])


def _empty_store_deps(tmp_path, monkeypatch, llm):
    from types import SimpleNamespace

    from langchain_core.embeddings import DeterministicFakeEmbedding

    from finassist.chains import make_deps
    from finassist.vectorstore import FilingStore

    def fake_price(ticker, cache_path, offline=False, timeout=None, **kw):
        from finassist.schemas import PriceInfo
        return PriceInfo(ticker=ticker, price=123.0, as_of="2026-10-28T00:00:00+00:00")

    monkeypatch.setattr("finassist.chains.get_stock_price", fake_price)
    store = FilingStore(tmp_path / "chroma", DeterministicFakeEmbedding(size=32))
    settings = SimpleNamespace(cache_dir=tmp_path, price_offline=False, price_timeout=8.0, retrieval_mode="dense",
                               reranker="none", candidates=30, top_k=5)
    return make_deps(settings, store, llm)


def test_empty_real_store_still_serves_price_only_questions(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from finassist.graph import build_graph
    from finassist.retrieval import EmptyStoreError
    from finassist.schemas import IntentResult

    class _Classifier:
        def invoke(self, _msgs):
            return IntentResult(intent="price", tickers=["AAPL"])

    class _LLM:
        def with_structured_output(self, schema, include_raw=False):
            return _Classifier()

        def invoke(self, _msgs):
            return SimpleNamespace(content="AAPL trades at 123.0.")

    deps = _empty_store_deps(tmp_path, monkeypatch, _LLM())
    assert deps.price("AAPL").price == 123.0
    with pytest.raises(EmptyStoreError):
        deps.retrieve("q", ["AAPL"])

    out = build_graph(deps).invoke({"question": "AAPL price?"})["output"]
    assert out.intent == "price"
    assert out.stock_price is not None and out.stock_price.price == 123.0
    assert out.citations == []


def test_graph_price_intent_never_calls_retrieve_when_it_would_raise():
    from finassist.graph import Deps, build_graph
    from finassist.retrieval import EmptyStoreError
    from finassist.schemas import FinanceAnswer, IntentResult, PriceInfo

    def retrieve(q, tks):
        raise EmptyStoreError("empty")

    deps = Deps(
        classify=lambda q: IntentResult(intent="price", tickers=["AAPL"]),
        retrieve=retrieve,
        price=lambda t: PriceInfo(ticker=t, price=1.0, as_of="2026-10-28T00:00:00+00:00"),
        generate=lambda q, cs, ps: "ok",
        structure=lambda q, a, cs, ps, it: FinanceAnswer(intent=it, answer=a),
    )
    out = build_graph(deps).invoke({"question": "AAPL price?"})["output"]
    assert out.intent == "price" and out.stock_price.price == 1.0
