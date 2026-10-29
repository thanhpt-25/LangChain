from __future__ import annotations

import re
from typing import Callable, Protocol

from finassist.config import Settings
from finassist.schemas import Chunk

Scorer = Callable[[str, list[str]], list[float]]

# A number with thousands separators or decimals stays one token ("133,749", "4.90"); words are lowercase runs.
_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?|[a-z]+")


def tokenize(text: str) -> list[str]:
    return [t.rstrip(",") for t in _TOKEN_RE.findall(text.lower())]


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank). Ties break on id for determinism."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, 1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


def make_scorer(model_name: str) -> Scorer | None:
    if model_name in ("", "none"):
        return None
    from sentence_transformers import CrossEncoder  # heavy import, only when a reranker is requested

    model = CrossEncoder(model_name)

    def score(query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in model.predict([(query, p) for p in passages])]

    return score


class SearchStore(Protocol):
    def search(self, query: str, ticker: str, k: int = 5) -> list[Chunk]: ...
    def chunks_for(self, ticker: str) -> list[Chunk]: ...


class Retriever:
    def __init__(self, store: SearchStore, *, mode: str = "dense", scorer: Scorer | None = None,
                 candidates: int = 30, top_k: int = 5):
        self.store = store
        self.mode = mode
        self.scorer = scorer
        self.candidates = candidates
        self.top_k = top_k
        self._bm25: dict[str, tuple[object | None, list[Chunk]]] = {}

    def retrieve(self, query: str, tickers: list[str]) -> list[Chunk]:
        return [c for t in tickers for c in self._for_ticker(query, t)]

    def _for_ticker(self, query: str, ticker: str) -> list[Chunk]:
        if self.mode == "dense" and self.scorer is None:
            return self.store.search(query, ticker, self.top_k)
        dense = self.store.search(query, ticker, self.candidates)
        pool = {c.chunk_id: c for c in dense}
        order = [c.chunk_id for c in dense]
        if self.mode == "hybrid":
            bm25 = self._bm25_top(query, ticker, self.candidates)
            for c in bm25:
                pool.setdefault(c.chunk_id, c)
            order = rrf([order, [c.chunk_id for c in bm25]])
        candidates = [pool[cid] for cid in order]
        if self.scorer is not None and candidates:
            scores = self.scorer(query, [c.text for c in candidates])
            ranked = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
            candidates = [candidates[i] for i in ranked]
        return candidates[: self.top_k]

    def _bm25_top(self, query: str, ticker: str, n: int) -> list[Chunk]:
        if ticker not in self._bm25:
            from rank_bm25 import BM25Okapi

            chunks = self.store.chunks_for(ticker)
            docs = [tokenize(c.text) for c in chunks]
            self._bm25[ticker] = (BM25Okapi(docs) if docs else None, chunks)
        index, chunks = self._bm25[ticker]
        if index is None:
            return []
        scores = index.get_scores(tokenize(query))
        order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))
        return [chunks[i] for i in order[:n]]


def make_retriever(settings: Settings, store: SearchStore) -> Retriever:
    if settings.retrieval_mode not in ("dense", "hybrid"):
        raise ValueError(f"FINASSIST_RETRIEVAL must be 'dense' or 'hybrid', got {settings.retrieval_mode!r}")
    return Retriever(store, mode=settings.retrieval_mode, scorer=make_scorer(settings.reranker),
                     candidates=settings.candidates, top_k=settings.top_k)
