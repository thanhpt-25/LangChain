from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from finassist.config import Settings
from finassist.schemas import Chunk


def get_embeddings(settings: Settings) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


class FilingStore:
    def __init__(self, persist_dir: Path, embeddings: Embeddings, collection: str = "filings"):
        self._db = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._db.add_texts(
            texts=[c.text for c in chunks],
            metadatas=[
                {"ticker": c.ticker, "fiscal_year": c.fiscal_year, "item": c.item, "source_url": c.source_url}
                for c in chunks
            ],
            ids=[c.chunk_id for c in chunks],
        )

    def search(self, query: str, ticker: str, k: int = 5) -> list[Chunk]:
        """Top-k chunks for `ticker`. `Chunk.score` is the Chroma distance (squared L2 on the
        normalised embeddings): lower means closer, it is not a similarity."""
        results = self._db.similarity_search_with_score(query, k=k, filter={"ticker": ticker})
        return [
            Chunk(
                chunk_id=doc.id or "", text=doc.page_content, ticker=doc.metadata["ticker"],
                fiscal_year=int(doc.metadata["fiscal_year"]), item=doc.metadata["item"],
                source_url=doc.metadata.get("source_url", ""), score=float(score),
            )
            for doc, score in results
        ]

    def count(self) -> int:
        return self._db._collection.count()
