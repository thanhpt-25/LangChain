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
    def __init__(self, persist_dir: Path, embeddings: Embeddings, collection: str = "filings",
                 embed_prefix: bool = False):
        self._persist_dir = persist_dir
        self._embeddings = embeddings
        self._collection_name = collection
        self._embed_prefix = embed_prefix
        self._db = self._open()

    def _open(self) -> Chroma:
        return Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
        )

    def reset(self) -> None:
        """Drop every stored chunk. Needed when the chunking changes, because chunk ids change with it."""
        self._db.delete_collection()
        self._db = self._open()

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {"ticker": c.ticker, "fiscal_year": c.fiscal_year, "item": c.item, "source_url": c.source_url}
            for c in chunks
        ]
        if self._embed_prefix:
            # Embed "AAPL FY2025 Item 8\n<text>" but store (and later show the LLM) only <text>.
            vectors = self._embeddings.embed_documents([f"{c.context_prefix}\n{c.text}" for c in chunks])
            self._db._collection.upsert(ids=ids, embeddings=vectors,
                                        documents=[c.text for c in chunks], metadatas=metadatas)
        else:
            self._db.add_texts(texts=[c.text for c in chunks], metadatas=metadatas, ids=ids)

    def chunks_for(self, ticker: str) -> list[Chunk]:
        res = self._db.get(where={"ticker": ticker}, include=["documents", "metadatas"])
        chunks = [
            Chunk(chunk_id=i, text=d, ticker=m["ticker"], fiscal_year=int(m["fiscal_year"]),
                  item=m["item"], source_url=m.get("source_url", ""))
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]
        return sorted(chunks, key=lambda c: c.chunk_id)

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
