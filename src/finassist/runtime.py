from __future__ import annotations

from finassist import __version__
from finassist.chains import build_llm, make_deps
from finassist.config import Settings, get_settings
from finassist.graph import build_graph
from finassist.vectorstore import FilingStore, get_embeddings


def build_app(settings: Settings | None = None):
    """Build the full graph app. Constructs heavy objects (embedding model, Chroma, LLM client) on
    every call; long-lived callers must build once and reuse the returned app."""
    s = settings or get_settings()
    store = FilingStore(s.chroma_dir, get_embeddings(s))
    return build_graph(make_deps(s, store, build_llm(s)))


def ask(app, question: str, tags: list[str] | None = None) -> dict:
    return app.invoke(
        {"question": question},
        config={
            "run_name": "finassist",
            "tags": tags or ["demo"],
            "metadata": {"app_version": __version__},
        },
    )
