from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
TICKERS = ("AAPL", "MSFT", "NVDA")


def normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    llm_model: str
    judge_model: str
    embedding_model: str
    chroma_dir: Path
    raw_dir: Path
    cache_dir: Path
    sec_user_agent: str
    top_k: int
    price_offline: bool
    retrieval_mode: str = "dense"
    reranker: str = "none"
    candidates: int = 30
    embed_prefix: bool = False
    table_rows: bool = False
    price_timeout: float = 8.0


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=True)
    data = ROOT / "data"
    return Settings(
        base_url=normalize_base_url(os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")),
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        llm_model=os.getenv("FINASSIST_LLM_MODEL", "kr/claude-sonnet-5.5"),
        judge_model=os.getenv("FINASSIST_JUDGE_MODEL", "kr/claude-sonnet-5.5"),
        embedding_model=os.getenv("FINASSIST_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        chroma_dir=Path(os.getenv("FINASSIST_CHROMA_DIR", str(data / "chroma"))),
        raw_dir=data / "raw",
        cache_dir=data / "cache",
        sec_user_agent=os.getenv("SEC_USER_AGENT", "finassist-demo demo@example.com"),
        top_k=int(os.getenv("FINASSIST_TOP_K", "5")),
        price_offline=os.getenv("FINASSIST_PRICE_OFFLINE", "0") == "1",
        retrieval_mode=os.getenv("FINASSIST_RETRIEVAL", "dense"),
        reranker=os.getenv("FINASSIST_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        candidates=int(os.getenv("FINASSIST_CANDIDATES", "30")),
        embed_prefix=os.getenv("FINASSIST_EMBED_PREFIX", "0") == "1",
        table_rows=os.getenv("FINASSIST_TABLE_ROWS", "0") == "1",
        price_timeout=float(os.getenv("FINASSIST_PRICE_TIMEOUT", "8")),
    )
