from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["price", "filing", "both", "out_of_scope"]


class Chunk(BaseModel):
    chunk_id: str
    text: str
    ticker: str
    fiscal_year: int
    item: str
    source_url: str = ""
    score: float | None = None


class Citation(BaseModel):
    source: str
    item: str
    chunk_id: str
    quote: str


class Money(BaseModel):
    value: float = Field(description="USD millions, exactly as printed in the filing")
    unit: str = "USD millions"
    period: str


class PriceInfo(BaseModel):
    ticker: str
    price: float
    market_cap: float | None = None
    pe_ratio: float | None = None
    as_of: str
    stale: bool = False


class FinanceAnswer(BaseModel):
    ticker: str | None = None
    fiscal_year: int | None = None
    revenue: Money | None = None
    eps: float | None = None
    risk_factors: list[str] = Field(default_factory=list)
    stock_price: PriceInfo | None = None
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    intent: Intent


class IntentResult(BaseModel):
    intent: Intent
    tickers: list[str] = Field(default_factory=list)


class LLMExtraction(BaseModel):
    """Fields the LLM may extract from retrieved chunks. Everything else is deterministic."""
    ticker: str | None = None
    fiscal_year: int | None = None
    revenue: Money | None = None
    eps: float | None = None
    risk_factors: list[str] = Field(default_factory=list)
