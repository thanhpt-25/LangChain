from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Intent = Literal["price", "filing", "both", "out_of_scope"]


class Chunk(BaseModel):
    chunk_id: str
    text: str
    ticker: str
    fiscal_year: int
    item: str
    source_url: str = ""
    score: float | None = None

    @property
    def context_prefix(self) -> str:
        return f"{self.ticker} FY{self.fiscal_year} Item {self.item}"


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
    stock_prices: list[PriceInfo] = Field(default_factory=list)
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    intent: Intent


class IntentResult(BaseModel):
    intent: Intent
    tickers: list[str] = Field(default_factory=list)
    language: Literal["en", "other"] = "en"
    search_query: str | None = Field(
        default=None, description="English rewrite of the question for retrieval; only when language is 'other'")


class LLMExtraction(BaseModel):
    """Fields the LLM may extract from retrieved chunks. Everything else is deterministic."""
    ticker: str | None = None
    fiscal_year: int | None = None
    revenue: Money | None = None
    eps: float | None = None
    risk_factors: list[str] = Field(default_factory=list)

    @field_validator("fiscal_year", mode="before")
    @classmethod
    def _parse_year(cls, v: object) -> int | None:
        """Accept 2026, "2026", "FY2026", "fiscal 2026"; anything without a 4-digit year becomes None."""
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        m = re.search(r"(?<!\d)(\d{4})(?!\d)", str(v)) if v is not None else None
        return int(m.group(1)) if m else None
