from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from finassist.schemas import PriceInfo


class PriceUnavailable(Exception):
    pass


def _yf_fetch(ticker: str) -> dict:
    import yfinance as yf

    info = yf.Ticker(ticker).info
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None:
        raise ValueError(f"no price for {ticker}")
    return {"price": float(price), "market_cap": info.get("marketCap"), "pe_ratio": info.get("trailingPE")}


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def get_stock_price(
    ticker: str,
    cache_path: Path,
    fetcher: Callable[[str], dict] | None = None,
    offline: bool = False,
) -> PriceInfo:
    cache = _read(cache_path)
    if not offline:
        try:
            data = (fetcher or _yf_fetch)(ticker)
            entry = {**data, "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            cache[ticker] = entry
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2))
            return PriceInfo(ticker=ticker, stale=False, **entry)
        except Exception:
            pass
    if ticker in cache:
        return PriceInfo(ticker=ticker, stale=True, **cache[ticker])
    raise PriceUnavailable(f"price for {ticker} unavailable (no network, no cache)")
