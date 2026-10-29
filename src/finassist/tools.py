from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from finassist.schemas import PriceInfo


log = logging.getLogger(__name__)


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
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        log.warning("price cache %s unreadable: %r", path, exc)
        return {}
    if not isinstance(data, dict):
        log.warning("price cache %s has unexpected shape %s; ignoring", path, type(data).__name__)
        return {}
    return data


def _write_atomic(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get_stock_price(
    ticker: str,
    cache_path: Path,
    fetcher: Callable[[str], dict] | None = None,
    offline: bool = False,
) -> PriceInfo:
    cache = _read(cache_path)
    last_err: Exception | None = None
    if not offline:
        try:
            data = (fetcher or _yf_fetch)(ticker)
            as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")
            info = PriceInfo(ticker=ticker, stale=False, **{**data, "as_of": as_of})
        except Exception as exc:
            last_err = exc
            log.warning("live price fetch failed for %s: %r", ticker, exc)
        else:
            entry = info.model_dump(exclude={"ticker", "stale"})
            try:
                _write_atomic(cache_path, {**cache, ticker: entry})
            except Exception as exc:
                log.warning("could not persist price cache %s: %r", cache_path, exc)
            return info
    if ticker in cache:
        try:
            return PriceInfo(ticker=ticker, stale=True, **cache[ticker])
        except Exception as exc:
            log.warning("invalid cached price entry for %s: %r", ticker, exc)
            last_err = last_err or exc
    detail = repr(last_err) if last_err is not None else "offline, no cache"
    raise PriceUnavailable(f"price for {ticker} unavailable: {detail}") from last_err
