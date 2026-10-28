import pytest

from finassist.tools import PriceUnavailable, get_stock_price


def ok(_t):
    return {"price": 200.5, "market_cap": 3.0e12, "pe_ratio": 31.2}


def boom(_t):
    raise ConnectionError("no network")


def test_fetch_writes_cache(tmp_path):
    p = get_stock_price("AAPL", tmp_path / "p.json", fetcher=ok)
    assert p.price == 200.5 and p.stale is False and p.ticker == "AAPL"
    assert (tmp_path / "p.json").exists()


def test_offline_is_deterministic_from_cache(tmp_path):
    cache = tmp_path / "p.json"
    get_stock_price("AAPL", cache, fetcher=ok)
    a = get_stock_price("AAPL", cache, fetcher=boom)
    b = get_stock_price("AAPL", cache, fetcher=boom)
    assert a == b and a.stale is True and a.price == 200.5


def test_forced_offline_never_calls_fetcher(tmp_path):
    cache = tmp_path / "p.json"
    get_stock_price("AAPL", cache, fetcher=ok)
    p = get_stock_price("AAPL", cache, fetcher=boom, offline=True)
    assert p.stale is True


def test_no_cache_and_no_network_raises(tmp_path):
    with pytest.raises(PriceUnavailable):
        get_stock_price("NVDA", tmp_path / "p.json", fetcher=boom)
