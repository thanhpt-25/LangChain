import json
import logging
import time

import pytest

from finassist.tools import PriceUnavailable, get_stock_price


def ok(_t):
    return {"price": 200.5, "market_cap": 3.0e12, "pe_ratio": 31.2}


def boom(_t):
    raise ConnectionError("no network")


def recording(result=None, exc=None):
    calls = []

    def fetcher(t):
        calls.append(t)
        if exc is not None:
            raise exc
        return result if result is not None else ok(t)

    return fetcher, calls


def test_fetch_writes_cache(tmp_path):
    p = get_stock_price("AAPL", tmp_path / "p.json", fetcher=ok)
    assert p.price == 200.5 and p.stale is False and p.ticker == "AAPL"
    assert (tmp_path / "p.json").exists()


def test_offline_is_deterministic_from_cache(tmp_path):
    cache = tmp_path / "p.json"
    get_stock_price("AAPL", cache, fetcher=ok)
    fetcher, calls = recording(exc=ConnectionError("x"))
    a = get_stock_price("AAPL", cache, fetcher=fetcher, offline=True)
    b = get_stock_price("AAPL", cache, fetcher=fetcher, offline=True)
    assert a == b and a.stale is True and a.price == 200.5
    assert calls == []


def test_forced_offline_never_calls_fetcher(tmp_path):
    cache = tmp_path / "p.json"
    get_stock_price("AAPL", cache, fetcher=ok)
    fetcher, calls = recording(exc=ConnectionError("x"))
    p = get_stock_price("AAPL", cache, fetcher=fetcher, offline=True)
    assert p.stale is True
    assert calls == []


def test_no_cache_and_no_network_raises(tmp_path):
    with pytest.raises(PriceUnavailable):
        get_stock_price("NVDA", tmp_path / "p.json", fetcher=boom)


@pytest.mark.parametrize("content", ["", '{"AAPL": {', "[]", "null"])
def test_corrupt_cache_is_repaired_by_live_fetch(tmp_path, content, caplog):
    cache = tmp_path / "prices.json"
    cache.write_text(content)
    with caplog.at_level(logging.WARNING):
        p = get_stock_price("AAPL", cache, fetcher=ok)
    assert p.stale is False and p.price == 200.5
    assert json.loads(cache.read_text())["AAPL"]["price"] == 200.5
    if content:
        assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.parametrize("content", ["", '{"AAPL": {', "[]", "null"])
def test_corrupt_cache_with_failing_fetcher_raises_unavailable(tmp_path, content):
    cache = tmp_path / "prices.json"
    cache.write_text(content)
    with pytest.raises(PriceUnavailable):
        get_stock_price("AAPL", cache, fetcher=boom)


def test_cache_write_is_atomic_no_tmp_left(tmp_path):
    get_stock_price("AAPL", tmp_path / "p.json", fetcher=ok)
    assert list(tmp_path.glob("*.tmp")) == []
    assert [f.name for f in tmp_path.iterdir()] == ["p.json"]


@pytest.mark.parametrize("bad", [{"market_cap": 1}, {"price": None}])
def test_invalid_fetch_result_does_not_poison_cache(tmp_path, bad):
    cache = tmp_path / "p.json"
    good = get_stock_price("AAPL", cache, fetcher=ok)
    before = cache.read_text()
    p = get_stock_price("AAPL", cache, fetcher=lambda _t: bad)
    assert cache.read_text() == before
    assert p.stale is True and p.price == 200.5 and p.as_of == good.as_of


def test_invalid_fetch_result_without_cache_raises_and_writes_nothing(tmp_path):
    cache = tmp_path / "p.json"
    with pytest.raises(PriceUnavailable):
        get_stock_price("AAPL", cache, fetcher=lambda _t: {"price": None})
    assert not cache.exists()


def test_invalid_cached_entry_yields_price_unavailable(tmp_path):
    cache = tmp_path / "p.json"
    cache.write_text(json.dumps({"AAPL": {"market_cap": 1}}))
    with pytest.raises(PriceUnavailable):
        get_stock_price("AAPL", cache, fetcher=boom)
    with pytest.raises(PriceUnavailable):
        get_stock_price("AAPL", cache, offline=True)


def test_multi_ticker_cache_preserved(tmp_path):
    cache = tmp_path / "p.json"
    get_stock_price("AAPL", cache, fetcher=ok)
    get_stock_price("MSFT", cache, fetcher=lambda _t: {"price": 400.0})
    assert set(json.loads(cache.read_text())) == {"AAPL", "MSFT"}
    assert get_stock_price("AAPL", cache, offline=True).price == 200.5


def test_failed_live_fetch_leaves_entry_untouched(tmp_path, caplog):
    cache = tmp_path / "p.json"
    get_stock_price("AAPL", cache, fetcher=ok)
    before = cache.read_text()
    with caplog.at_level(logging.WARNING):
        p = get_stock_price("AAPL", cache, fetcher=boom)
    assert p.stale is True
    assert cache.read_text() == before
    assert "live price fetch failed for AAPL" in caplog.text


def test_live_fetch_after_stale_read_is_fresh_and_newer(tmp_path):
    cache = tmp_path / "p.json"
    first = get_stock_price("AAPL", cache, fetcher=ok)
    stale = get_stock_price("AAPL", cache, fetcher=boom)
    assert stale.stale is True
    time.sleep(1.1)
    fresh = get_stock_price("AAPL", cache, fetcher=lambda _t: {"price": 210.0})
    assert fresh.stale is False and fresh.price == 210.0
    assert fresh.as_of > first.as_of


def test_persistence_failure_still_returns_fresh_price(tmp_path, caplog):
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not dir")
    with caplog.at_level(logging.WARNING):
        p = get_stock_price("AAPL", blocker / "sub" / "p.json", fetcher=ok)
    assert p.stale is False and p.price == 200.5
    assert "cache" in caplog.text.lower()


def slow(seconds):
    def fetcher(_t):
        time.sleep(seconds)
        return ok(_t)

    return fetcher


def test_timeout_falls_back_to_stale_cache_without_waiting(tmp_path, caplog):
    cache = tmp_path / "p.json"
    get_stock_price("AAPL", cache, fetcher=ok)
    start = time.monotonic()
    with caplog.at_level(logging.WARNING):
        p = get_stock_price("AAPL", cache, fetcher=slow(3), timeout=0.2)
    assert time.monotonic() - start < 1.5
    assert p.stale is True and p.price == 200.5
    assert any("timed out after 0.2s" in r.getMessage() for r in caplog.records)


def test_timeout_without_cache_raises_unavailable_without_waiting(tmp_path):
    start = time.monotonic()
    with pytest.raises(PriceUnavailable, match="timed out"):
        get_stock_price("NVDA", tmp_path / "p.json", fetcher=slow(3), timeout=0.2)
    assert time.monotonic() - start < 1.5


def test_fast_fetcher_is_unaffected_by_timeout(tmp_path):
    p = get_stock_price("AAPL", tmp_path / "p.json", fetcher=ok, timeout=5)
    assert p.stale is False and p.price == 200.5


def test_exception_in_fetch_thread_propagates_as_before(tmp_path):
    with pytest.raises(PriceUnavailable, match="no network"):
        get_stock_price("AAPL", tmp_path / "p.json", fetcher=boom, timeout=5)


def test_timeout_none_keeps_old_behaviour(tmp_path):
    p = get_stock_price("AAPL", tmp_path / "p.json", fetcher=slow(0.3), timeout=None)
    assert p.stale is False


def test_fetch_thread_is_a_daemon(tmp_path):
    import threading

    seen = {}

    def fetcher(t):
        seen["daemon"] = threading.current_thread().daemon
        return ok(t)

    get_stock_price("AAPL", tmp_path / "p.json", fetcher=fetcher, timeout=5)
    assert seen["daemon"] is True
