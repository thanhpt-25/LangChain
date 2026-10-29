import json
import sys

import pytest

from finassist.cli import to_record, main, render_record
from finassist.schemas import Chunk, FinanceAnswer


def test_render_record_prints_answer_and_citations():
    rec = {"question": "q", "error": None, "retrieved": [],
           "output": {"intent": "filing", "answer": "Net sales 416,161 [AAPL-2025-8-000]",
                      "citations": [{"source": "AAPL 10-K FY2025", "item": "8", "chunk_id": "AAPL-2025-8-000", "quote": "x"}],
                      "risk_factors": []}}
    text = render_record(rec)
    assert "Net sales 416,161" in text and "AAPL-2025-8-000" in text and "AAPL 10-K FY2025" in text


def test_render_record_prints_stock_price_lines():
    rec = {"question": "q", "error": None, "retrieved": [],
           "output": {"intent": "combined", "answer": "a", "citations": [], "risk_factors": [],
                      "stock_price": {"ticker": "NVDA", "price": 123.45, "as_of": "2026-10-28"},
                      "stock_prices": [{"ticker": "NVDA", "price": 123.45, "as_of": "2026-10-28"}]}}
    text = render_record(rec)
    assert "Stock price" in text and "NVDA" in text and "123.45" in text


def test_replay_runs_without_building_the_app(tmp_path, capsys, monkeypatch):
    f = tmp_path / "demo.json"
    f.write_text(json.dumps({"question": "q", "error": None, "retrieved": [],
                             "output": {"intent": "out_of_scope", "answer": "nope", "citations": [], "risk_factors": []}}))
    monkeypatch.setitem(sys.modules, "finassist.runtime", None)  # any import of it raises ImportError
    monkeypatch.setattr("sys.argv", ["finassist", "ask", "ignored", "--replay", str(f)])
    main()
    assert "nope" in capsys.readouterr().out


def test_replay_missing_file_is_a_one_line_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["finassist", "ask", "q", "--replay", str(tmp_path / "nope.json")])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "nope.json" in err and len(err.strip().splitlines()) == 1


def test_replay_corrupt_file_is_a_one_line_error(tmp_path, capsys, monkeypatch):
    f = tmp_path / "bad.json"
    f.write_text("{not json")
    monkeypatch.setattr("sys.argv", ["finassist", "ask", "q", "--replay", str(f)])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and len(err.strip().splitlines()) == 1


def test_live_failure_prints_one_line_with_replay_hint(capsys, monkeypatch):
    import types

    def boom():
        raise ConnectionError("router down")

    fake = types.SimpleNamespace(build_app=boom, ask=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "finassist.runtime", fake)
    monkeypatch.setattr("sys.argv", ["finassist", "ask", "q"])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "error: ConnectionError: router down" in err and "--replay" in err
    assert len(err.strip().splitlines()) == 1


def test_record_tolerates_state_without_optional_keys():
    state = {"question": "q", "output": FinanceAnswer(intent="out_of_scope", answer="a")}
    rec = to_record(state)
    assert rec["error"] is None and rec["retrieved"] == [] and rec["output"]["answer"] == "a"


def test_record_includes_retrieved_chunks():
    c = Chunk(chunk_id="AAPL-2025-8-000", text="t", ticker="AAPL", fiscal_year=2025, item="8", score=0.3)
    rec = to_record({"question": "q", "error": "e", "output": FinanceAnswer(intent="filing", answer="a"),
                   "retrieved_chunks": [c]})
    assert rec["error"] == "e" and rec["retrieved"][0]["chunk_id"] == "AAPL-2025-8-000"


def test_ingest_accepts_rebuild_flag(monkeypatch):
    import finassist.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "_cmd_ingest", lambda rebuild=False: seen.update(rebuild=rebuild))
    monkeypatch.setattr("sys.argv", ["finassist", "ingest", "--rebuild"])
    cli.main()
    assert seen == {"rebuild": True}
    monkeypatch.setattr("sys.argv", ["finassist", "ingest"])
    cli.main()
    assert seen == {"rebuild": False}
