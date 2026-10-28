import json

from finassist.cli import render_record, main


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
    monkeypatch.setattr("sys.argv", ["finassist", "ask", "ignored", "--replay", str(f)])
    main()
    assert "nope" in capsys.readouterr().out
