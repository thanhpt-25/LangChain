import json

from streamlit.testing.v1 import AppTest

from finassist.config import ROOT

APP = str(ROOT / "app" / "streamlit_app.py")


def _boom(*a, **k):
    raise AssertionError("real app must not be built in replay mode")


def test_replay_renders_answer_and_citations(monkeypatch):
    monkeypatch.setattr("finassist.runtime.build_app", _boom)
    rec = json.loads((ROOT / "runs" / "demo.json").read_text())
    out = rec["output"]

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.sidebar.checkbox[0].check()
    at.button[0].click()
    at.run()

    assert not at.exception
    assert not at.warning
    rendered = " ".join(m.value for m in at.markdown)
    assert out["answer"].replace("$", r"\$") in rendered
    table_text = json.dumps([t.value.to_dict("records") for t in at.table])
    for c in out["citations"]:
        assert c["chunk_id"] in table_text
    assert len(at.expander) == len(rec["retrieved"])


def _replay(monkeypatch, tmp_path, rec=None):
    monkeypatch.setattr("finassist.runtime.build_app", _boom)
    monkeypatch.setattr("finassist.config.ROOT", tmp_path)
    if rec is not None:
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "demo.json").write_text(json.dumps(rec))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.checkbox[0].check()
    at.button[0].click()
    return at.run()


def _rec(**kw):
    out = {"intent": "stock_price", "answer": "ok", "citations": [], "stock_prices": []}
    out.update(kw.pop("out", {}))
    rec = {"question": "recorded Q?", "output": out, "retrieved": [], "error": None}
    rec.update(kw)
    return rec


def test_price_none_score_error_branches(monkeypatch, tmp_path):
    rec = _rec(
        error="boom",
        out={"stock_prices": [{"ticker": "NVDA", "price": 120.5, "market_cap": None, "pe_ratio": None,
                               "as_of": "2024-01-01", "stale": True}]},
        retrieved=[{"chunk_id": "chunk-1", "score": None, "text": "some text"}],
    )
    at = _replay(monkeypatch, tmp_path, rec)
    assert not at.exception
    assert any("boom" in w.value for w in at.warning)
    assert "120.50" in [m.value for m in at.metric]
    assert "n/a" in [m.value for m in at.metric]
    assert any("stale" in m.value for m in at.markdown) or any("stale" in c.value for c in at.caption)
    assert [e.label for e in at.expander] == ["chunk-1"]
    assert any(c.value == "No citations." for c in at.caption)


def test_missing_demo_json_shows_error(monkeypatch, tmp_path):
    at = _replay(monkeypatch, tmp_path)
    assert not at.exception
    assert at.error
    assert "finassist ask" in at.error[0].value


def test_out_of_scope_record(monkeypatch, tmp_path):
    at = _replay(monkeypatch, tmp_path, _rec(out={"intent": "out_of_scope", "answer": "Out of scope."}))
    assert not at.exception


def test_replay_info_shows_recorded_question(monkeypatch, tmp_path):
    at = _replay(monkeypatch, tmp_path, _rec())
    assert at.info
    assert "recorded Q?" in at.info[0].value
    assert "recorded" in at.info[0].value.lower()


def test_dollar_signs_escaped(monkeypatch, tmp_path):
    ans = "Revenue $416,161 million vs $391,035 million."
    at = _replay(monkeypatch, tmp_path, _rec(out={"answer": ans}))
    assert not at.exception
    rendered = [m.value for m in at.markdown]
    assert ans.replace("$", r"\$") in rendered


def test_live_failure_shows_error(monkeypatch, tmp_path):
    def fail(*a, **k):
        raise RuntimeError("no key")

    monkeypatch.setattr("finassist.runtime.build_app", lambda: object())
    monkeypatch.setattr("finassist.runtime.ask", fail)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[0].click()
    at.run()
    assert not at.exception
    assert any("Live run failed: RuntimeError: no key" in e.value for e in at.error)
