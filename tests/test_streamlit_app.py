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
    assert out["answer"] in rendered
    table_text = json.dumps([t.value.to_dict("records") for t in at.table])
    for c in out["citations"]:
        assert c["chunk_id"] in table_text
    assert len(at.expander) == len(rec["retrieved"])
