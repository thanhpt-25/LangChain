import json

import pytest

from finassist.config import ROOT, TICKERS, get_settings

GOLD = json.loads((ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))


def test_evidence_shape():
    for q in GOLD:
        ev = q.get("evidence")
        if q["category"] == "trap":
            assert not ev, q["id"]
            continue
        assert ev, q["id"]
        for g in ev:
            assert set(g) == {"ticker", "contains"}, q["id"]
            assert g["ticker"] in TICKERS, q["id"]
            assert g["contains"] and all(isinstance(s, str) and s.strip() for s in g["contains"]), q["id"]


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "AAPL.html").exists(), reason="needs data/raw")
def test_every_evidence_group_is_satisfiable_by_some_chunk():
    from finassist.ingest import load_filing

    s = get_settings()
    corpus = {t: [" ".join(c.text.lower().split()) for c in load_filing(t, s.raw_dir)] for t in TICKERS}
    for q in GOLD:
        for g in q.get("evidence", []):
            needles = [" ".join(n.lower().split()) for n in g["contains"]]
            assert any(all(n in text for n in needles) for text in corpus[g["ticker"]]), (q["id"], g)
