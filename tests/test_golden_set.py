import json
from collections import Counter

from finassist.config import ROOT

DATA = json.loads((ROOT / "eval" / "golden_set.json").read_text())
INTENTS = {"price", "filing", "both", "out_of_scope"}


def test_size_and_distribution():
    assert len(DATA) == 30
    assert Counter(d["category"] for d in DATA) == {"numeric": 10, "risk": 10, "combined": 5, "trap": 5}


def test_ids_unique_and_fields_present():
    assert len({d["id"] for d in DATA}) == 30
    for d in DATA:
        assert set(d) >= {"id", "category", "question", "ticker", "expected_intent", "ground_truth"}
        assert d["expected_intent"] in INTENTS


def test_ground_truth_rules():
    for d in DATA:
        if d["category"] == "trap":
            assert d["expected_intent"] == "out_of_scope"
        else:
            assert len(d["ground_truth"]) >= 20, d["id"]
            assert d["ticker"] in {"AAPL", "MSFT", "NVDA"}
