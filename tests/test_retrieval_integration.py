import json
import os

import pytest

from finassist.config import ROOT, get_settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration"),
]


def _golden(qid: str) -> dict:
    items = json.loads((ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))
    return next(x for x in items if x["id"] == qid)


def _hits_evidence(qid: str) -> bool:
    from finassist.retrieval import make_retriever
    from finassist.vectorstore import FilingStore, get_embeddings

    s = get_settings()
    retriever = make_retriever(s, FilingStore(s.chroma_dir, get_embeddings(s)))
    q = _golden(qid)
    tickers = sorted({g["ticker"] for g in q["evidence"]})
    chunks = retriever.retrieve(q["question"], tickers)

    def satisfied(group: dict) -> bool:
        needles = [n.lower() for n in group["contains"]]
        return any(
            c.ticker == group["ticker"] and all(n in " ".join(c.text.lower().split()) for n in needles)
            for c in chunks
        )

    return all(satisfied(g) for g in q["evidence"])


@pytest.mark.parametrize("qid", ["q03", "q21"])
def test_default_retriever_finds_evidence_in_top5(qid):
    assert _hits_evidence(qid)


@pytest.mark.xfail(
    strict=True,
    reason="known miss: q05 evidence ranks 12 with the default retriever (reports/retrieval_experiments.md)",
)
def test_default_retriever_q05_known_miss():
    assert _hits_evidence("q05")
