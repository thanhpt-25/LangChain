from finassist.graph import INSUFFICIENT_TEXT, build_graph


def test_filing_answer_carries_citation_to_retrieved_chunk(make_deps):
    deps, _ = make_deps(intent="filing")
    out = build_graph(deps).invoke({"question": "q"})["output"]
    assert [c.chunk_id for c in out.citations] == ["AAPL-2025-7-000"]


def test_missing_marker_retries_once_then_flags_error(make_deps):
    n = {"calls": 0}

    def gen(q, cs, ps):
        n["calls"] += 1
        return "No marker here."

    deps, _ = make_deps(intent="filing", generate=gen)
    state = build_graph(deps).invoke({"question": "q"})
    assert n["calls"] == 2
    assert state["error"] == "missing_citation"
    assert state["output"].citations == []
    assert state["output"].answer == INSUFFICIENT_TEXT


def test_hallucinated_chunk_id_is_not_cited(make_deps):
    deps, _ = make_deps(intent="filing", generate=lambda q, c, p: "x [MSFT-2025-7-099] y [AAPL-2025-7-000]")
    out = build_graph(deps).invoke({"question": "q"})["output"]
    assert [c.chunk_id for c in out.citations] == ["AAPL-2025-7-000"]


import pytest


@pytest.mark.parametrize("intent", ["filing", "both"])
def test_only_hallucinated_marker_retries_then_insufficient(make_deps, intent):
    n = {"calls": 0}

    def gen(q, cs, ps):
        n["calls"] += 1
        return "Sales rose [MSFT-2025-7-099]."

    deps, _ = make_deps(intent=intent, generate=gen)
    state = build_graph(deps).invoke({"question": "q"})
    assert n["calls"] == 2
    assert state["output"].answer == INSUFFICIENT_TEXT
    assert "missing_citation" in state["error"]
    assert state["output"].citations == []


def test_multi_id_bracket_passes_without_retry_and_yields_two_citations(make_deps):
    from finassist.schemas import Chunk

    a = Chunk(chunk_id="AAPL-2025-7-000", text="Net sales 416,161", ticker="AAPL", fiscal_year=2025, item="7")
    b = Chunk(chunk_id="AAPL-2025-1A-001", text="Supply risk", ticker="AAPL", fiscal_year=2025, item="1A")
    n = {"calls": 0}

    def gen(q, cs, ps):
        n["calls"] += 1
        return "Sales and risks [AAPL-2025-7-000, AAPL-2025-1A-001]."

    deps, _ = make_deps(intent="filing", chunks=[a, b], generate=gen)
    state = build_graph(deps).invoke({"question": "q"})
    assert n["calls"] == 1
    assert not state.get("error")
    assert [c.chunk_id for c in state["output"].citations] == ["AAPL-2025-7-000", "AAPL-2025-1A-001"]
