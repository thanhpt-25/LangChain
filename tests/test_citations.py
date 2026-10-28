from finassist.graph import build_graph


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


def test_hallucinated_chunk_id_is_not_cited(make_deps):
    deps, _ = make_deps(intent="filing", generate=lambda q, c, p: "x [MSFT-2025-7-099] y [AAPL-2025-7-000]")
    out = build_graph(deps).invoke({"question": "q"})["output"]
    assert [c.chunk_id for c in out.citations] == ["AAPL-2025-7-000"]
