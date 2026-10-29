import pytest

from eval.retrieval_eval import (
    _quantile,
    candidate_warning,
    evaluate,
    first_rank,
    question_metrics,
    render_markdown,
    summarize,
)
from finassist.schemas import Chunk


def ch(text):
    return Chunk(chunk_id="X-2025-8-000", text=text, ticker="AAPL", fiscal_year=2025, item="8")


def test_first_rank_needs_every_string_in_the_same_chunk_case_insensitive():
    texts = ["Net  income\nis shown", "133,749 only", "NET INCOME 133,749 total"]
    assert first_rank(texts, ["net income", "133,749"]) == 3
    assert first_rank(texts, ["missing"]) is None


def test_question_metrics_any_all_and_rr():
    m = question_metrics([2, 12], ks=(5, 10, 20))
    assert (m["any@5"], m["all@5"]) == (True, False)
    assert (m["any@10"], m["all@10"]) == (True, False)
    assert (m["any@20"], m["all@20"]) == (True, True)
    assert m["rr"] == 0.5
    miss = question_metrics([None], ks=(5,))
    assert miss["any@5"] is False and miss["rr"] == 0.0


def test_summarize_overall_and_by_category():
    rows = [
        {"id": "a", "category": "numeric", "metrics": question_metrics([1], ks=(5,))},
        {"id": "b", "category": "numeric", "metrics": question_metrics([None], ks=(5,))},
        {"id": "c", "category": "risk", "metrics": question_metrics([3], ks=(5,))},
    ]
    s = summarize(rows, ks=(5,))
    assert s["overall"]["n"] == 3
    assert round(s["overall"]["any@5"], 3) == 0.667
    assert s["by_category"]["numeric"]["any@5"] == 0.5
    assert s["by_category"]["risk"]["any@5"] == 1.0


def test_evaluate_skips_traps_caches_queries_and_times_each_retrieval():
    golden = [
        {"id": "q1", "category": "numeric", "question": "net income?",
         "evidence": [{"ticker": "MSFT", "contains": ["133,749"]}, {"ticker": "MSFT", "contains": ["net income"]}]},
        {"id": "q2", "category": "trap", "question": "should I buy?"},
    ]
    calls = []

    def retrieve(query, ticker):
        calls.append((query, ticker))
        return [ch("irrelevant"), ch("Net income 133,749")]

    rows, times = evaluate(golden, retrieve)
    assert calls == [("net income?", "MSFT")]          # one retrieval for the two groups, trap skipped
    assert len(times) == 1
    assert rows[0]["id"] == "q1" and rows[0]["ranks"] == [2, 2]


def test_render_markdown_contains_label_numbers_and_latency():
    rows = [{"id": "a", "category": "numeric", "metrics": question_metrics([1], ks=(5, 10, 20))}]
    md = render_markdown("stage 0", {"mode": "dense", "reranker": "none"}, summarize(rows), [10.0, 20.0, 30.0])
    for needle in ("stage 0", "dense", "any@5", "100%", "p50"):
        assert needle in md


@pytest.mark.parametrize("ranks, expected", [
    ([5], {"any@5": True, "any@10": True, "any@20": True}),
    ([6], {"any@5": False, "any@10": True, "any@20": True}),
    ([10], {"any@5": False, "any@10": True, "any@20": True}),
    ([11], {"any@5": False, "any@10": False, "any@20": True}),
    ([20], {"any@5": False, "any@10": False, "any@20": True}),
    ([21], {"any@5": False, "any@10": False, "any@20": False}),
])
def test_question_metrics_k_boundaries_are_inclusive(ranks, expected):
    m = question_metrics(ranks)
    assert {k: m[k] for k in expected} == expected


def test_question_metrics_two_groups_across_boundary():
    m = question_metrics([5, 21])
    assert m["any@20"] is True and m["all@20"] is False
    assert m["any@5"] is True and m["all@5"] is False


def test_summarize_all_at_k_and_mrr_overall_and_by_category():
    rows = [
        {"id": "a", "category": "numeric", "metrics": question_metrics([1, 7], ks=(5,))},
        {"id": "b", "category": "numeric", "metrics": question_metrics([4, 5], ks=(5,))},
        {"id": "c", "category": "risk", "metrics": question_metrics([None], ks=(5,))},
    ]
    s = summarize(rows, ks=(5,))
    o = s["overall"]
    assert (o["any@5"], o["all@5"]) == (pytest.approx(2 / 3), pytest.approx(1 / 3))
    assert o["mrr"] == pytest.approx((1.0 + 0.25 + 0.0) / 3)
    n = s["by_category"]["numeric"]
    assert (n["n"], n["any@5"], n["all@5"], n["mrr"]) == (2, 1.0, 0.5, pytest.approx(0.625))
    r = s["by_category"]["risk"]
    assert (r["any@5"], r["all@5"], r["mrr"]) == (0.0, 0.0, 0.0)


def test_quantile_known_values():
    assert _quantile([7.0], 0.5) == 7.0 and _quantile([7.0], 0.95) == 7.0
    assert _quantile([20.0, 10.0], 0.5) == 10.0 and _quantile([20.0, 10.0], 0.95) == 20.0
    xs = [50.0, 10.0, 40.0, 20.0, 30.0]
    assert (_quantile(xs, 0.0), _quantile(xs, 0.5), _quantile(xs, 0.95), _quantile(xs, 1.0)) == (10.0, 30.0, 50.0, 50.0)


def test_render_markdown_exact_percent_row_and_latency_numbers():
    rows = [
        {"id": "a", "category": "numeric", "metrics": question_metrics([1])},
        {"id": "b", "category": "numeric", "metrics": question_metrics([None])},
        {"id": "c", "category": "numeric", "metrics": question_metrics([None])},
    ]
    md = render_markdown("x", {"mode": "dense"}, summarize(rows), [50.0, 10.0, 40.0, 20.0, 30.0])
    assert "| overall | 3 | 33% | 33% | 33% | 33% | 33% | 33% | 0.333 |" in md
    assert "p50 30, p95 50" in md


def test_empty_input_raises_clear_error():
    with pytest.raises(ValueError, match="no scored questions"):
        summarize([])
    with pytest.raises(ValueError, match="no scored questions"):
        _quantile([], 0.5)


def test_candidate_warning():
    assert candidate_warning(30, "dense", "none") is None
    assert candidate_warning(20, "hybrid", "none") is None
    assert candidate_warning(30, "hybrid", "bge") is None
    assert candidate_warning(10, "dense", "none") is None  # dense returns top_k directly
    for mode, rr in (("hybrid", "none"), ("dense", "bge")):
        w = candidate_warning(10, mode, rr)
        assert w is not None and "10" in w and "20" in w


def test_query_for_applies_the_graph_rules():
    from eval.retrieval_eval import query_for

    table = {
        "q21": {"intent": "both", "tickers": ["AAPL"], "language": "en", "search_query": "What were Apple's net sales?"},
        "q01": {"intent": "filing", "tickers": ["AAPL"], "language": "en", "search_query": "ignored"},
        "q90": {"intent": "both", "tickers": ["TSLA"], "language": "en", "search_query": "rewrite"},
    }
    assert query_for({"id": "q21", "question": "price and revenue?"}, table) == "What were Apple's net sales?"
    assert query_for({"id": "q01", "question": "net sales?"}, table) == "net sales?"
    assert query_for({"id": "q90", "question": "tesla?"}, table) == "tesla?"      # uncovered ticker: out_of_scope
    assert query_for({"id": "q21", "question": "price and revenue?"}, None) == "price and revenue?"
    assert query_for({"id": "q77", "question": "x"}, table) == "x"                # not in the table


def test_rewrite_table_classifies_each_scored_question_once():
    from eval.retrieval_eval import rewrite_table
    from finassist.schemas import IntentResult

    golden = [{"id": "q1", "category": "numeric", "question": "a"}, {"id": "q2", "category": "trap", "question": "b"}]
    calls = []

    def classify(q):
        calls.append(q)
        return IntentResult(intent="filing", tickers=["AAPL"])

    table = rewrite_table(golden, classify)
    assert calls == ["a"]
    assert table == {"q1": IntentResult(intent="filing", tickers=["AAPL"]).model_dump()}


def test_evaluate_uses_query_of_for_retrieval_and_cache_key():
    golden = [{"id": "q1", "category": "combined", "question": "price and revenue?",
               "evidence": [{"ticker": "AAPL", "contains": ["416,161"]}]}]
    seen = []

    def retrieve(query, ticker):
        seen.append(query)
        return [ch("Total net sales 416,161")]

    rows, _ = evaluate(golden, retrieve, query_of=lambda q: "net sales?")
    assert seen == ["net sales?"]
    assert rows[0]["ranks"] == [1]


def test_main_rejects_conflicting_rewrite_options():
    import pytest as _pytest

    from eval.retrieval_eval import main

    with _pytest.raises(SystemExit):
        main(["--label", "x", "--classify", "--rewrites", "r.json"])
    with _pytest.raises(SystemExit):
        main(["--label", "x", "--save-rewrites", "r.json"])
