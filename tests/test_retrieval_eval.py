from eval.retrieval_eval import evaluate, first_rank, question_metrics, render_markdown, summarize
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
