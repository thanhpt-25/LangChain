import pytest

from finassist.graph import Deps, build_graph, resolve_intent, retrieval_query
from finassist.schemas import Chunk, FinanceAnswer, IntentResult, PriceInfo

Q = "What is NVIDIA's market capitalization now, and what was its total revenue last fiscal year?"
RW = "What was NVIDIA's total revenue in its most recent fiscal year?"


def res(intent="both", language="en", search_query=RW, tickers=("NVDA",)):
    return IntentResult(intent=intent, tickers=list(tickers), language=language, search_query=search_query)


@pytest.mark.parametrize("intent,language,sq,expected", [
    ("both", "en", RW, RW),
    ("both", "en", None, Q),
    ("both", "en", "   ", Q),
    ("both", "en", f"  {RW}  ", RW),
    ("filing", "en", RW, Q),
    ("price", "en", RW, Q),
    ("filing", "other", RW, RW),
    ("both", "other", RW, RW),
    ("filing", "other", "", Q),
])
def test_retrieval_query_rules(intent, language, sq, expected):
    assert retrieval_query(Q, intent, res(intent, language, sq)) == expected


def test_resolve_intent_filters_tickers_and_downgrades():
    assert resolve_intent(IntentResult(intent="filing", tickers=[" aapl ", "TSLA", "AAPL"])) == ("filing", ["AAPL"])
    assert resolve_intent(IntentResult(intent="both", tickers=["TSLA"])) == ("out_of_scope", [])
    assert resolve_intent(IntentResult(intent="out_of_scope", tickers=[])) == ("out_of_scope", [])


def run(question, result):
    seen = {}
    chunk = Chunk(chunk_id="NVDA-2026-7-000", text="Revenue 215,938", ticker="NVDA", fiscal_year=2026, item="7")

    def retrieve(query, tickers):
        seen["retrieve"] = (query, tickers)
        return [chunk]

    def generate(q, chunks, prices):
        seen["generate"] = q
        return "Revenue was 215,938 [NVDA-2026-7-000]."

    deps = Deps(
        classify=lambda q: result,
        retrieve=retrieve,
        price=lambda t: PriceInfo(ticker=t, price=200.0, as_of="2026-10-29T00:00:00+00:00"),
        generate=generate,
        structure=lambda q, a, cs, ps, it: FinanceAnswer(intent=it, answer=a),
    )
    return seen, build_graph(deps).invoke({"question": question})


def test_both_question_retrieves_with_the_rewrite_but_answers_the_original_question():
    seen, state = run(Q, res("both"))
    assert seen["retrieve"] == (RW, ["NVDA"])
    assert seen["generate"] == Q
    assert state["search_query"] == RW


def test_english_filing_question_ignores_a_rewrite():
    seen, _ = run(Q, res("filing"))
    assert seen["retrieve"][0] == Q


def test_both_without_a_rewrite_falls_back_to_the_question():
    seen, _ = run(Q, res("both", search_query=None))
    assert seen["retrieve"][0] == Q


def test_both_with_only_uncovered_tickers_is_refused_and_never_retrieves():
    seen, state = run(Q, res("both", tickers=("TSLA",)))
    assert "retrieve" not in seen
    assert state["output"].intent == "out_of_scope"
