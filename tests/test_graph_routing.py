import pytest

from finassist.graph import INSUFFICIENT_TEXT, PRICE_UNAVAILABLE_TEXT, REFUSAL_TEXT, build_graph, price_unavailable_note
from finassist.schemas import Chunk, FinanceAnswer, IntentResult, PriceInfo
from finassist.tools import PriceUnavailable


def run(deps, q="q"):
    return build_graph(deps).invoke({"question": q})


@pytest.mark.parametrize("intent,expected", [
    ("price", ["classify", "price", "generate", "structure"]),
    ("filing", ["classify", "retrieve", "generate", "structure"]),
    ("both", ["classify", "price", "retrieve", "generate", "structure"]),
])
def test_each_intent_takes_its_own_path(make_deps, intent, expected):
    deps, calls = make_deps(intent=intent)
    run(deps)
    assert list(calls) == expected


def test_out_of_scope_refuses_without_retrieval(make_deps):
    deps, calls = make_deps(intent="out_of_scope", tickers=())
    out = run(deps)["output"]
    assert list(calls) == ["classify"]
    assert out.intent == "out_of_scope" and out.answer == REFUSAL_TEXT and out.citations == []


def test_in_scope_intent_without_known_ticker_is_refused(make_deps):
    deps, calls = make_deps(intent="filing", tickers=("TSLA",))
    assert run(deps)["output"].intent == "out_of_scope"
    assert "retrieve" not in calls


def test_no_chunks_returns_insufficient_without_calling_llm(make_deps):
    deps, calls = make_deps(intent="filing", chunks=[])
    out = run(deps)["output"]
    assert out.answer == INSUFFICIENT_TEXT
    assert "generate" not in calls


def test_price_failure_is_reported_not_raised(make_deps):
    def fail(_t):
        raise PriceUnavailable("down")
    deps, _ = make_deps(intent="price", price_fn=fail)
    state = run(deps)
    assert state["output"].answer == PRICE_UNAVAILABLE_TEXT
    assert "down" in state["error"]


def _fail(_t):
    raise PriceUnavailable("down")


def test_both_price_failure_with_chunks_appends_note_and_keeps_citation(make_deps):
    deps, _ = make_deps(intent="both", price_fn=_fail)
    out = run(deps)["output"]
    assert out.answer.endswith(price_unavailable_note(["AAPL"]))
    assert "[AAPL-2025-7-000]" in out.answer
    assert [c.chunk_id for c in out.citations] == ["AAPL-2025-7-000"]


def test_price_two_tickers_one_fails_names_failed_ticker(make_deps):
    def fn(t):
        if t == "MSFT":
            raise PriceUnavailable("down")
        return PriceInfo(ticker=t, price=1.0, as_of="x")
    deps, _ = make_deps(intent="price", tickers=("AAPL", "MSFT"), price_fn=fn)
    out = run(deps)["output"]
    assert out.answer.endswith("[Giá hiện không khả dụng / price unavailable: MSFT]")
    assert "AAPL" not in out.answer.split("price unavailable:")[1]
    assert [p.ticker for p in out.stock_prices] == ["AAPL"]
    assert out.stock_price is not None and out.stock_price.ticker == "AAPL"


def test_both_no_chunks_is_insufficient_but_keeps_tool_price(make_deps):
    deps, _ = make_deps(intent="both", chunks=[])
    out = run(deps)["output"]
    assert out.answer == INSUFFICIENT_TEXT
    assert out.stock_price is not None and out.stock_price.ticker == "AAPL"
    assert [p.ticker for p in out.stock_prices] == ["AAPL"]


def _fabricating(deps):
    fake = PriceInfo(ticker="AAPL", price=999.0, as_of="llm")
    deps.structure = lambda q, a, c, p, i: FinanceAnswer(intent=i, answer=a, stock_price=fake)
    return deps


def test_stock_price_never_comes_from_structure_llm(make_deps):
    # intent="price" with a failing tool short-circuits before `structure` runs, so it cannot exercise the
    # override; the "both" path (chunks present, price missing) does reach `structure`.
    deps, _ = make_deps(intent="both", price_fn=_fail)
    out = run(_fabricating(deps))["output"]
    assert out.stock_price is None and out.stock_prices == []


def test_two_tickers_stock_price_none_stock_prices_has_both(make_deps):
    deps, _ = make_deps(intent="price", tickers=("AAPL", "MSFT"))
    out = run(_fabricating(deps))["output"]
    assert out.stock_price is None
    assert [p.ticker for p in out.stock_prices] == ["AAPL", "MSFT"]


def test_one_ticker_sets_stock_price_from_tool(make_deps):
    deps, _ = make_deps(intent="price")
    out = run(_fabricating(deps))["output"]
    assert out.stock_price.price == 200.0
    assert [p.price for p in out.stock_prices] == [200.0]


def test_lowercase_classifier_ticker_is_accepted(make_deps):
    deps, calls = make_deps(intent="filing", tickers=(" aapl ",))
    state = run(deps)
    assert state["tickers"] == ["AAPL"] and "retrieve" in calls


def test_refusal_state_keys_are_safe_to_index(make_deps):
    deps, _ = make_deps(intent="out_of_scope", tickers=())
    state = run(deps)
    assert state["retrieved_chunks"] == [] and state["price_data"] == [] and state["error"] is None


def test_normal_answer_starting_with_insufficient_text_is_still_structured(make_deps):
    gen = lambda q, cs, ps: INSUFFICIENT_TEXT + " However, net sales were 416,161 [AAPL-2025-7-000]."
    deps, calls = make_deps(intent="filing", generate=gen)
    out = run(deps)["output"]
    assert "structure" in calls
    assert [c.chunk_id for c in out.citations] == ["AAPL-2025-7-000"]


def test_sentinel_paths_skip_structure_and_set_flag(make_deps):
    deps, calls = make_deps(intent="filing", chunks=[])
    state = run(deps)
    assert "structure" not in calls and state["short_circuit"] is True
    deps, calls = make_deps(intent="filing")
    assert run(deps)["short_circuit"] is False


def _lang_deps(make_deps, language, search_query, tickers=("AAPL",), intent="filing"):
    deps, calls = make_deps(intent=intent, tickers=tickers)
    seen = {}
    deps.classify = lambda q: IntentResult(intent=intent, tickers=list(tickers), language=language,
                                           search_query=search_query)

    def retrieve(q, tks):
        seen["retrieve"] = q
        return [Chunk(chunk_id="AAPL-2025-7-000", text="Total net sales 416,161", ticker="AAPL",
                      fiscal_year=2025, item="7")]

    def gen(q, cs, ps):
        seen["generate"] = q
        return "Net sales were 416,161 [AAPL-2025-7-000]."

    deps.retrieve, deps.generate = retrieve, gen
    return deps, seen


def test_non_english_question_retrieves_with_english_search_query_but_generates_with_original(make_deps):
    deps, seen = _lang_deps(make_deps, "other", "Apple total net sales last fiscal year")
    state = run(deps, "Doanh thu thuần của Apple là bao nhiêu?")
    assert seen["retrieve"] == "Apple total net sales last fiscal year"
    assert seen["generate"] == "Doanh thu thuần của Apple là bao nhiêu?"
    assert state["search_query"] == "Apple total net sales last fiscal year"


def test_english_question_ignores_search_query(make_deps):
    deps, seen = _lang_deps(make_deps, "en", "something else")
    run(deps, "What were Apple's net sales?")
    assert seen["retrieve"] == "What were Apple's net sales?"


def test_english_default_language_is_unchanged(make_deps):
    deps, seen = make_deps(intent="filing")
    assert IntentResult(intent="filing").language == "en" and IntentResult(intent="filing").search_query is None
    run(deps, "What were Apple's net sales?")


@pytest.mark.parametrize("sq", [None, "", "   "])
def test_other_language_with_empty_search_query_falls_back_to_original(make_deps, sq):
    deps, seen = _lang_deps(make_deps, "other", sq)
    state = run(deps, "Doanh thu của Apple?")
    assert seen["retrieve"] == "Doanh thu của Apple?"
    assert state["search_query"] == "Doanh thu của Apple?"


def test_single_ticker_fills_output_ticker_on_price_path(make_deps):
    deps, _ = make_deps(intent="price")
    assert run(deps)["output"].ticker == "AAPL"


def test_single_ticker_fills_output_ticker_on_short_circuit_path(make_deps):
    deps, _ = make_deps(intent="filing", chunks=[])
    assert run(deps)["output"].ticker == "AAPL"


def test_multi_ticker_leaves_output_ticker_none_and_keeps_structure_ticker(make_deps):
    deps, _ = make_deps(intent="price", tickers=("AAPL", "MSFT"))
    assert run(deps)["output"].ticker is None
    deps, _ = make_deps(intent="filing")
    deps.structure = lambda q, a, c, p, i: FinanceAnswer(intent=i, answer=a, ticker="MSFT")
    assert run(deps)["output"].ticker == "MSFT"
