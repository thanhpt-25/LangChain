import pytest

from finassist.graph import INSUFFICIENT_TEXT, REFUSAL_TEXT, build_graph
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
    assert "unavailable" in state["output"].answer.lower() or "không" in state["output"].answer.lower()
    assert "down" in state["error"]
