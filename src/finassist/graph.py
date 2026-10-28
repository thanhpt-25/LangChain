from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from finassist.config import TICKERS
from finassist.postprocess import MARKER_RE, extract_citations, ground_numbers
from finassist.schemas import Chunk, FinanceAnswer, Intent, IntentResult, PriceInfo
from finassist.tools import PriceUnavailable

REFUSAL_TEXT = (
    "Tôi chỉ hỗ trợ câu hỏi về báo cáo 10-K và giá cổ phiếu của AAPL, MSFT, NVDA, "
    "và không đưa ra tư vấn đầu tư. / I only answer questions about the 10-K filings and "
    "stock prices of AAPL, MSFT and NVDA, and I do not give investment advice."
)
INSUFFICIENT_TEXT = "Không đủ thông tin trong 10-K để trả lời. / Not enough information in the 10-K filings."
PRICE_UNAVAILABLE_TEXT = "Dữ liệu giá hiện không khả dụng. / Price data is currently unavailable."


class GraphState(TypedDict, total=False):
    question: str
    intent: Intent
    tickers: list[str]
    price_data: list[PriceInfo]
    retrieved_chunks: list[Chunk]
    answer: str
    output: FinanceAnswer
    error: str | None


@dataclass
class Deps:
    classify: Callable[[str], IntentResult]
    retrieve: Callable[[str, list[str]], list[Chunk]]
    price: Callable[[str], PriceInfo]
    generate: Callable[[str, list[Chunk], list[PriceInfo]], str]
    structure: Callable[[str, str, list[Chunk], list[PriceInfo], Intent], FinanceAnswer]


def _join(*parts: str | None) -> str | None:
    return "; ".join(p for p in parts if p) or None


def build_graph(deps: Deps):
    def classify_intent(state: GraphState) -> GraphState:
        res = deps.classify(state["question"])
        tickers = [t for t in dict.fromkeys(res.tickers) if t in TICKERS]
        intent = res.intent if (tickers or res.intent == "out_of_scope") else "out_of_scope"
        return {"intent": intent, "tickers": tickers}

    def route(state: GraphState) -> str:
        return {"price": "price_tool", "both": "price_tool", "filing": "retrieve", "out_of_scope": "refuse"}[state["intent"]]

    def price_tool(state: GraphState) -> GraphState:
        prices, errors = [], []
        for t in state["tickers"]:
            try:
                prices.append(deps.price(t))
            except PriceUnavailable as e:
                errors.append(str(e))
        return {"price_data": prices, "error": _join(*errors)}

    def after_price(state: GraphState) -> str:
        return "retrieve" if state["intent"] == "both" else "generate_answer"

    def retrieve(state: GraphState) -> GraphState:
        return {"retrieved_chunks": deps.retrieve(state["question"], state["tickers"])}

    def generate_answer(state: GraphState) -> GraphState:
        intent = state["intent"]
        chunks = state.get("retrieved_chunks", [])
        prices = state.get("price_data", [])
        needs_filing = intent in ("filing", "both")
        if needs_filing and not chunks:
            return {"answer": INSUFFICIENT_TEXT}
        if intent == "price" and not prices:
            return {"answer": PRICE_UNAVAILABLE_TEXT}
        answer = ""
        for _ in range(2):
            answer = deps.generate(state["question"], chunks, prices)
            if not needs_filing or MARKER_RE.search(answer):
                return {"answer": answer}
        return {"answer": answer, "error": _join(state.get("error"), "missing_citation")}

    def format_output(state: GraphState) -> GraphState:
        intent, answer = state["intent"], state["answer"]
        chunks = state.get("retrieved_chunks", [])
        prices = state.get("price_data", [])
        if answer in (INSUFFICIENT_TEXT, PRICE_UNAVAILABLE_TEXT):
            return {"output": FinanceAnswer(intent=intent, answer=answer)}
        out = deps.structure(state["question"], answer, chunks, prices, intent)
        out = ground_numbers(out, chunks)
        out = out.model_copy(update={
            "citations": extract_citations(answer, chunks) if intent in ("filing", "both") else [],
            "stock_price": prices[0] if len(prices) == 1 else out.stock_price,
        })
        return {"output": out}

    def refuse(state: GraphState) -> GraphState:
        return {"answer": REFUSAL_TEXT,
                "output": FinanceAnswer(intent="out_of_scope", answer=REFUSAL_TEXT)}

    g = StateGraph(GraphState)
    for name, fn in [("classify_intent", classify_intent), ("price_tool", price_tool), ("retrieve", retrieve),
                     ("generate_answer", generate_answer), ("format_output", format_output), ("refuse", refuse)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify_intent")
    g.add_conditional_edges("classify_intent", route, ["price_tool", "retrieve", "refuse"])
    g.add_conditional_edges("price_tool", after_price, ["retrieve", "generate_answer"])
    g.add_edge("retrieve", "generate_answer")
    g.add_edge("generate_answer", "format_output")
    g.add_edge("format_output", END)
    g.add_edge("refuse", END)
    return g.compile()
