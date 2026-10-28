from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from finassist.config import TICKERS, Settings
from finassist.graph import Deps
from finassist.schemas import Chunk, FinanceAnswer, Intent, IntentResult, LLMExtraction, PriceInfo
from finassist.tools import get_stock_price
from finassist.vectorstore import FilingStore

CLASSIFY_SYSTEM = f"""You route questions for a finance assistant covering ONLY these companies: AAPL (Apple), MSFT (Microsoft), NVDA (NVIDIA).
Return intent:
- "price": needs current stock price, market cap or P/E only.
- "filing": needs facts from the company's latest 10-K (revenue, EPS, net income, risk factors, MD&A, business).
- "both": needs both a current price/market data and 10-K facts.
- "out_of_scope": other companies, investment advice (should I buy/sell, price predictions), or anything not about these companies' finances.
Return tickers as a list drawn from {list(TICKERS)} (map company names to tickers). Use [] if none apply."""

GENERATE_SYSTEM = """You answer finance questions using ONLY the provided context.
Rules:
1. Every factual claim from a filing must end with the chunk id in square brackets, e.g. [AAPL-2025-1A-012]. Use only ids that appear in the context.
2. Price data comes from the price block; mention it is stale if marked stale. Do not cite chunk ids for price data.
3. If the context lacks the answer, say so. Never guess or use outside knowledge.
4. No investment advice. Answer in the language of the question. Be concise."""

EXTRACT_SYSTEM = """Extract structured fields from the ANSWER using the CONTEXT.
- revenue: an OBJECT {"value": number, "unit": "USD millions", "period": e.g. "FY2025"}, never a bare number. value is total revenue in USD millions EXACTLY as printed in the context (e.g. 416,161 -> 416161). Leave revenue null if not asked or not stated.
- eps: diluted EPS exactly as printed. Leave null if not asked or not stated.
- risk_factors: short phrases, only if the question is about risks.
- ticker / fiscal_year: only if the answer concerns a single company."""


def build_llm(settings: Settings, model: str | None = None) -> ChatAnthropic:
    return ChatAnthropic(
        model=model or settings.llm_model, base_url=settings.base_url,
        api_key=settings.api_key, temperature=0, max_tokens=2048,
    )


def message_text(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text")


def format_context(chunks: list[Chunk], prices: list[PriceInfo]) -> str:
    parts = [f"[{c.chunk_id}] ({c.ticker} FY{c.fiscal_year} Item {c.item})\n{c.text}" for c in chunks]
    if prices:
        lines = [
            f"{p.ticker}: price={p.price}, market_cap={p.market_cap}, pe={p.pe_ratio}, as_of={p.as_of}"
            + (" (stale)" if p.stale else "")
            for p in prices
        ]
        parts.append("PRICE DATA\n" + "\n".join(lines))
    return "\n\n".join(parts)


def coerce_extraction(args: dict) -> LLMExtraction:
    """Lenient parse: the model sometimes returns revenue as a bare number instead of a Money object."""
    args = dict(args)
    rev = args.get("revenue")
    if isinstance(rev, (int, float)) and not isinstance(rev, bool):
        fy = args.get("fiscal_year")
        args["revenue"] = {"value": rev, "period": f"FY{fy}" if fy else "latest fiscal year"}
    elif isinstance(rev, dict) and "period" not in rev:
        rev["period"] = "latest fiscal year"
    return LLMExtraction.model_validate(args)


def make_deps(settings: Settings, store: FilingStore, llm: ChatAnthropic) -> Deps:
    classifier = llm.with_structured_output(IntentResult)
    extractor = llm.with_structured_output(LLMExtraction, include_raw=True)
    cache_path = settings.cache_dir / "prices.json"

    def classify(question: str) -> IntentResult:
        return classifier.invoke([("system", CLASSIFY_SYSTEM), ("human", question)])

    def retrieve(question: str, tickers: list[str]) -> list[Chunk]:
        return [c for t in tickers for c in store.search(question, t, settings.top_k)]

    def price(ticker: str) -> PriceInfo:
        return get_stock_price(ticker, cache_path, offline=settings.price_offline)

    def generate(question: str, chunks: list[Chunk], prices: list[PriceInfo]) -> str:
        msg = llm.invoke([("system", GENERATE_SYSTEM),
                          ("human", f"CONTEXT\n{format_context(chunks, prices)}\n\nQUESTION\n{question}")])
        return message_text(msg)

    def structure(question: str, answer: str, chunks: list[Chunk], prices: list[PriceInfo],
                  intent: Intent) -> FinanceAnswer:
        # price and citations are filled deterministically by the graph, not here
        ext = LLMExtraction()
        if chunks:
            res = extractor.invoke([("system", EXTRACT_SYSTEM),
                                    ("human", f"QUESTION\n{question}\n\nANSWER\n{answer}\n\nCONTEXT\n{format_context(chunks, [])}")])
            if res.get("parsed") is not None:
                ext = res["parsed"]
            else:
                calls = getattr(res["raw"], "tool_calls", None) or []
                try:
                    ext = coerce_extraction(calls[0]["args"]) if calls else LLMExtraction()
                except ValueError:
                    ext = LLMExtraction()
        return FinanceAnswer(intent=intent, answer=answer, ticker=ext.ticker, fiscal_year=ext.fiscal_year,
                             revenue=ext.revenue, eps=ext.eps, risk_factors=ext.risk_factors)

    return Deps(classify=classify, retrieve=retrieve, price=price, generate=generate, structure=structure)
