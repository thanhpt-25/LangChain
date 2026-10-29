from __future__ import annotations

import copy
import logging

from langchain_anthropic import ChatAnthropic

from finassist.config import TICKERS, Settings
from finassist.graph import Deps
from finassist.postprocess import MARKER_RE
from finassist.schemas import Chunk, FinanceAnswer, Intent, IntentResult, LLMExtraction, Money, PriceInfo
from finassist.tools import get_stock_price
from finassist.vectorstore import FilingStore

log = logging.getLogger(__name__)

CLASSIFY_SYSTEM = f"""You route questions for a finance assistant covering ONLY these companies: AAPL (Apple), MSFT (Microsoft), NVDA (NVIDIA).
Return intent:
- "price": needs current stock price, market cap or P/E only.
- "filing": needs facts from the company's latest 10-K (revenue, EPS, net income, risk factors, MD&A, business).
- "both": needs both a current price/market data and 10-K facts.
- "out_of_scope": other companies, investment advice (should I buy/sell, price predictions), or anything not about these companies' finances.
Return tickers as a list drawn from {list(TICKERS)} (map company names to tickers). Use [] if none apply.
Also return language: "en" if the question is in English, otherwise "other" (e.g. Vietnamese). When language is "other", set search_query to a faithful English rewrite of the question (keep company names and financial terms, so it can be matched against English 10-K text); when language is "en", set search_query to null."""

GENERATE_SYSTEM = """You answer finance questions using ONLY the provided context.
Rules:
1. Every factual claim from a filing must end with the chunk id in square brackets, e.g. [AAPL-2025-1A-012]. Use only ids that appear in the context.
2. Price data comes from the price block; mention it is stale if marked stale. Do not cite chunk ids for price data.
3. If the context lacks the answer, say so. Never guess or use outside knowledge.
4. No investment advice. Answer in the language of the question. Be concise."""

EXTRACT_SYSTEM = """Extract structured fields from the ANSWER using the CONTEXT.
- revenue: an OBJECT {"value": number, "unit": "USD millions", "period": e.g. "FY2025"}, never a bare number. value is total revenue in USD millions EXACTLY as printed in the context (e.g. 416,161 -> 416161). Leave revenue null if not asked or not stated.
- If the ANSWER states total revenue / net sales you MUST fill revenue (USD millions exactly as printed in the context, e.g. 416,161 -> 416161). If it states diluted EPS you MUST fill eps. Do not leave fields null when the answer states them.
- eps: diluted EPS exactly as printed. Leave null only if the answer does not state it.
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


def _num(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def _money(rev, fy) -> Money | None:
    default_period = f"FY{fy}" if isinstance(fy, int) and not isinstance(fy, bool) else "latest fiscal year"
    if isinstance(rev, dict):
        value = _num(rev.get("value"))
        if value is None:
            return None
        kw = {"value": value, "period": str(rev.get("period") or default_period)}
        if isinstance(rev.get("unit"), str) and rev["unit"]:
            kw["unit"] = rev["unit"]
        return Money(**kw)
    value = _num(rev)
    return None if value is None else Money(value=value, period=default_period)


def coerce_extraction(args) -> LLMExtraction:
    """Lenient per-field parse: a bad field degrades to None/default without losing the others."""
    if not isinstance(args, dict):
        return LLMExtraction()
    args = copy.deepcopy(args)
    fy = args.get("fiscal_year")
    fy = int(fy) if isinstance(fy, (int, float, str)) and not isinstance(fy, bool) and _num(fy) is not None else None
    ticker = args.get("ticker")
    rf = args.get("risk_factors")
    return LLMExtraction(
        ticker=ticker if isinstance(ticker, str) and ticker else None,
        fiscal_year=fy,
        revenue=_money(args.get("revenue"), fy),
        eps=_num(args.get("eps")),
        risk_factors=[x for x in rf if isinstance(x, str)] if isinstance(rf, list) else [],
    )


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
            cited = set(MARKER_RE.findall(answer))
            ctx_chunks = [c for c in chunks if c.chunk_id in cited] or chunks
            res = extractor.invoke([("system", EXTRACT_SYSTEM),
                                    ("human", f"QUESTION\n{question}\n\nANSWER\n{answer}\n\nCONTEXT\n{format_context(ctx_chunks, [])}")])
            if res.get("parsed") is not None:
                ext = res["parsed"]
            else:
                log.warning("structured extraction failed, falling back: parsing_error=%r", res.get("parsing_error"))
                calls = getattr(res.get("raw"), "tool_calls", None) or []
                try:
                    ext = coerce_extraction(calls[0]["args"]) if calls else LLMExtraction()
                except (ValueError, TypeError, KeyError, IndexError):
                    ext = LLMExtraction()
            if ext.revenue is None and ext.eps is None and not ext.ticker and not ext.risk_factors:
                log.warning("empty extraction despite %d chunks: %s", len(chunks), ext.model_dump())
            tickers = {c.ticker for c in chunks}
            update: dict = {"fiscal_year": None}
            if len(tickers) == 1:
                update["ticker"] = next(iter(tickers))
                update["fiscal_year"] = max(c.fiscal_year for c in chunks)
            ext = ext.model_copy(update=update)
        return FinanceAnswer(intent=intent, answer=answer, ticker=ext.ticker, fiscal_year=ext.fiscal_year,
                             revenue=ext.revenue, eps=ext.eps, risk_factors=ext.risk_factors)

    return Deps(classify=classify, retrieve=retrieve, price=price, generate=generate, structure=structure)
