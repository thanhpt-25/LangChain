from __future__ import annotations

import re

from finassist.schemas import Chunk, Citation, FinanceAnswer

ID_RE = re.compile(r"[A-Z]+-\d{4}-[0-9A-Z]+-\d{3}")
MARKER_RE = re.compile(rf"\[({ID_RE.pattern})\]")  # single-id marker, kept for existing callers
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_SEPARATORS_RE = re.compile(r"[,;\s]+|\b(?:và|and)\b")


def marker_ids(answer: str) -> list[str]:
    """Chunk ids cited in square brackets, in order of first appearance, without duplicates.

    A bracket group counts only if, once separators (comma, semicolon, space, "và", "and") are
    removed, it holds nothing but chunk ids. "[A, B]" and "[A][B]" both yield A and B; "[see A]" and
    the price-unavailable note yield nothing.
    """
    found: list[str] = []
    for group in _BRACKET_RE.findall(answer):
        parts = [p for p in _SEPARATORS_RE.split(group) if p]
        if parts and all(ID_RE.fullmatch(p) for p in parts):
            found.extend(p for p in parts if p not in found)
    return found


def extract_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    by_id = {c.chunk_id: c for c in chunks}
    seen = [cid for cid in marker_ids(answer) if cid in by_id]
    return [
        Citation(
            source=f"{by_id[i].ticker} 10-K FY{by_id[i].fiscal_year}",
            item=by_id[i].item, chunk_id=i, quote=by_id[i].text[:200],
        )
        for i in seen
    ]


def _appears(token: str, corpus: str) -> bool:
    """True if token occurs as a standalone number, not embedded in a longer one.

    Rejects a preceding digit or digit+separator ("1,416,161", "1.4.90") and a following
    digit or separator+digit ("416,1615", "416,161,000", "416,161.5"). A sentence-ending
    "416,161." is still accepted.
    """
    pat = r"(?<!\d)(?<!\d[,.])" + re.escape(token) + r"(?!\d)(?![,.]\d)"
    return re.search(pat, corpus) is not None


def ground_numbers(out: FinanceAnswer, chunks: list[Chunk]) -> FinanceAnswer:
    """Drop revenue/eps unless the printed number appears in a retrieved chunk."""
    corpus = " ".join(c.text for c in chunks)
    update: dict = {}
    if out.revenue is not None and not _appears(f"{out.revenue.value:,.0f}", corpus):
        update["revenue"] = None
    if out.eps is not None and not _appears(f"{out.eps:.2f}", corpus):
        update["eps"] = None
    return out.model_copy(update=update)
