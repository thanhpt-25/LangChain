from __future__ import annotations

import re

from finassist.schemas import Chunk, Citation, FinanceAnswer

MARKER_RE = re.compile(r"\[([A-Z]+-\d{4}-[0-9A-Z]+-\d{3})\]")


def extract_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    by_id = {c.chunk_id: c for c in chunks}
    seen: list[str] = []
    for cid in MARKER_RE.findall(answer):
        if cid in by_id and cid not in seen:
            seen.append(cid)
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
