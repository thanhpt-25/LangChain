from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finassist.schemas import Chunk

KEEP_ITEMS = {"1", "1A", "7", "7A", "8"}
# \b after the group stops "Item 10" from matching as Item "1".
ITEM_RE = re.compile(
    r"^[ \t]*item[ \t]+(1a|1b|1c|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16|1)\b[ \t]*[\.\:\-–—]?",
    re.IGNORECASE | re.MULTILINE,
)
_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    text = soup.get_text("\n").replace("\xa0", " ")
    lines = [ln.strip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_items(text: str) -> dict[str, str]:
    """Return {item: body}. 10-Ks list every Item twice (table of contents + body);
    keeping the longest section per item selects the body."""
    matches = list(ITEM_RE.finditer(text))
    best: dict[str, str] = {}
    for i, m in enumerate(matches):
        item = m.group(1).upper()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if item in KEEP_ITEMS and len(body) > len(best.get(item, "")):
            best[item] = body
    return best


def chunk_filing(ticker: str, fiscal_year: int, text: str, source_url: str = "") -> list[Chunk]:
    chunks: list[Chunk] = []
    for item, body in sorted(split_items(text).items()):
        for i, piece in enumerate(_splitter.split_text(body)):
            chunks.append(Chunk(
                chunk_id=f"{ticker}-{fiscal_year}-{item}-{i:03d}",
                text=piece, ticker=ticker, fiscal_year=fiscal_year,
                item=item, source_url=source_url,
            ))
    return chunks


def load_filing(ticker: str, raw_dir: Path) -> list[Chunk]:
    meta = json.loads((raw_dir / f"{ticker}.json").read_text())
    text = html_to_text((raw_dir / f"{ticker}.html").read_text(errors="ignore"))
    return chunk_filing(ticker, meta["fiscal_year"], text, meta["source_url"])
