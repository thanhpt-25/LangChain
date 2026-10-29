"""Usage: uv run python scripts/find_in_filing.py AAPL "total net sales" [--item 8]"""
import argparse

from finassist.config import get_settings
from finassist.ingest import load_filing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ticker")
    p.add_argument("term")
    p.add_argument("--item")
    a = p.parse_args()
    for c in load_filing(a.ticker, get_settings().raw_dir):
        if a.item and c.item != a.item:
            continue
        if a.term.lower() in c.text.lower():
            print(f"--- {c.chunk_id}\n{c.text}\n")


if __name__ == "__main__":
    main()
