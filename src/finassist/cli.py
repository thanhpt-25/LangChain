from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _price_line(p: dict) -> str:
    extra = ""
    if p.get("market_cap") is not None:
        extra += f", market cap {p['market_cap']}"
    if p.get("pe_ratio") is not None:
        extra += f", P/E {p['pe_ratio']}"
    if p.get("stale"):
        extra += " (stale)"
    return f"  - {p['ticker']}: {p['price']} as of {p['as_of']}{extra}"


def render_record(rec: dict) -> str:
    out = rec["output"]
    lines = [f"Q: {rec['question']}", f"Intent: {out['intent']}", "", out["answer"], ""]
    prices = out.get("stock_prices") or ([out["stock_price"]] if out.get("stock_price") else [])
    if prices:
        lines.append("Stock price:")
        lines += [_price_line(p) for p in prices]
        lines.append("")
    if out.get("citations"):
        lines.append("Citations:")
        lines += [f"  - {c['source']} · Item {c['item']} · {c['chunk_id']}" for c in out["citations"]]
        lines.append("")
    if rec.get("error"):
        lines.append(f"[error] {rec['error']}")
    lines += ["Structured output:", json.dumps(out, indent=2, ensure_ascii=False)]
    return "\n".join(lines)


def to_record(state: dict) -> dict:
    return {
        "question": state["question"],
        "error": state.get("error"),
        "output": state["output"].model_dump(),
        "retrieved": [
            {"chunk_id": c.chunk_id, "item": c.item, "score": c.score, "text": c.text}
            for c in state.get("retrieved_chunks", [])
        ],
    }


def _cmd_ingest(rebuild: bool = False) -> None:
    from finassist.config import TICKERS, get_settings
    from finassist.ingest import load_filing
    from finassist.vectorstore import FilingStore, get_embeddings

    s = get_settings()
    store = FilingStore(s.chroma_dir, get_embeddings(s), embed_prefix=s.embed_prefix)
    if rebuild:
        store.reset()
    for t in TICKERS:
        chunks = load_filing(t, s.raw_dir, table_rows=s.table_rows)
        store.upsert(chunks)
        print(f"{t}: {len(chunks)} chunks")
    print(f"total in store: {store.count()}")


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def _cmd_ask(question: str, record: str | None, replay: str | None) -> None:
    if replay:
        try:
            rec = json.loads(Path(replay).read_text(encoding="utf-8"))
            text = render_record(rec)
        except (OSError, ValueError, KeyError, TypeError) as e:
            _fail(f"error: cannot replay {replay}: {type(e).__name__}: {e}")
        print(text)
        return
    try:
        from finassist.runtime import ask, build_app

        rec = to_record(ask(build_app(), question, tags=["demo"]))
        if record:
            Path(record).parent.mkdir(parents=True, exist_ok=True)
            Path(record).write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:  # noqa: BLE001 - CLI boundary: one readable line instead of a traceback
        msg = " ".join(str(e).split())
        _fail(f"error: {type(e).__name__}: {msg} (is the LLM router up and are the models available offline? try --replay runs/demo.json)")
    print(render_record(rec))


def main() -> None:
    p = argparse.ArgumentParser(prog="finassist")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("--rebuild", action="store_true")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--record")
    a.add_argument("--replay")
    args = p.parse_args()
    if args.cmd == "ingest":
        _cmd_ingest(args.rebuild)
    else:
        _cmd_ask(args.question, args.record, args.replay)
