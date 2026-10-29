"""Retrieval-only benchmark: for each golden question, does a retrieved chunk contain the evidence?

Usage: uv run python -m eval.retrieval_eval --label "stage 0: dense" [--mode hybrid] [--reranker NAME]
       [--candidates 30] [--show-misses] [--append reports/retrieval_experiments.md]
The store is opened as it is on disk (set FINASSIST_CHROMA_DIR to benchmark another store).
No LLM is called. Runs in seconds once the embedding (and reranker) models are loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from finassist.config import ROOT, TICKERS, get_settings
from finassist.retrieval import make_retriever
from finassist.vectorstore import FilingStore, get_embeddings

KS = (5, 10, 20)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def first_rank(texts: list[str], needles: list[str]) -> int | None:
    want = [_norm(n) for n in needles]
    for i, text in enumerate(texts, 1):
        t = _norm(text)
        if all(w in t for w in want):
            return i
    return None


def question_metrics(ranks: list[int | None], ks: tuple[int, ...] = KS) -> dict:
    out: dict = {}
    for k in ks:
        hits = [r is not None and r <= k for r in ranks]
        out[f"any@{k}"] = any(hits)
        out[f"all@{k}"] = all(hits)
    found = [r for r in ranks if r is not None]
    out["rr"] = 1.0 / min(found) if found else 0.0
    return out


def summarize(rows: list[dict], ks: tuple[int, ...] = KS) -> dict:
    if not rows:
        raise ValueError("no scored questions")

    def agg(sub: list[dict]) -> dict:
        n = len(sub)
        d: dict = {"n": n}
        for k in ks:
            d[f"any@{k}"] = sum(r["metrics"][f"any@{k}"] for r in sub) / n
            d[f"all@{k}"] = sum(r["metrics"][f"all@{k}"] for r in sub) / n
        d["mrr"] = sum(r["metrics"]["rr"] for r in sub) / n
        return d

    cats = sorted({r["category"] for r in rows})
    return {"overall": agg(rows), "by_category": {c: agg([r for r in rows if r["category"] == c]) for c in cats}}


def evaluate(golden: list[dict], retrieve) -> tuple[list[dict], list[float]]:
    """retrieve(query, ticker) -> ranked list[Chunk]. Returns per-question rows and latencies (ms)."""
    cache: dict[tuple[str, str], list[str]] = {}
    rows: list[dict] = []
    times: list[float] = []
    for q in golden:
        if q["category"] == "trap":
            continue
        ranks = []
        for g in q["evidence"]:
            key = (q["question"], g["ticker"])
            if key not in cache:
                t0 = time.perf_counter()
                chunks = retrieve(q["question"], g["ticker"])
                times.append((time.perf_counter() - t0) * 1000)
                cache[key] = [c.text for c in chunks]
            ranks.append(first_rank(cache[key], g["contains"]))
        rows.append({"id": q["id"], "category": q["category"], "ranks": ranks, "metrics": question_metrics(ranks)})
    return rows, times


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        raise ValueError("no scored questions")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def render_markdown(label: str, config: dict, summary: dict, times_ms: list[float]) -> str:
    cols = [f"any@{k}" for k in KS] + [f"all@{k}" for k in KS] + ["mrr"]
    cfg = ", ".join(f"{k}={v}" for k, v in config.items())
    lines = [f"## {label}", "", f"Cấu hình: {cfg}", "",
             "| nhóm | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name, d in [("overall", summary["overall"]), *summary["by_category"].items()]:
        vals = [f"{d[c]:.3f}" if c == "mrr" else _pct(d[c]) for c in cols]
        lines.append(f"| {name} | {d['n']} | " + " | ".join(vals) + " |")
    lines += ["", f"Độ trễ mỗi lần truy hồi (ms): p50 {_quantile(times_ms, 0.5):.0f}, p95 {_quantile(times_ms, 0.95):.0f}", ""]
    return "\n".join(lines)


def candidate_warning(candidates: int, mode: str, reranker: str) -> str | None:
    if candidates >= max(KS) or (mode != "hybrid" and reranker == "none"):
        return None
    return (f"warning: candidates={candidates} < {max(KS)}: ranks beyond the candidate pool are unreachable, "
            f"so hit@10/@20 are understated")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--mode")
    p.add_argument("--reranker")
    p.add_argument("--candidates", type=int)
    p.add_argument("--show-misses", action="store_true")
    p.add_argument("--append")
    a = p.parse_args(argv)

    s = get_settings()
    s = replace(
        s,
        retrieval_mode=a.mode or s.retrieval_mode,
        reranker=a.reranker or s.reranker,
        candidates=a.candidates or s.candidates,
        top_k=max(KS),
    )
    warning = candidate_warning(s.candidates, s.retrieval_mode, s.reranker)
    if warning:
        print(warning, file=sys.stderr)
    store = FilingStore(s.chroma_dir, get_embeddings(s))
    retriever = make_retriever(s, store)
    for t in TICKERS:  # warm-up: load models and build BM25 indexes before timing
        retriever.retrieve("warm up", [t])
    golden = json.loads((ROOT / "eval" / "golden_set.json").read_text(encoding="utf-8"))
    rows, times = evaluate(golden, lambda q, t: retriever.retrieve(q, [t]))
    config = {"mode": s.retrieval_mode, "reranker": s.reranker, "candidates": s.candidates,
              "chroma_dir": s.chroma_dir.name}
    md = render_markdown(a.label, config, summarize(rows), times)
    print(md)
    if a.show_misses:
        for r in rows:
            if not r["metrics"]["any@5"]:
                print(f"miss@5 {r['id']} ({r['category']}): ranks {r['ranks']}")
    if a.append:
        path = Path(a.append)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    main()
