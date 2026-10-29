"""Run the golden set through the app and score it with RAGAS.
Usage: uv run python -m eval.run_ragas
"""
from __future__ import annotations

import json
import math
import re
import time

import eval._ragas_compat  # noqa: F401,E402  (must precede ragas imports)

import pandas as pd  # noqa: E402
import ragas  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from eval.report import render_markdown, summarize  # noqa: E402
from finassist.chains import build_llm, format_context  # noqa: E402
from finassist.config import ROOT, get_settings  # noqa: E402
from finassist.runtime import ask, build_app  # noqa: E402
from finassist.vectorstore import get_embeddings  # noqa: E402

INPUT_COLS = ("user_input", "response", "retrieved_contexts", "reference")


def key_figure(ground_truth: str) -> str | None:
    m = re.search(r"\$([\d,]+(?:\.\d+)?)", ground_truth)
    return m.group(1) if m else None


def _nan_to_none(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def run_app(app, g: dict) -> dict:
    rec = {"id": g["id"], "category": g["category"], "question": g["question"],
           "ground_truth": g["ground_truth"], "expected_intent": g["expected_intent"],
           "predicted_intent": None, "answer": "", "retrieved": [], "contexts": [],
           "ticker": None, "fiscal_year": None, "revenue": None, "eps": None, "error": None}
    try:
        state = ask(app, g["question"], tags=["eval"])
        out = state["output"]
        rec.update(predicted_intent=out.intent, answer=out.answer, ticker=out.ticker,
                   fiscal_year=out.fiscal_year, eps=out.eps,
                   revenue=out.revenue.model_dump() if out.revenue else None)
        chunks = state.get("retrieved_chunks", [])
        rec["retrieved"] = [{"chunk_id": c.chunk_id, "item": c.item, "distance": c.score} for c in chunks]
        ctx = [f"[{c.chunk_id}] {c.text}" for c in chunks]
        if state.get("price_data"):
            ctx.append(format_context([], state["price_data"]))
        rec["contexts"] = ctx or ["(no context retrieved)"]
        if state.get("error"):
            rec["error"] = f"graph: {state['error']}"
    except Exception as e:  # one router hiccup must not kill the run
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return rec


def score(rows: list[dict], judge, emb) -> pd.DataFrame:
    ds = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"],
         "retrieved_contexts": r["contexts"], "reference": r["ground_truth"]} for r in rows])
    metrics = [Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall(), ResponseRelevancy()]
    res = evaluate(ds, metrics=metrics, llm=judge, embeddings=emb,
                   run_config=RunConfig(timeout=240, max_retries=10, max_workers=4))
    sc = res.to_pandas()
    cols = [c for c in sc.columns if c not in INPUT_COLS and pd.api.types.is_numeric_dtype(sc[c])]
    return sc[cols].reset_index(drop=True)


def main() -> None:
    t0 = time.time()
    s = get_settings()
    golden = json.loads((ROOT / "eval" / "golden_set.json").read_text())
    app = build_app(s)  # built once, reused for all questions

    recs = [run_app(app, g) for g in golden]
    correct = sum(r["predicted_intent"] == r["expected_intent"] for r in recs)
    for r, g in zip(recs, golden):
        r["answer_has_expected_number"] = None
        if g["category"] == "numeric":
            fig = key_figure(g["ground_truth"])
            r["answer_has_expected_number"] = bool(fig and fig in r["answer"])
            r["expected_figure"] = fig

    evalrecs = [r for r in recs if r["category"] != "trap" and not r["error"] and r["answer"]]
    failed = [r for r in recs if r["category"] != "trap" and r not in evalrecs]
    judge = LangchainLLMWrapper(build_llm(s, s.judge_model))
    emb = LangchainEmbeddingsWrapper(get_embeddings(s))
    scores = score(evalrecs, judge, emb)
    metric_cols = list(scores.columns)
    # one extra pass for rows where the judge produced NaN (RAGAS already retried each call internally)
    nan_rows = [i for i in range(len(scores)) if scores.iloc[i].isna().any()]
    if nan_rows:
        retry = score([evalrecs[i] for i in nan_rows], judge, emb)
        for j, i in enumerate(nan_rows):
            for c in metric_cols:
                if pd.isna(scores.at[i, c]) and c in retry.columns:
                    scores.at[i, c] = retry.at[j, c]

    for r in evalrecs + failed:
        r["scores"] = {}
    for i, r in enumerate(evalrecs):
        r["scores"] = {c: _nan_to_none(float(scores.at[i, c])) for c in metric_cols}
    for r in failed:  # app failure: scored as 0, flagged via `error`
        r["scores"] = {c: 0.0 for c in metric_cols}

    scored = [r for r in recs if "scores" in r]
    df = pd.DataFrame([{"id": r["id"], "category": r["category"], "question": r["question"],
                        **{c: (float("nan") if r["scores"][c] is None else r["scores"][c]) for c in metric_cols}}
                       for r in scored])
    summary = summarize(df, metric_cols)
    nan_counts = {c: int(df[c].isna().sum()) for c in metric_cols}
    faith = next((c for c in metric_cols if "faith" in c), metric_cols[0])
    recall = next((c for c in metric_cols if "recall" in c), metric_cols[0])
    order = df.assign(_f=df[faith].fillna(-1), _r=df[recall].fillna(-1)).sort_values(["_f", "_r"])
    worst = order.head(5).rename(columns={faith: "faithfulness"})[["id", "question", "faithfulness"]].to_dict("records")

    nums = [r for r in recs if r["category"] == "numeric"]
    ns = {"hits": sum(bool(r["answer_has_expected_number"]) for r in nums), "total": len(nums)}
    errs = [f"{r['id']}: {r['error']}" for r in recs if r["error"]]
    meta = {"model": s.judge_model, "n": len(scored), "ragas_version": ragas.__version__,
            "ragas_api": "legacy ragas.metrics classes (deprecated shims) + LangchainLLMWrapper/LangchainEmbeddingsWrapper + evaluate",
            "runtime_s": time.time() - t0, "nan_counts": nan_counts, "numeric_sanity": ns,
            "errors": "; ".join(errs) if errs else None}
    ia = {"correct": int(correct), "total": len(golden)}

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    for r in recs:
        r.pop("contexts", None)
    (out / "ragas_baseline.json").write_text(json.dumps(
        {"summary": summary, "nan_counts": nan_counts, "intent_accuracy": ia, "numeric_sanity": ns,
         "meta": {k: v for k, v in meta.items() if k != "errors"}, "per_question": recs}, indent=2))
    (out / "ragas_baseline.md").write_text(render_markdown(summary, worst, ia, meta))
    print((out / "ragas_baseline.md").read_text())


if __name__ == "__main__":
    main()
