from __future__ import annotations

import math

import pandas as pd


def summarize(df: pd.DataFrame, metric_cols: list[str]) -> dict:
    return {
        "mean": {c: float(df[c].mean()) for c in metric_cols},
        "by_category": {
            cat: {c: float(g[c].mean()) for c in metric_cols} for cat, g in df.groupby("category")
        },
    }


def _f(v: float) -> str:
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def _caveats(meta: dict) -> list[str]:
    out = ["## Caveats", ""]
    if meta.get("app_model", meta["model"]) == meta["model"]:
        out += [f"- The judge model (`{meta['model']}`) is the same model as the application LLM, so it grades its "
                "own output; this likely inflates faithfulness.", ""]
    if meta.get("errors"):
        out += [f"- App failures occurred in this run; failed rows were scored 0.0 and are listed above "
                f"under Run errors: {meta['errors']}", ""]
    else:
        out += ["- 0 app failures in this run (a failed row would be scored 0.0 and listed in `meta.errors`).", ""]
    out += ["- answer_relevancy is 0.0 for refusal or noncommittal answers, even when the answer is honest or "
            "grounded; this is a RAGAS design artifact, not necessarily a bad answer.", ""]
    if "numeric_sanity" in meta:
        out += ["- Numeric sanity is a substring check: a hit can be a derived figure (for example a subtraction) "
                "rather than one stated directly, and substring matching of short figures is loose.", ""]
    out += ["- The retrieved contexts passed to the judge are not stored in the JSON, so judge inputs cannot be "
            "audited after the fact.", ""]
    return out


def render_markdown(summary: dict, worst: list[dict], intent_acc: dict, meta: dict) -> str:
    cols = list(summary["mean"])
    lines = [
        "# RAGAS baseline",
        "",
        f"Judge/LLM: `{meta['model']}` · RAGAS-evaluated questions: {meta['n']}",
        "",
    ]
    if meta.get("ragas_version"):
        lines += [f"RAGAS version: {meta['ragas_version']} · API: {meta.get('ragas_api', 'n/a')}", ""]
    if meta.get("runtime_s") is not None:
        lines += [f"Runtime: {meta['runtime_s']:.0f}s", ""]
    lines += [
        "## Overall means",
        "",
        "| metric | mean | NaN count |", "|---|---|---|",
        *[f"| {c} | {_f(v)} | {meta.get('nan_counts', {}).get(c, 'n/a')} |" for c, v in summary["mean"].items()],
        "",
    ]
    nan_total = sum(meta.get("nan_counts", {}).values())
    if nan_total and meta.get("n"):
        lines += [f"**WARNING: {nan_total} NaN judge scores out of {meta['n'] * len(cols)}; "
                  "NaNs are excluded from means, not treated as 0.**", ""]
    lines += [
        "## By category",
        "",
        "| category | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
        *[f"| {cat} | " + " | ".join(_f(vals[c]) for c in cols) + " |" for cat, vals in summary["by_category"].items()],
        "",
        f"## Intent accuracy (all 30 questions, incl. traps): {intent_acc['correct']}/{intent_acc['total']}",
        "",
    ]
    if "numeric_sanity" in meta:
        ns = meta["numeric_sanity"]
        lines += [f"## Numeric sanity (key figure appears in answer): {ns['hits']}/{ns['total']}", ""]
    if meta.get("errors"):
        lines += [f"**Run errors:** {meta['errors']}", ""]
    lines += _caveats(meta)
    lines += [
        "## Worst cases (for Debug questions)",
        "",
        *[f"- **{w['id']}** (faithfulness {_f(w.get('faithfulness', float('nan')))}): {w['question']}\n  - Why: {w.get('note', 'TO ANALYZE')}" for w in worst],
    ]
    return "\n".join(lines) + "\n"
