"""Print a compact tree of the latest root run(s) in the finassist LangSmith project.

Usage: uv run python scripts/trace_evidence.py [--limit N] [--out FILE.md]
"""
import argparse
import json
import os
import warnings

from langsmith import Client

from finassist.config import get_settings

NODES = {"classify_intent", "price_tool", "retrieve", "generate_answer", "format_output", "refuse"}


def _dur(run) -> str:
    if run.start_time and run.end_time:
        return f"{(run.end_time - run.start_time).total_seconds():.2f}s"
    return "?"


def _short(v, n=160) -> str:
    s = json.dumps(v, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 3] + "..."


def _find_chunks(o, acc):
    if isinstance(o, dict):
        if "chunk_id" in o:
            acc.append(o["chunk_id"])
        for v in o.values():
            _find_chunks(v, acc)
    elif isinstance(o, list):
        for v in o:
            _find_chunks(v, acc)


def _details(run) -> list[str]:
    out = run.outputs or {}
    ins = run.inputs or {}
    lines = []
    if run.name == "classify_intent":
        lines.append(f"intent = {out.get('intent')}")
    elif run.name == "price_tool":
        lines.append(f"price_data = {_short(out.get('price_data'))}")
    elif run.name == "retrieve":
        ids: list[str] = []
        _find_chunks(out, ids)
        lines.append(f"{len(ids)} chunks: {', '.join(ids)}")
    elif run.name in ("generate_answer", "refuse", "format_output"):
        for k, v in out.items():
            if k in ("final", "answer", "draft", "refusal", "result"):
                lines.append(f"{k} = {_short(v, 200)}")
    if run.run_type == "llm":
        msgs = ins.get("messages") or ins.get("prompts") or ins
        lines.append(f"prompt = {_short(msgs, 300)}")
    return lines


def render(client, root) -> list[str]:
    runs = list(client.list_runs(trace_id=root.trace_id, project_id=root.session_id, limit=100))
    by_parent: dict = {}
    for r in runs:
        by_parent.setdefault(r.parent_run_id, []).append(r)
    for v in by_parent.values():
        v.sort(key=lambda r: r.start_time)
    lines = [
        f"## Trace `{root.name}` ({root.id})",
        "",
        f"- question: {_short((root.inputs or {}).get('question'), 200)}",
        f"- tags: {root.tags}",
        f"- metadata app_version: {(root.extra or {}).get('metadata', {}).get('app_version')}",
        f"- duration: {_dur(root)}",
        "",
        "```",
    ]

    def walk(r, depth):
        interesting = r.name in NODES or r.run_type == "llm" or depth == 0
        if interesting:
            lines.append(f"{'  ' * depth}{r.name} [{r.run_type}] {_dur(r)}")
            for d in _details(r):
                lines.append(f"{'  ' * depth}  - {d}")
            depth += 1
        for c in by_parent.get(r.id, []):
            walk(c, depth)

    walk(root, 0)
    lines.append("```")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    get_settings()
    warnings.simplefilter("ignore")
    client = Client()
    project = os.getenv("LANGSMITH_PROJECT", "finassist-w1")
    roots = list(client.list_runs(project_name=project, is_root=True, limit=a.limit))
    if not roots:
        raise SystemExit(f"no runs in project {project}")
    lines = [f"# LangSmith trace tree (project {project})", ""]
    for root in roots:
        lines += render(client, root) + [""]
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
