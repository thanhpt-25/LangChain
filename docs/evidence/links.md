# Liên kết bằng chứng

Dự án LangSmith: `finassist-w1`

- Intent `both` (price + filing retrieval), question "What is NVIDIA's current stock price and how do export controls threaten it?":
  https://smith.langchain.com/o/80181169-cd2d-47d8-b72d-0a72214e4700/projects/p/a7764340-befe-428d-9018-0422833d27ab/r/01a18548-f611-70d1-ad75-5bb267a4a30e?trace_id=01a18548-f611-70d1-ad75-5bb267a4a30e&start_time=2026-10-29T01:31:53.745703
  Nodes seen: classify_intent, price_tool, retrieve, generate_answer, format_output.
- Intent `out_of_scope`, question "Should I buy NVIDIA stock now?": run id `01a18549-d281-74e8-93cf-cb568132d110` in the same project.
  Nodes seen: classify_intent, refuse only (no retrieve, no price_tool, no generate_answer).

A text rendering of both traces, fetched through the LangSmith API, is in `trace_tree.md`.

Regenerate:

- Latest trace URL: `uv run python scripts/trace_link.py`
- Trace tree markdown: `uv run python scripts/trace_evidence.py --limit 2 --out docs/evidence/trace_tree.md`

Screenshots (`docs/evidence/trace_tree.png`) were not captured automatically, because the LangSmith UI requires a login. A human can open the URL above and capture them if desired.
