import pandas as pd

from eval.report import render_markdown, summarize


def test_summarize_means_overall_and_by_category():
    df = pd.DataFrame({
        "category": ["numeric", "numeric", "risk"],
        "faithfulness": [1.0, 0.5, 0.2],
        "context_recall": [1.0, 1.0, 0.0],
    })
    s = summarize(df, ["faithfulness", "context_recall"])
    assert round(s["mean"]["faithfulness"], 3) == 0.567
    assert s["by_category"]["numeric"]["faithfulness"] == 0.75
    assert s["by_category"]["risk"]["context_recall"] == 0.0


def test_render_markdown_has_all_sections():
    md = render_markdown(
        {"mean": {"faithfulness": 0.8}, "by_category": {"numeric": {"faithfulness": 0.9}}},
        [{"id": "q03", "question": "Q?", "faithfulness": 0.1, "note": "retrieved wrong item"}],
        {"correct": 4, "total": 5},
        {"model": "kr/claude-sonnet-5.5", "n": 25},
    )
    for needle in ["RAGAS baseline", "faithfulness", "numeric", "q03", "4/5", "kr/claude-sonnet-5.5"]:
        assert needle in md


def _md(**meta):
    base = {"model": "kr/claude-sonnet-5.5", "n": 25, "numeric_sanity": {"hits": 7, "total": 10}}
    base.update(meta)
    return render_markdown(
        {"mean": {"faithfulness": 0.8}, "by_category": {"numeric": {"faithfulness": 0.9}}},
        [{"id": "q03", "question": "Q?", "faithfulness": 0.1, "note": "n"}],
        {"correct": 4, "total": 5}, base,
    )


def test_caveats_section_covers_known_limitations():
    md = _md()
    assert "## Caveats" in md
    assert "grades its own output" in md and "inflates faithfulness" in md
    assert "0 app failures in this run" in md
    assert "answer_relevancy" in md and "RAGAS design artifact" in md
    assert "derived figure" in md and "substring" in md
    assert "contexts" in md and "not stored" in md
    assert "grounded but misread" in md


def test_caveats_report_app_failures_dynamically():
    md = _md(errors="q07: boom")
    assert "0 app failures" not in md
    assert "scored 0.0" in md and "q07: boom" in md


def test_self_grading_caveat_omitted_when_judge_differs_from_app_model():
    assert "grades its own output" not in _md(app_model="some/other-model")
    assert "grades its own output" in _md(app_model="kr/claude-sonnet-5.5")


def test_title_comes_from_meta_with_baseline_default():
    assert _md().startswith("# RAGAS baseline")
    assert _md(title="RAGAS retrieval_v2").startswith("# RAGAS retrieval_v2")
