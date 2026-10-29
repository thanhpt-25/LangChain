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
