import pytest

pytest.importorskip("ragas")


def test_tag_is_required_and_parsed():
    from eval.run_ragas import build_parser

    assert build_parser().parse_args(["--tag", "x"]).tag == "x"
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
