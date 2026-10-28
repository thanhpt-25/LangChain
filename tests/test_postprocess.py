from finassist.postprocess import MARKER_RE, extract_citations, ground_numbers
from finassist.schemas import Chunk, FinanceAnswer, Money


def ch(cid, text):
    t, fy, item, _ = cid.split("-")
    return Chunk(chunk_id=cid, text=text, ticker=t, fiscal_year=int(fy), item=item)


CHUNKS = [ch("AAPL-2025-8-000", "Total net sales 416,161 and diluted EPS 7.46"), ch("AAPL-2025-1A-003", "Supply risk")]


def test_marker_regex():
    assert MARKER_RE.findall("a [AAPL-2025-1A-003] b [NVDA-2026-7A-012]") == ["AAPL-2025-1A-003", "NVDA-2026-7A-012"]


def test_marker_regex_matches_item_15():
    assert MARKER_RE.findall("see [NVDA-2026-15-003]") == ["NVDA-2026-15-003"]


def test_extract_citations_dedupes_and_drops_unknown():
    ans = "x [AAPL-2025-8-000] y [AAPL-2025-8-000] z [AAPL-2025-1A-003] w [MSFT-2025-7-001]"
    cites = extract_citations(ans, CHUNKS)
    assert [c.chunk_id for c in cites] == ["AAPL-2025-8-000", "AAPL-2025-1A-003"]
    assert cites[0].item == "8" and cites[0].source == "AAPL 10-K FY2025"
    assert cites[0].quote.startswith("Total net sales")


def test_extract_citations_item_15():
    chunks = [ch("NVDA-2026-15-003", "Revenue 130,497")]
    cites = extract_citations("rev [NVDA-2026-15-003]", chunks)
    assert [(c.chunk_id, c.item) for c in cites] == [("NVDA-2026-15-003", "15")]


def test_ground_numbers_keeps_supported_and_nulls_unsupported():
    good = FinanceAnswer(intent="filing", answer="a", revenue=Money(value=416161, period="FY2025"), eps=7.46)
    assert ground_numbers(good, CHUNKS).revenue is not None
    assert ground_numbers(good, CHUNKS).eps == 7.46
    bad = FinanceAnswer(intent="filing", answer="a", revenue=Money(value=999999, period="FY2025"), eps=1.23)
    out = ground_numbers(bad, CHUNKS)
    assert out.revenue is None and out.eps is None


def test_ground_numbers_flattened_table_lines():
    chunks = [ch("AAPL-2025-8-001", "Net sales\n416,161\nDiluted\n4.90\n")]
    out = ground_numbers(FinanceAnswer(intent="filing", answer="a", revenue=Money(value=416161, period="FY2025"), eps=4.9), chunks)
    assert out.revenue is not None and out.eps == 4.9


def test_ground_numbers_rejects_substring_of_longer_number():
    out = ground_numbers(
        FinanceAnswer(intent="filing", answer="a", revenue=Money(value=6161, period="FY2025"), eps=7.46),
        [ch("AAPL-2025-8-000", "Total net sales 416,161 and diluted EPS 17.46")],
    )
    assert out.revenue is None and out.eps is None
    out2 = ground_numbers(
        FinanceAnswer(intent="filing", answer="a", revenue=Money(value=416161, period="FY2025"), eps=7.46),
        [ch("AAPL-2025-8-000", "sales 1,416,161 eps 7.465")],
    )
    assert out2.revenue is None and out2.eps is None
