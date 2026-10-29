from finassist.ingest import html_to_text, split_items, chunk_filing

TOC = "Table of Contents\nItem 1.\nBusiness\nItem 1A.\nRisk Factors\nItem 7.\nMD&A\nItem 8.\nFinancial Statements\n"
BODY = (
    "Item 1.  Business\n" + "We make devices. " * 40 + "\n"
    "Item 1A.  Risk Factors\n" + "Supply chain risk matters. " * 80 + "\n"
    "Item 1B.  Unresolved Staff Comments\nNone\n"
    "Item 7.  Management's Discussion\n" + "Revenue grew. " * 50 + "\n"
    "Item 8.  Financial Statements\n" + "Total net sales 416,161. " * 30 + "\n"
    "Item 10.  Directors\n" + "Board stuff. " * 20 + "\n"
)


def test_split_items_uses_body_not_toc():
    items = split_items(TOC + BODY)
    assert set(items) == {"1", "1A", "7", "8"}
    assert "Supply chain risk" in items["1A"]
    assert "Board stuff" not in items["8"]      # Item 10 must not be read as Item 1


def test_chunks_have_metadata_and_stable_ids():
    chunks = chunk_filing("AAPL", 2025, TOC + BODY, "http://x")
    assert chunks
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    first_1a = next(c for c in chunks if c.item == "1A")
    assert first_1a.chunk_id == "AAPL-2025-1A-000"
    assert first_1a.ticker == "AAPL" and first_1a.fiscal_year == 2025
    assert all(len(c.text) <= 1100 for c in chunks)
    assert chunk_filing("AAPL", 2025, TOC + BODY, "http://x") == chunks   # deterministic


def test_html_to_text_drops_script_and_hidden_xbrl():
    html = "<html><body><ix:header>HIDDEN</ix:header><script>x=1</script><p>Item 1.</p><p>Hello&nbsp;world</p></body></html>"
    text = html_to_text(html)
    assert "HIDDEN" not in text and "x=1" not in text
    assert "Hello world" in text


XREF = (
    "Item 1.  Business\n" + "We make devices. " * 40 + "\n"
    "Item 1A.  Risk Factors\n"
    "Early risk text. " * 20 + "\n"
    "As discussed in Part I,\nItem 1A\nof this Form 10-K. More risk. SENTINEL_AFTER_1A\n"
    "See also\nItem 8\nof this report\nSENTINEL_AFTER_8REF " + "Supply chain risk matters. " * 40 + "\n"
    "Item 1B.  Unresolved Staff Comments\nNone\n"
    "Item 8.  Financial Statements\n" + "Total net sales 416,161. " * 30 + "\n"
)


def test_cross_references_are_not_headings():
    items = split_items(XREF)
    assert "SENTINEL_AFTER_1A" in items["1A"]
    assert "SENTINEL_AFTER_8REF" in items["1A"]      # stays in the section it was in
    assert "Total net sales" not in items["1A"]
    assert "Total net sales" in items["8"]
    assert "SENTINEL_AFTER_8REF" not in items["8"]


def test_inline_cross_reference_with_lowercase_remainder_rejected():
    text = XREF.replace("Item 8\nof this report", "Item 8 of this report")
    items = split_items(text)
    assert "SENTINEL_AFTER_8REF" in items["1A"]


def test_real_headings_same_line_and_next_line_recognised():
    same = split_items("Item 1A.  Risk Factors\n" + "Risk body. " * 20)
    nxt = split_items("Item 1A.\nRisk Factors\n" + "Risk body. " * 20)
    assert "Risk body" in same["1A"]
    assert "Risk body" in nxt["1A"]


RUNNING_HEADER = (
    "Item 1.  Business\n" + "We make devices. " * 40 + "\n"
    "Item 7.  Management's Discussion\nPAGE_ONE_TEXT " + "Revenue grew. " * 30 + "\n"
    "\n43\n\nPART II\nItem 7\n\nPAGE_TWO_TEXT " + "Margins moved. " * 30 + "\n"
    "\n44\n\nPART II\nItem 7\n\nPAGE_THREE_TEXT " + "Cash flow rose. " * 30 + "\n"
    "Item 8.  Financial Statements\n" + "Total net sales 416,161. " * 30 + "\n"
)


def test_repeated_running_page_header_does_not_truncate_section():
    """MSFT repeats 'PART II / Item N' at the top of every page; the section must span all pages."""
    items = split_items(RUNNING_HEADER)
    assert "PAGE_ONE_TEXT" in items["7"]
    assert "PAGE_TWO_TEXT" in items["7"]
    assert "PAGE_THREE_TEXT" in items["7"]
    assert "Total net sales" not in items["7"]
    assert "Total net sales" in items["8"]


ITEM15 = (
    "Item 1.  Business\n" + "We make chips. " * 40 + "\n"
    "Item 8.  Financial Statements\nSee Item 15 for the statements.\n"
    "Item 9.  Changes in Accountants\nNone\n"
    "Item 15.  Exhibits and Financial Statement Schedules\n"
    + "Net income 4,368. Diluted EPS 1.76. " * 30 + "\n"
    "Item 16.  Form 10-K Summary\nNone\n"
)


def test_item_15_is_kept_and_chunked():
    items = split_items(ITEM15)
    assert "Net income 4,368" in items["15"]
    assert "Form 10-K Summary" not in items["15"]
    chunks = chunk_filing("NVDA", 2026, ITEM15, "http://x")
    assert {c.item for c in chunks} >= {"1", "15"}
    assert any(c.chunk_id == "NVDA-2026-15-000" and c.item == "15" for c in chunks)


def test_page_header_item_with_lowercase_continuation_does_not_split_section():
    text = (
        "Item 1A.  Risk Factors\n" + "Risk one. " * 30 + "\n12\n\nItem 7\nand other factors, SENTINEL_STAYS_IN_1A. " + "More risk. " * 30 + "\n"
        "Item 7.  Management's Discussion\n" + "Revenue grew. " * 30 + "\n"
    )
    items = split_items(text)
    assert "SENTINEL_STAYS_IN_1A" in items["1A"]
    assert "SENTINEL_STAYS_IN_1A" not in items["7"]
    assert "Revenue grew" in items["7"]


def test_toc_followed_by_body_resolves_to_body_even_with_page_numbers():
    toc = "Item 1.\nBusiness\n3\nItem 1A.\nRisk Factors\n9\nItem 7.\nManagement's Discussion\n25\nItem 8.\nFinancial Statements\n40\n"
    body = (
        "Item 1.  Business\n" + "We make devices. " * 40 + "\n"
        "Item 1A.  Risk Factors\n" + "Supply risk. " * 40 + "\n"
        "Item 7.  Management's Discussion\n" + "BODY_SEVEN " * 40 + "\n"
        "Item 8.  Financial Statements\n" + "BODY_EIGHT " * 40 + "\n"
    )
    items = split_items(toc + body)
    assert "BODY_SEVEN" in items["7"] and "BODY_EIGHT" not in items["7"]
    assert len(items["7"]) > 300 and "BODY_EIGHT" in items["8"]


def test_load_filing_tolerates_invalid_bytes(tmp_path):
    from finassist.ingest import load_filing

    (tmp_path / "AAPL.json").write_text('{"fiscal_year": 2025, "source_url": "u"}', encoding="utf-8")
    (tmp_path / "AAPL.html").write_bytes(b"<html><body>Item 1.  Business\n" + b"We make \xff devices. " * 40 + b"</body></html>")
    assert load_filing("AAPL", tmp_path)


def test_fetch_filings_find_10k(tmp_path):
    import importlib.util
    from pathlib import Path

    import pytest

    spec = importlib.util.spec_from_file_location("fetch_filings", Path(__file__).parent.parent / "scripts" / "fetch_filings.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.find_10k({"form": ["8-K", "10-K", "10-K"]}, "AAPL") == 1
    with pytest.raises(SystemExit, match="NVDA: no 10-K"):
        mod.find_10k({"form": ["8-K", "10-Q"]}, "NVDA")
