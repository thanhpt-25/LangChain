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
