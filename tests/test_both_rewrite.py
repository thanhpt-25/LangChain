"""The real classifier must give `both` questions a search_query that holds only the 10-K part.

Needs the router: RUN_INTEGRATION=1.
"""
import os
import re

import pytest

from finassist.config import get_settings

pytestmark = pytest.mark.integration

PRICE_WORDS = re.compile(r"\b(price|prices|market cap\w*|p/e|pe ratio|stock price|share price)\b|giá|vốn hóa", re.I)
VIETNAMESE = re.compile("[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)

BOTH_CASES = [
    ("What is NVIDIA's current stock price and how do export controls threaten it?", ["NVIDIA"]),
    ("What is Apple's current stock price, and what were its total net sales last fiscal year?", ["Apple"]),
    ("What is NVIDIA's market capitalization now, and what was its total revenue last fiscal year?", ["NVIDIA"]),
    ("Compare Apple's and Microsoft's latest annual revenue and tell me both current stock prices.", ["Apple", "Microsoft"]),
    ("What is Microsoft's current P/E ratio, and what is its biggest competitive risk?", ["Microsoft"]),
    ("Giá cổ phiếu NVIDIA hiện tại là bao nhiêu và rủi ro kiểm soát xuất khẩu của họ ra sao?", ["NVIDIA"]),
    ("Vốn hóa của Microsoft hiện nay và doanh thu năm tài chính gần nhất là bao nhiêu?", ["Microsoft"]),
    ("Is Apple's stock up today, and what risks does its 10-K describe for the supply chain?", ["Apple"]),
]


@pytest.fixture(scope="module")
def classify():
    from finassist.chains import build_llm, make_deps

    s = get_settings()
    return make_deps(s, store=object(), llm=build_llm(s)).classify


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
@pytest.mark.parametrize("question,companies", BOTH_CASES)
def test_both_questions_get_an_english_filing_only_query(classify, question, companies):
    r = classify(question)
    assert r.intent == "both"
    sq = (r.search_query or "").strip()
    assert sq, "a both question must carry a search_query"
    assert not PRICE_WORDS.search(sq), sq
    assert not VIETNAMESE.search(sq), sq
    for name in companies:
        assert name.lower() in sq.lower(), (name, sq)
