import pytest
from pydantic import ValidationError
from finassist.schemas import FinanceAnswer, Money, Citation


def test_valid_answer_roundtrip():
    a = FinanceAnswer(
        intent="filing", ticker="AAPL", fiscal_year=2025,
        revenue=Money(value=416161.0, period="FY2025"), eps=7.46,
        risk_factors=["supply chain"], answer="ok",
        citations=[Citation(source="AAPL 10-K FY2025", item="7", chunk_id="AAPL-2025-7-001", quote="x")],
    )
    assert FinanceAnswer.model_validate_json(a.model_dump_json()) == a


def test_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        FinanceAnswer(intent="weather", answer="x")


def test_defaults_are_null_safe():
    a = FinanceAnswer(intent="out_of_scope", answer="no")
    assert a.revenue is None and a.risk_factors == [] and a.citations == []


def test_stock_prices_defaults_to_empty_list():
    a = FinanceAnswer(intent="price", answer="x")
    assert a.stock_prices == [] and a.stock_price is None
