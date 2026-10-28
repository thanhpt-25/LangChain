import os
import pytest
from finassist.config import get_settings

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
def test_router_answers():
    from langchain_anthropic import ChatAnthropic
    s = get_settings()
    llm = ChatAnthropic(model=s.llm_model, base_url=s.base_url, api_key=s.api_key, max_tokens=20)
    assert "ok" in llm.invoke("Reply with: ok").text.lower()
