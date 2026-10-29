from finassist.config import normalize_base_url, get_settings


def test_strips_trailing_v1():
    assert normalize_base_url("http://localhost:20128/v1") == "http://localhost:20128"
    assert normalize_base_url("http://localhost:20128/v1/") == "http://localhost:20128"
    assert normalize_base_url("http://localhost:20128") == "http://localhost:20128"


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("FINASSIST_LLM_MODEL", "kr/claude-sonnet-5.5")
    s = get_settings()
    assert s.base_url == "http://localhost:20128"
    assert s.llm_model == "kr/claude-sonnet-5.5"
    assert s.top_k == 5
