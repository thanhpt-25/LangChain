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


def test_price_timeout_default_and_env(monkeypatch):
    monkeypatch.delenv("FINASSIST_PRICE_TIMEOUT", raising=False)
    assert get_settings().price_timeout == 8.0
    monkeypatch.setenv("FINASSIST_PRICE_TIMEOUT", "2.5")
    assert get_settings().price_timeout == 2.5


def test_retrieval_settings_defaults(monkeypatch):
    for name in ("FINASSIST_RETRIEVAL", "FINASSIST_RERANKER", "FINASSIST_CANDIDATES",
                 "FINASSIST_EMBED_PREFIX", "FINASSIST_TABLE_ROWS", "FINASSIST_CHROMA_DIR"):
        monkeypatch.delenv(name, raising=False)
    s = get_settings()
    assert s.retrieval_mode == "dense"
    assert s.reranker == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.candidates == 30
    assert s.embed_prefix is False
    assert s.table_rows is False
    assert s.chroma_dir.name == "chroma"


def test_retrieval_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FINASSIST_RETRIEVAL", "hybrid")
    monkeypatch.setenv("FINASSIST_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    monkeypatch.setenv("FINASSIST_CANDIDATES", "40")
    monkeypatch.setenv("FINASSIST_EMBED_PREFIX", "1")
    monkeypatch.setenv("FINASSIST_TABLE_ROWS", "1")
    monkeypatch.setenv("FINASSIST_CHROMA_DIR", str(tmp_path / "alt"))
    s = get_settings()
    assert (s.retrieval_mode, s.candidates) == ("hybrid", 40)
    assert s.reranker == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.embed_prefix is True and s.table_rows is True
    assert s.chroma_dir == tmp_path / "alt"
