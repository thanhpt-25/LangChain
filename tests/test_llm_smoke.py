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


CLASSIFY_CASES = [
    ("What was Apple's total net sales last fiscal year?", "filing", ["AAPL"]),
    ("What is Microsoft's current stock price?", "price", ["MSFT"]),
    ("NVIDIA price now and its export control risk?", "both", ["NVDA"]),
    ("Should I buy NVIDIA stock now?", "out_of_scope", None),
    ("What was Tesla's revenue?", "out_of_scope", None),
    ("Thời tiết Hà Nội hôm nay thế nào?", "out_of_scope", None),
]


@pytest.fixture(scope="module")
def live_deps():
    from finassist.chains import build_llm, make_deps
    s = get_settings()
    return s, make_deps(s, None, build_llm(s))


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
@pytest.mark.parametrize("question,intent,tickers", CLASSIFY_CASES)
def test_classify_live(live_deps, question, intent, tickers):
    res = live_deps[1].classify(question)
    assert res.intent == intent
    if tickers is not None:
        assert res.tickers == tickers


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
def test_extractor_live(live_deps):
    from finassist.schemas import Chunk
    c = Chunk(chunk_id="AAPL-2025-7-001", text="Total net sales 416,161 for fiscal 2025. Diluted EPS 7.46.",
              ticker="AAPL", fiscal_year=2025, item="7")
    out = live_deps[1].structure("Revenue and EPS?", "Total net sales were $416,161 million [AAPL-2025-7-001]; diluted EPS $7.46.",
                                 [c], [], "filing")
    assert out.revenue is not None and out.revenue.value == 416161
    assert out.eps == 7.46


@pytest.fixture(scope="module")
def live_store(tmp_path_factory):
    from finassist.ingest import load_filing
    from finassist.vectorstore import FilingStore, get_embeddings
    s = get_settings()
    store = FilingStore(tmp_path_factory.mktemp("chroma"), get_embeddings(s))
    for t in ("AAPL", "NVDA"):
        store.upsert(load_filing(t, s.raw_dir))
    return store


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
def test_generate_live_cites_retrieved_chunk(live_store):
    from finassist.chains import build_llm, make_deps
    from finassist.postprocess import marker_ids
    s = get_settings()
    deps = make_deps(s, live_store, build_llm(s))
    q = "What were Apple's total net sales last fiscal year?"
    chunks = deps.retrieve(q, ["AAPL"])[:3]
    ans = deps.generate(q, chunks, [])
    ids = {c.chunk_id for c in chunks}
    found = set(marker_ids(ans))
    assert found and found <= ids


def _run_graph(live_store, q):
    from finassist.chains import build_llm, make_deps
    from finassist.graph import build_graph
    s = get_settings()
    return build_graph(make_deps(s, live_store, build_llm(s))).invoke({"question": q})


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
def test_graph_live_aapl_revenue(live_store):
    st = _run_graph(live_store, "What were Apple's total net sales in its most recent fiscal year?")
    out = st["output"]
    print("CHUNKS", [(c.chunk_id, c.score) for c in st["retrieved_chunks"]], "\nANSWER", st["answer"], "\nOUT", out)
    assert out.intent == "filing"
    assert out.citations
    assert out.revenue is not None and out.revenue.value == 416161
    assert out.ticker == "AAPL"
    assert out.fiscal_year == 2025


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
def test_graph_live_nvda_eps(live_store):
    st = _run_graph(live_store, "What was NVIDIA's diluted earnings per share in its most recent fiscal year?")
    out = st["output"]
    print("CHUNKS", [(c.chunk_id, c.score) for c in st["retrieved_chunks"]], "\nANSWER", st["answer"], "\nOUT", out)
    assert out.intent == "filing"
    assert out.citations
    assert out.ticker == "NVDA"
    assert out.eps == 4.90


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="integration")
def test_graph_live_vietnamese_export_control_hits_item_1a(live_store):
    st = _run_graph(live_store, "Rủi ro kiểm soát xuất khẩu đối với NVIDIA là gì?")
    out = st["output"]
    print("SEARCH_QUERY", st["search_query"], "\nCHUNKS", [(c.chunk_id, c.item, round(c.score, 3)) for c in st["retrieved_chunks"]],
          "\nANSWER", st["answer"])
    assert any(c.item == "1A" for c in st["retrieved_chunks"])
    assert out.answer.strip() and out.citations
    assert {c.chunk_id for c in out.citations} <= {c.chunk_id for c in st["retrieved_chunks"]}
