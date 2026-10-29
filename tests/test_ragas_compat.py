def test_ragas_llms_imports():
    import eval._ragas_compat  # noqa: F401
    import ragas.llms
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: F401
    from ragas.metrics import Faithfulness, LLMContextRecall  # noqa: F401

    assert hasattr(ragas.llms, "LangchainLLMWrapper")
