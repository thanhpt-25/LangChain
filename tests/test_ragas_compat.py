def test_ragas_llms_imports():
    import eval._ragas_compat  # noqa: F401
    import ragas.llms
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: F401
    from ragas.metrics import Faithfulness, LLMContextRecall  # noqa: F401

    assert hasattr(ragas.llms, "LangchainLLMWrapper")


import sys

import pytest

from eval._ragas_compat import _stub


def test_stub_registers_only_when_exact_module_is_missing(monkeypatch):
    name = "email.definitely_missing_xyz"
    monkeypatch.delitem(sys.modules, name, raising=False)
    _stub(name, Thing=int)
    assert sys.modules[name].Thing is int
    first = sys.modules[name]
    _stub(name, Thing=str)  # idempotent: the stub is importable now, so it is left alone
    assert sys.modules[name] is first and first.Thing is int
    monkeypatch.delitem(sys.modules, name)


def test_stub_leaves_a_real_module_alone():
    import json

    _stub("json", Fake=1)
    assert sys.modules["json"] is json and not hasattr(json, "Fake")


@pytest.mark.parametrize("body", [
    "raise ImportError('boom')",
    "raise ModuleNotFoundError('other dep', name='some_other_dep')",
])
def test_stub_reraises_unrelated_import_errors(tmp_path, monkeypatch, body):
    (tmp_path / "broken_compat_mod.py").write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "broken_compat_mod", raising=False)
    with pytest.raises(ImportError):
        _stub("broken_compat_mod")
    assert "broken_compat_mod" not in sys.modules
