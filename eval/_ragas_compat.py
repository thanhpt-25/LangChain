"""Compat shim: import this BEFORE ragas.

ragas 0.4.3 (latest release) imports ChatVertexAI / VertexAI from langchain_community, which
langchain-community 0.4.x no longer ships (and we cannot downgrade langchain-core 1.x).
We never use Vertex, so register inert stubs only if the real modules are missing.
"""
from __future__ import annotations

import sys
import types


def _stub(name: str, **attrs) -> None:
    try:
        __import__(name)
        return
    except ModuleNotFoundError as e:
        if e.name != name:  # the module exists but one of its own imports failed: surface that
            raise
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


_stub("langchain_community.chat_models.vertexai", ChatVertexAI=type("ChatVertexAI", (), {}))
import langchain_community.llms as _llms  # noqa: E402

if not hasattr(_llms, "VertexAI"):
    _llms.VertexAI = type("VertexAI", (), {})
