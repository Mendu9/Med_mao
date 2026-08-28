"""arch-L2 / scope item 19 — consolidate duplicate embedding model instances.

`mao/rag/embedder.py` and `mao/rag/chunker.py` each constructed their own
`SentenceTransformer(cfg.embed_model)`, loading the same ~400 MB of weights into
memory twice. Scope item 19 asked for exactly this consolidation and the
architecture review recorded it as NOT DONE.
"""
from __future__ import annotations

import inspect

from mao.rag import chunker, embedder


class TestOneOwnerOfTheEmbeddingModel:
    def test_the_chunker_does_not_construct_its_own(self) -> None:
        assert "SentenceTransformer(" not in inspect.getsource(chunker)

    def test_the_embedder_is_the_owner(self) -> None:
        assert "SentenceTransformer(" in inspect.getsource(embedder)

    def test_the_chunker_asks_the_embedder(self) -> None:
        assert "get_embedding_model" in inspect.getsource(chunker._get_embed_model)

    def test_both_paths_return_the_same_object(self, monkeypatch) -> None:
        sentinel = object()
        monkeypatch.setattr(embedder, "_model", sentinel)
        assert embedder.get_embedding_model() is sentinel
        assert chunker._get_embed_model() is sentinel

    def test_the_chunker_degrades_rather_than_raising(self, monkeypatch) -> None:
        """Boundary similarity is an optimisation; it must not fail ingestion."""

        def _boom() -> object:
            raise RuntimeError("no model")

        monkeypatch.setattr(embedder, "get_embedding_model", _boom)
        assert chunker._get_embed_model() is None
