"""Tests for services/retriever.py

Focuses on pure functions (_normalize_scores, _hybrid_merge) and cache management.
retrieve() and retrieve_multi_query() are integration-heavy (ChromaDB + embedder),
so they are tested with full mocks.
"""
import pytest
from unittest.mock import MagicMock, patch

import services.retriever as retriever_module
from services.retriever import _normalize_scores, _hybrid_merge, invalidate_bm25_cache


# ---------------------------------------------------------------------------
# _normalize_scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:
    def test_empty_returns_empty(self):
        assert _normalize_scores([]) == {}

    def test_single_item_returns_one(self):
        result = _normalize_scores([("a", 5.0)])
        assert result["a"] == 1.0

    def test_equal_scores_all_one(self):
        result = _normalize_scores([("a", 3.0), ("b", 3.0)])
        assert result["a"] == 1.0
        assert result["b"] == 1.0

    def test_min_max_normalization(self):
        result = _normalize_scores([("a", 10.0), ("b", 5.0), ("c", 0.0)])
        assert abs(result["a"] - 1.0) < 1e-9
        assert abs(result["b"] - 0.5) < 1e-9
        assert abs(result["c"] - 0.0) < 1e-9

    def test_preserves_ordering(self):
        result = _normalize_scores([("x", 100.0), ("y", 50.0), ("z", 1.0)])
        assert result["x"] > result["y"] > result["z"]


# ---------------------------------------------------------------------------
# _hybrid_merge
# ---------------------------------------------------------------------------

class TestHybridMerge:
    def test_keyword_match_boosts_ranking(self):
        # "a" has slightly higher vector, but "b" has a huge BM25 keyword match
        # "c" is a third item to break min-max symmetry
        vec = [("a", 0.9), ("b", 0.85), ("c", 0.3)]
        bm25 = [("b", 10.0), ("a", 0.5), ("c", 0.1)]
        scores = _hybrid_merge(vec, bm25)
        # "b" should beat "a" because its BM25 boost outweighs the small vector gap
        assert scores["b"] > scores["a"]

    def test_vector_only_item_gets_partial_score(self):
        vec = [("a", 0.9)]
        bm25 = [("b", 5.0)]
        scores = _hybrid_merge(vec, bm25)
        # "a" has vector score but no BM25; "b" has BM25 but no vector
        assert "a" in scores
        assert "b" in scores

    def test_both_signals_beat_one(self):
        vec = [("a", 0.9), ("b", 0.85)]
        bm25 = [("a", 8.0), ("b", 0.1)]
        scores = _hybrid_merge(vec, bm25)
        # "a" is strong in both — should dominate
        assert scores["a"] > scores["b"]

    def test_empty_bm25_uses_vector_only(self):
        vec = [("a", 0.9), ("b", 0.5)]
        scores = _hybrid_merge(vec, [])
        # With no BM25, only vector contributes (scaled by ALPHA)
        assert scores["a"] > scores["b"]

    def test_empty_vector_uses_bm25_only(self):
        bm25 = [("a", 10.0), ("b", 2.0)]
        scores = _hybrid_merge([], bm25)
        assert scores["a"] > scores["b"]

    def test_shared_item_gets_combined_score(self):
        vec = [("a", 0.8)]
        bm25 = [("a", 5.0)]
        scores = _hybrid_merge(vec, bm25)
        # Both normalized to 1.0 (single item each), so combined = ALPHA + BETA
        expected = retriever_module.ALPHA * 1.0 + retriever_module.BETA * 1.0
        assert abs(scores["a"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# BM25 cache invalidation
# ---------------------------------------------------------------------------

class TestBm25CacheInvalidation:
    def setup_method(self):
        # Reset cache state before each test
        retriever_module._bm25_dirty = True
        retriever_module._bm25_index = None
        retriever_module._bm25_corpus = None

    def test_invalidate_sets_dirty_flag(self):
        retriever_module._bm25_dirty = False
        invalidate_bm25_cache()
        assert retriever_module._bm25_dirty is True

    def test_get_bm25_empty_corpus_returns_none(self):
        with patch("services.retriever.get_all_chunks", return_value=[]):
            index, corpus, _ = retriever_module._get_bm25()
        assert index is None
        assert corpus == []

    def test_get_bm25_builds_index_from_chunks(self):
        fake_chunks = [
            {"id": "doc1_chunk_0", "text": "hello world", "meta": {"doc_id": "doc1"}},
            {"id": "doc1_chunk_1", "text": "foo bar baz", "meta": {"doc_id": "doc1"}},
        ]
        with patch("services.retriever.get_all_chunks", return_value=fake_chunks):
            index, corpus, _ = retriever_module._get_bm25()

        assert index is not None
        assert len(corpus) == 2
        assert retriever_module._bm25_dirty is False

    def test_get_bm25_uses_cache_when_not_dirty(self):
        retriever_module._bm25_dirty = False
        sentinel_index = object()
        sentinel_corpus = [{"id": "x", "text": "cached", "meta": {}}]
        retriever_module._bm25_index = sentinel_index
        retriever_module._bm25_corpus = sentinel_corpus
        retriever_module._bm25_corpus_by_id = {"x": sentinel_corpus[0]}

        with patch("services.retriever.get_all_chunks") as mock_get:
            index, corpus, _ = retriever_module._get_bm25()
            mock_get.assert_not_called()

        assert index is sentinel_index
        assert corpus is sentinel_corpus

    def test_invalidate_forces_rebuild(self):
        # Pre-populate cache
        retriever_module._bm25_dirty = False
        retriever_module._bm25_index = object()
        retriever_module._bm25_corpus = []

        invalidate_bm25_cache()

        fake_chunks = [{"id": "doc_chunk_0", "text": "new data", "meta": {}}]
        with patch("services.retriever.get_all_chunks", return_value=fake_chunks):
            index, corpus, _ = retriever_module._get_bm25()

        assert index is not None
        assert len(corpus) == 1


# ---------------------------------------------------------------------------
# retrieve() — mocked integration
# ---------------------------------------------------------------------------

class TestRetrieve:
    def _make_chroma_result(self, docs, metas, dists):
        return {"documents": [docs], "metadatas": [metas], "distances": [dists]}

    def test_returns_empty_when_no_vector_results(self):
        with patch("services.retriever.embed_query", return_value=[0.0] * 384), \
             patch("services.retriever.query_chunks", return_value={"documents": [[]], "metadatas": [[]], "distances": [[]]}):
            citations, has_relevant = retriever_module.retrieve("test question")

        assert citations == []
        assert has_relevant is False

    def test_returns_empty_documents_list(self):
        with patch("services.retriever.embed_query", return_value=[0.0] * 384), \
             patch("services.retriever.query_chunks", return_value={"documents": [], "metadatas": [], "distances": []}):
            citations, has_relevant = retriever_module.retrieve("test question")

        assert citations == []
        assert has_relevant is False

    def test_filters_chunks_above_relevance_threshold(self):
        docs = ["irrelevant chunk"]
        metas = [{"doc_id": "doc1", "filename": "file.txt", "chunk_index": 0}]
        dists = [0.95]  # above CHUNK_RELEVANCE_THRESHOLD (0.80)

        retriever_module._bm25_dirty = True
        with patch("services.retriever.embed_query", return_value=[0.0] * 384), \
             patch("services.retriever.query_chunks", return_value=self._make_chroma_result(docs, metas, dists)), \
             patch("services.retriever.get_all_chunks", return_value=[]):
            citations, has_relevant = retriever_module.retrieve("test question")

        assert citations == []
        assert has_relevant is False

    def test_returns_relevant_citations_below_threshold(self):
        docs = ["very relevant chunk about the topic"]
        metas = [{"doc_id": "doc1", "filename": "file.txt", "chunk_index": 0}]
        dists = [0.20]  # well below both thresholds

        retriever_module._bm25_dirty = True
        with patch("services.retriever.embed_query", return_value=[0.0] * 384), \
             patch("services.retriever.query_chunks", return_value=self._make_chroma_result(docs, metas, dists)), \
             patch("services.retriever.get_all_chunks", return_value=[]):
            citations, has_relevant = retriever_module.retrieve("test question")

        assert len(citations) == 1
        assert citations[0].document == "file.txt"
        assert citations[0].chunk_text == docs[0]
        assert has_relevant is True

    def test_score_reflects_vector_similarity_when_no_bm25(self):
        docs = ["chunk text"]
        metas = [{"doc_id": "d1", "filename": "f.txt", "chunk_index": 0}]
        dists = [0.30]

        retriever_module._bm25_dirty = True
        with patch("services.retriever.embed_query", return_value=[0.0] * 384), \
             patch("services.retriever.query_chunks", return_value=self._make_chroma_result(docs, metas, dists)), \
             patch("services.retriever.get_all_chunks", return_value=[]):
            citations, _ = retriever_module.retrieve("q")

        # With no BM25 corpus, score = vector similarity (1 - distance)
        assert abs(citations[0].score - round(1 - 0.30, 4)) < 1e-4

    def test_doc_ids_filter_skips_bm25(self):
        docs = ["chunk"]
        metas = [{"doc_id": "d1", "filename": "f.txt", "chunk_index": 0}]
        dists = [0.10]

        retriever_module._bm25_dirty = True
        with patch("services.retriever.embed_query", return_value=[0.0] * 384), \
             patch("services.retriever.query_chunks", return_value=self._make_chroma_result(docs, metas, dists)) as mock_qc, \
             patch("services.retriever.get_all_chunks") as mock_all:
            retriever_module.retrieve("q", doc_ids=["d1"])
            # get_all_chunks should NOT be called when doc_ids is set
            mock_all.assert_not_called()

    def test_optional_meta_fields(self):
        """page and section are optional; absence should not raise."""
        docs = ["chunk"]
        metas = [{"doc_id": "d1", "filename": "f.txt", "chunk_index": 0}]
        dists = [0.10]

        retriever_module._bm25_dirty = True
        with patch("services.retriever.embed_query", return_value=[0.0] * 384), \
             patch("services.retriever.query_chunks", return_value=self._make_chroma_result(docs, metas, dists)), \
             patch("services.retriever.get_all_chunks", return_value=[]):
            citations, _ = retriever_module.retrieve("q")

        assert citations[0].page is None
        assert citations[0].section is None
