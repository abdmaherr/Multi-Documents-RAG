import asyncio

from rank_bm25 import BM25Okapi

from db.chroma_client import query_chunks, get_all_chunks
from models.schemas import SourceCitation
from services.embedder import embed_query

NO_ANSWER_THRESHOLD = 0.85  # ChromaDB cosine distance: lower = more similar
CHUNK_RELEVANCE_THRESHOLD = 0.80  # Per-chunk filter: drop chunks with distance above this

# Hybrid scoring weights: α for vector similarity, β for BM25 keyword relevance
ALPHA = 0.5  # vector weight
BETA = 0.5   # BM25 weight

# BM25 cache — invalidated whenever documents are mutated
_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict] | None = None
_bm25_corpus_by_id: dict[str, dict] | None = None
_bm25_dirty = True


def invalidate_bm25_cache() -> None:
    global _bm25_dirty
    _bm25_dirty = True


def _get_bm25() -> tuple[BM25Okapi | None, list[dict], dict[str, dict]]:
    global _bm25_index, _bm25_corpus, _bm25_corpus_by_id, _bm25_dirty
    if _bm25_dirty or _bm25_index is None:
        corpus = get_all_chunks()
        if not corpus:
            return None, [], {}
        _bm25_corpus = corpus
        _bm25_corpus_by_id = {c["id"]: c for c in corpus}
        tokenized = [c["text"].lower().split() for c in corpus]
        _bm25_index = BM25Okapi(tokenized)
        _bm25_dirty = False
    return _bm25_index, _bm25_corpus or [], _bm25_corpus_by_id or {}


def _normalize_scores(scored: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scored:
        return {}
    vals = [s for _, s in scored]
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng == 0:
        return {cid: 1.0 for cid, _ in scored}
    return {cid: (s - lo) / rng for cid, s in scored}


def _hybrid_merge(
    vector_scores: list[tuple[str, float]],
    bm25_scores: list[tuple[str, float]],
) -> dict[str, float]:
    """Weighted combination of normalized vector and BM25 scores."""
    norm_vec = _normalize_scores(vector_scores)
    norm_bm25 = _normalize_scores(bm25_scores)

    all_ids = set(norm_vec) | set(norm_bm25)
    combined: dict[str, float] = {}
    for cid in all_ids:
        v = norm_vec.get(cid, 0.0)
        b = norm_bm25.get(cid, 0.0)
        combined[cid] = ALPHA * v + BETA * b
    return combined


def retrieve(
    question: str,
    n_results: int = 5,
    doc_ids: list[str] | None = None,
) -> tuple[list[SourceCitation], bool]:
    """Hybrid retrieve: vector + BM25 via weighted score combination.

    final_score = α * norm_vector + β * norm_bm25

    When doc_ids filter is active (compare mode), falls back to vector-only
    because the BM25 index covers all documents.
    """
    fetch_n = n_results * 2

    query_embedding = embed_query(question)
    results = query_chunks(query_embedding, n_results=fetch_n, doc_ids=doc_ids)

    if not results["documents"] or not results["documents"][0]:
        return [], False

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # Map chunk_id -> data for later lookup
    chunk_map: dict[str, dict] = {}
    vector_scored: list[tuple[str, float]] = []
    for doc_text, meta, dist in zip(docs, metas, dists):
        cid = f"{meta['doc_id']}_chunk_{meta['chunk_index']}"
        chunk_map[cid] = {"text": doc_text, "meta": meta, "distance": dist}
        vector_scored.append((cid, 1 - dist))  # similarity = 1 - distance

    # BM25 (skip when filtering by doc_ids — index spans all docs)
    bm25_scored: list[tuple[str, float]] = []
    if not doc_ids:
        bm25_index, bm25_corpus, corpus_by_id = _get_bm25()
        if bm25_index and bm25_corpus:
            tokenized_q = question.lower().split()
            raw_scores = bm25_index.get_scores(tokenized_q)
            ranked = sorted(
                zip([c["id"] for c in bm25_corpus], raw_scores),
                key=lambda x: x[1],
                reverse=True,
            )[:fetch_n]
            for cid, score in ranked:
                bm25_scored.append((cid, score))
                if cid not in chunk_map:
                    match = corpus_by_id.get(cid)
                    if match:
                        chunk_map[cid] = {"text": match["text"], "meta": match["meta"], "distance": 1.0}

    # Combine scores
    if bm25_scored:
        hybrid_scores = _hybrid_merge(vector_scored, bm25_scored)
    else:
        hybrid_scores = {cid: s for cid, s in vector_scored}

    top_ids = sorted(hybrid_scores, key=lambda x: hybrid_scores[x], reverse=True)[:n_results]

    citations: list[SourceCitation] = []
    best_distance = 1.0
    for cid in top_ids:
        data = chunk_map.get(cid)
        if not data:
            continue
        dist = data["distance"]
        if dist > CHUNK_RELEVANCE_THRESHOLD:
            continue
        best_distance = min(best_distance, dist)
        citations.append(SourceCitation(
            document=data["meta"]["filename"],
            chunk_text=data["text"],
            score=round(hybrid_scores.get(cid, 1 - dist), 4),
            page=data["meta"].get("page"),
            section=data["meta"].get("section"),
        ))

    has_relevant = best_distance < NO_ANSWER_THRESHOLD
    return citations, has_relevant


async def retrieve_multi_query(
    question: str,
    n_results: int = 5,
) -> tuple[list[SourceCitation], bool]:
    """Multi-query retrieval: generate query variants, retrieve for each, merge.

    Deduplicates by (document, chunk prefix) keeping the highest-scoring copy.
    """
    from services.generator import generate_query_variants

    variants = await generate_query_variants(question)
    queries = [question] + variants

    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        *(loop.run_in_executor(None, retrieve, q, n_results) for q in queries)
    )

    seen: dict[str, SourceCitation] = {}
    has_relevant = False

    for cits, relevant in results:
        if relevant:
            has_relevant = True
        for c in cits:
            key = f"{c.document}::{c.chunk_text[:80]}"
            if key not in seen or c.score > seen[key].score:
                seen[key] = c

    merged = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:n_results]
    return merged, has_relevant
