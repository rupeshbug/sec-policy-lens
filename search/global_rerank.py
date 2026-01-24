from typing import List
from ingest.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker()


def global_rerank(query: str, candidates: List, top_k: int = 8):
    """
    Global reranking using the original user query.
    """

    if not candidates:
        return []

    passages = [c.payload.get("text", "") for c in candidates]

    scores = reranker.rerank(query, passages)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [c for c, _ in scored[:top_k]]
