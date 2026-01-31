from typing import Dict

def answer_regulatory_question(
    query: str,
    version: str | None = None,
    mode: str = "fast",  # fast | full
) -> Dict:
    """
    Production entry point.

    fast:
      - dense-only retrieval
      - low latency
    full:
      - hybrid retrieval
      - query decomposition
      - reranking
    """

    if mode == "full":
        from search.rag_answer import answer_query
        return answer_query(
            query=query,
            version_filter=version,
            decompose=True,
            global_rerank_enabled=True,
        )

    from search.rag_answer_fast import answer_query_fast
    return answer_query_fast(
        query=query,
        version_filter=version,
    )
