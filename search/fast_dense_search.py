from qdrant_client.models import Filter, FieldCondition, MatchValue
from search.runtime import get_qdrant
from search.remote_embeddings import embed_query

COLLECTION_NAME = "regulens"
TOP_K = 5


def fast_dense_search(
    query: str,
    version_filter: str | None = None,
    top_k: int = TOP_K,
):
    """
    Latency-optimized dense-only retrieval.
    No SPLADE, no reranking.
    """
    
    client = get_qdrant()

    qdrant_filter = None
    if version_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="version",
                    match=MatchValue(value=version_filter)
                )
            ]
        )

    query_vector = embed_query(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        limit=top_k,
        with_payload=True,
        query_filter=qdrant_filter,
    )

    return response.points
