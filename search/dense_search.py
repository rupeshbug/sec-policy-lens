from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "regulens"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def main():
    client = QdrantClient(url="http://localhost:6333")
    model = SentenceTransformer(MODEL_NAME)

    query = (
        "Why does the SEC believe additional climate-related disclosure "
        "requirements are necessary for investors?"
    )

    print("\n[QUERY]")
    print(query)

    # Embed query
    query_vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # query call
    response = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        using = "dense",     
        limit = TOP_K,
        with_payload = True
    )

    print("\n[RESULTS]\n")

    for rank, point in enumerate(response.points, start=1):
        payload = point.payload

        print(f"--- Rank {rank} | score={point.score:.4f} ---")
        print(f"Doc: {payload['document_id']} ({payload['version']})")
        print(f"Section: {payload['section_id']} | {payload['title']}")
        print("Text preview:")
        print(payload["text"][:500])
        print()


if __name__ == "__main__":
    main()