import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion

# --------------------
# CONFIG
# --------------------
COLLECTION_NAME = "regulens"
TOP_K = 5

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPLADE_MODEL_ID = "naver/splade-cocondenser-ensembledistil"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------
# LOAD MODELS
# --------------------
print("[INFO] Loading models...")

dense_model = SentenceTransformer(DENSE_MODEL_NAME, device=DEVICE)

splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_ID)
splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_ID).to(DEVICE)
splade_model.eval()

client = QdrantClient(url="http://localhost:6333")


# --------------------
# SPLADE QUERY VECTOR
# --------------------
@torch.no_grad()
def compute_splade_query(text: str):
    tokens = splade_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    output = splade_model(**tokens)
    logits = output.logits

    relu_log = torch.log1p(torch.relu(logits))
    weighted = relu_log * tokens.attention_mask.unsqueeze(-1)

    vec, _ = torch.max(weighted, dim=1)
    vec = vec.squeeze()

    nonzero = vec.nonzero(as_tuple=False).squeeze().cpu()
    values = vec[nonzero].cpu()

    return {
        "indices": nonzero.tolist(),
        "values": values.tolist()
    }


# --------------------
# HYBRID SEARCH
# --------------------
def hybrid_search(query: str):
    print("\n[QUERY]")
    print(query)

    dense_query = dense_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    sparse_query = compute_splade_query(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                using="dense",
                query=dense_query,
                limit=TOP_K,
            ),
            Prefetch(
                using="sparse",
                query=sparse_query,
                limit=TOP_K,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=TOP_K,
    )

    print("\n[HYBRID RESULTS]\n")

    for rank, point in enumerate(response.points, start=1):
        payload = point.payload

        print(f"--- Rank {rank} | score={point.score:.4f} ---")
        print(f"Doc: {payload['document_id']} ({payload['version']})")
        print(f"Section: {payload['section_id']} | {payload['title']}")
        print("Text preview:")
        print(payload["text"][:500])
        print()


# --------------------
# ENTRY POINT
# --------------------
if __name__ == "__main__":
    hybrid_search(
        "Why does the SEC believe additional climate-related disclosure "
        "requirements are necessary for investors?"
    )
