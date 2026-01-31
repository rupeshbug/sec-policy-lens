import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def embed_query(text: str) -> list[float]:
    """
    Generate embedding using HuggingFace Inference API.
    Returns 384-dim vector compatible with Qdrant.
    """

    embedding = client.feature_extraction(
        text,
        model=MODEL_ID,
    )

    return embedding
