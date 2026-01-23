import json
import hashlib
from pathlib import Path
from typing import List
import uuid

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


COLLECTION_NAME = "regulens"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CHUNKS_DIR = DATA_DIR / "chunks"


def load_chunks(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

def stable_point_id(doc_id: str, section_id: str, chunk_index: int) -> str:
    """
    Create a stable deterministic ID so re-upserts overwrite correctly.
    """
    raw = f"{doc_id}|{section_id}|{chunk_index}"
    return str(uuid.uuid5(NAMESPACE, raw))


def ingest_chunks(chunks_file: str):
    print(f"\n=== Ingesting {chunks_file} ===")

    # Load data
    chunks = load_chunks(CHUNKS_DIR / chunks_file)
    print(f"[INFO] Loaded {len(chunks)} chunks")

    # initialize model
    print("[INFO] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    client = QdrantClient(url="http://localhost:6333")
    
    # batch encode for speed boost
    print(f"[EMBED] Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk["text"] for chunk in chunks]
    
    embeddings = model.encode(texts, batch_size = 32, show_progress_bar = True, normalize_embeddings = True)

    points = []

    for i, chunk in enumerate(chunks):
        point_id = stable_point_id(
            chunk["document_id"],
            chunk["section_id"],
            chunk["chunk_index"]
        )

        payload = {
            "document_id": chunk["document_id"],
            "version": chunk["version"],
            "section_id": chunk["section_id"],
            "section_path": chunk["section_path"],
            "title": chunk["title"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"]
        }

        points.append(
            PointStruct(
                id = point_id,
                vector = {"dense": embeddings[i].tolist()},
                payload = payload
            )
        )

    print(f"[INFO] Upserting {len(points)} points into Qdrant...")

    client.upsert(
        collection_name = COLLECTION_NAME,
        points = points
    )

    print("[DONE] Upsert complete.")

    # validate 
    count = client.count(collection_name=COLLECTION_NAME, exact=True)
    print(f"[VERIFY] Collection now contains {count.count} points")


if __name__ == "__main__":
    ingest_chunks("2022_proposed_chunks.json")
    ingest_chunks("2024_final_chunks.json")
