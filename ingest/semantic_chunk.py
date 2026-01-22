import json
from pathlib import Path
from typing import List, Dict

# paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
ALIGNED_DIR = DATA_DIR / "aligned"
CHUNKS_DIR = DATA_DIR / "chunks"

CHUNKS_DIR.mkdir(exist_ok=True)


# Paragraph-aware semantic chunking:
# We target ~1500 characters per chunk to balance retrieval precision and context,
# allow a small soft overflow to preserve paragraph continuity, and use paragraph-level
# overlap to maintain semantic coherence across adjacent chunks in long regulatory text. 
MAX_CHARS = 1500
SOFT_OVERFLOW = 200
HARD_LIMIT = MAX_CHARS + SOFT_OVERFLOW
OVERLAP_PARAGRAPHS = 1

# helper funcitons

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_paragraphs(paragraphs: List[str]) -> List[str]:
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current and current_len + para_len > HARD_LIMIT:
            chunks.append("\n\n".join(current))

            # paragraph-level overlap
            current = current[-OVERLAP_PARAGRAPHS:]
            current_len = sum(len(p) for p in current)

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# chunk the document
def chunk_document(aligned_file: str, output_file: str):
    aligned_sections = load_json(ALIGNED_DIR / aligned_file)

    all_chunks = []

    for sec in aligned_sections:
        paragraphs = split_into_paragraphs(sec["text"])
        chunks = chunk_paragraphs(paragraphs)

        print(
            f"[SECTION] {sec['section_id']} → "
            f"{len(chunks)} chunks"
        )

        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "document_id": sec["document_id"],
                "version": sec["version"],
                "section_id": sec["section_id"],
                "section_path": sec["section_path"],
                "title": sec["title"],
                "chunk_index": idx,
                "text": chunk_text
            })

    output_path = CHUNKS_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"[DONE] Saved {len(all_chunks)} chunks → {output_path}")

# entry
if __name__ == "__main__":
    chunk_document(
        aligned_file="2022_proposed_sections.json",
        output_file="2022_proposed_chunks.json"
    )

    chunk_document(
        aligned_file="2024_final_sections.json",
        output_file="2024_final_chunks.json"
    )
