import json
from pathlib import Path
from typing import List, Dict

# paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
STRUCTURE_DIR = DATA_DIR / "structure"
EXTRACTED_DIR = DATA_DIR / "extracted"
ALIGNED_DIR = DATA_DIR / "aligned"

ALIGNED_DIR.mkdir(exist_ok=True)

# helper functions

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_sections(
    sections: List[Dict],
    parent_path: List[str] = None
) -> List[Dict]:
    """
    Recursively flatten hierarchical sections into a linear list.
    Preserves section path.
    """
    if parent_path is None:
        parent_path = []

    flat = []

    for sec in sections:
        section_id = sec["id"]
        section_path = parent_path + [section_id]

        flat.append({
            "section_id": section_id,
            "section_path": section_path,
            "title": sec.get("title", ""),
            "page_start": sec.get("page"),
        })

        if "children" in sec:
            flat.extend(
                flatten_sections(
                    sec["children"],
                    parent_path=section_path
                )
            )

    return flat


def infer_page_ranges(flat_sections: List[Dict], max_page: int):
    """
    Given ordered sections with page_start,
    infer page_end from the next section.
    """
    for i, sec in enumerate(flat_sections):
        start = sec["page_start"]

        if i < len(flat_sections) - 1:
            next_start = flat_sections[i + 1]["page_start"]
            sec["page_end"] = next_start - 1
        else:
            sec["page_end"] = max_page

    return flat_sections


def merge_page_text(pages: Dict[int, str], start: int, end: int) -> str:
    texts = []
    for p in range(start, end + 1):
        if p in pages:
            texts.append(pages[p])
    return "\n\n".join(texts)


# main pipeline

def process_document(structure_file: str, pages_file: str, version: str):
    print(f"\n=== Processing {version} ===")

    structure = load_json(STRUCTURE_DIR / structure_file)
    raw_pages = load_json(EXTRACTED_DIR / pages_file)
    pages_data = raw_pages["pages"]

    # pages_data: [{page_number, text}]
    pages = {p["page_number"]: p["text"] for p in pages_data}
    max_page = max(pages.keys())

    print(f"[INFO] Loaded {len(pages)} pages (max page={max_page})")

    # Flatten structure
    flat_sections = flatten_sections(structure["sections"])
    print(f"[INFO] Flattened into {len(flat_sections)} sections")

    # Sort by page_start (important)
    flat_sections.sort(key=lambda x: x["page_start"])

    # Infer page ranges
    flat_sections = infer_page_ranges(flat_sections, max_page)

    aligned_sections = []

    for sec in flat_sections:
        text = merge_page_text(
            pages,
            sec["page_start"],
            sec["page_end"]
        )

        char_count = len(text)

        print(
            f"[SECTION] {sec['section_id']:8s} "
            f"| pages {sec['page_start']:>4}-{sec['page_end']:<4} "
            f"| chars={char_count}"
        )

        aligned_sections.append({
            "document_id": structure["document_id"],
            "version": structure["version"],
            "section_id": sec["section_id"],
            "section_path": sec["section_path"],
            "title": sec["title"],
            "page_start": sec["page_start"],
            "page_end": sec["page_end"],
            "text": text,
        })

    output_path = ALIGNED_DIR / f"{version}_sections.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aligned_sections, f, indent=2)

    print(f"[DONE] Saved {len(aligned_sections)} sections → {output_path}")


# entry point

if __name__ == "__main__":
    process_document(
        structure_file="sec_2022.json",
        pages_file="sec_2022_pages.json",
        version="2022_proposed"
    )

    process_document(
        structure_file="sec_2024.json",
        pages_file="sec_2024_pages.json",
        version="2024_final"
    )
