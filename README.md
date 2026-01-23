# ReguLens

ReguLens is a retrieval-augmented generation (RAG) system focused on U.S. SEC Climate‑Related Disclosure rules. It is designed to answer regulatory and investor‑facing questions using **only** official SEC source documents, with strong focus to prevent hallucination.

A core capability of ReguLens is its ability to analyze, explain, and contrast different versions of the same regulation, with particular emphasis on comparing:
- The **2022 SEC Climate‑Related Disclosure Proposed Rule**
- The **2024 SEC Climate‑Related Disclosure Final Rule**

The system prioritizes accuracy, traceability, and compliance-safe reasoning, with explicit safeguards to prevent hallucination or unsupported interpretation.

---

## What ReguLens Does

- Retrieves relevant regulatory passages using **hybrid search** (dense + sparse)
- Leverages version-aware metadata (document, year, section) for precise filtering
- Reranks results using a **cross‑encoder** for semantic precision
- Generates answers using a large language model constrained by retrieved context
- Prefers **final rules over proposed rules** when both are available
- Explicitly refuses to answer when context is insufficient

This allows ReguLens to support:

- Regulatory interpretation
- Comparison between proposed and finalized rules
- Compliance and legal research
- Investor-facing explanations grounded in official disclosures

---

## Key Capabilities

- **Hybrid Retrieval**: Combines dense semantic embeddings with sparse lexical matching to capture both meaning and exact regulatory language.
- **Chunk-Based Indexing**: SEC documents are chunked into semantically coherent sections to improve retrieval precision and reduce noise.
- **Metadata-Aware Search**: Each chunk is stored with structured metadata (document version, section, title), enabling fast filtering and version-sensitive retrieval.
- **RRF Fusion**: Reciprocal Rank Fusion (RRF) merges multiple retrieval signals into a robust candidate set.
- **Cross‑Encoder Reranking**: A transformer-based reranker refines results to ensure the most contextually relevant passages are used for generation.
- **Strict Grounding**: No external knowledge or assumptions
- **Source Awareness**: Tracks document version and section metadata

---

## Models Used

- Dense embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Sparse retrieval: `naver/splade-cocondenser-ensembledistil`
- Reranker: Sentence‑Transformers cross‑encoder
- Vector database: Qdrant
- LLM: Groq‑hosted `llama‑3.3‑70b‑versatile`

---
