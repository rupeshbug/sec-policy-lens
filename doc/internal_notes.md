# ReguLens – Internal Project Notes

This document explains **what was built, why it was built, and the reasoning behind key design decisions**. It is intended as a long‑term memory aid and as a narrative to be used when explaining the project to others.

---

## 1. Problem Definition

Regulatory documents are:
- Long
- Highly structured
- Easy to misinterpret when summarized loosely

Generic LLMs often:
- Hallucinate regulatory intent
- Mix proposed and final rules
- Answer confidently without sufficient grounding

**Goal:** Build a system that answers regulatory questions *only* from official SEC text, with strong guarantees of correctness and traceability.

---

## 2. Why Retrieval‑Augmented Generation (RAG)

Pure LLM answers are unacceptable for compliance use cases.

RAG ensures:
- Answers are grounded in official documents
- The model cannot exceed its evidence
- Missing context is detected and acknowledged

---

## 3. Why Hybrid Search (Dense + Sparse)

### Dense Search (Embeddings)
- Captures semantic similarity
- Handles paraphrased questions
- Weak at exact regulatory phrasing

### Sparse Search (SPLADE)
- Preserves legal and regulatory terminology
- Handles exact keyword matching
- Weak at semantic generalization

### Why Both

SEC questions often mix:
- Conceptual phrasing ("why does the SEC believe…")
- Legal language ("disclosure requirements", "investors")

Hybrid search gives robustness across both dimensions.

---

## 4. Why Reciprocal Rank Fusion (RRF)

RRF:
- Does not require score normalization
- Is robust when retrievers disagree
- Is production‑proven in IR systems

This makes it ideal for combining dense and sparse retrievers.

---

## 5. Why Cross‑Encoder Reranking

Vector similarity ≠ answer relevance.

Reranking:
- Scores query–passage pairs jointly
- Dramatically improves top‑k quality
- Reduces irrelevant but semantically close chunks

This is critical before sending context to the LLM.

---

## 6. Why No LangChain / LlamaIndex

This was a **deliberate choice**.

Reasons:
- Full control over retrieval logic
- Easier debugging and inspection
- No hidden prompt manipulation
- Better suitability for regulated domains

Frameworks are helpful for speed, but risky for compliance‑grade systems.

---

## 7. Prompt Design Philosophy

The system prompt enforces:
- Scope limitation (SEC climate rules only)
- Preference for final rules (2024 > 2022)
- Explicit refusal when context is insufficient

The user prompt:
- Injects retrieved passages verbatim
- Forbids external knowledge
- Encourages coherence across sections

---

## 8. Observed Behavior (Important)

### Good Behavior
- Correctly prefers 2024 Final Rule
- Refuses to hallucinate differences when context is missing
- Combines sections coherently when evidence exists

### Example
Question:
"What are the specific changes in the 2024 version from the previous one?"

Result:
- Model explicitly stated that context was insufficient
- This is **correct behavior**, not a failure

---

## 9. Current Limitations

- No explicit diff‑aware retrieval yet
- Comparison questions require targeted chunking
- No automated evaluation metrics yet

These are **known and intentional tradeoffs** at this stage.

---

## 10. Next Planned Stages

1. **Evaluation**
   - RAGAS (faithfulness, context precision, recall)

2. **API Layer**
   - FastAPI
   - Stateless retrieval endpoint
   - Deterministic answer generation

3. **Advanced Retrieval**
   - Version‑aware chunk pairing
   - Explicit proposal vs final comparison chains

---

## 11. Key Takeaway

ReguLens is not a demo RAG.

It is a **compliance‑grade retrieval and reasoning system** built with:
- Intentional constraints
- Auditable logic
- Production‑ready architecture

The system’s refusal to answer is a feature, not a bug.

