import os
from typing import List

USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ):
        """
        Cross-encoder reranker for (query, passage) pairs.

        In production (USE_LOCAL_MODELS=false), this becomes a no-op
        and does NOT import sentence-transformers.
        """
        self.model = None

        if not USE_LOCAL_MODELS:
            return

        # ⬇️ import ONLY when explicitly enabled
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """
        Returns relevance scores aligned with passages order.
        """
        if not self.model:
            # prod-safe fallback: neutral scores
            return [0.0] * len(passages)

        pairs = [(query, passage) for passage in passages]
        scores = self.model.predict(pairs)
        return scores.tolist()
