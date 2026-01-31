from functools import lru_cache
import os

from groq import Groq
from qdrant_client import QdrantClient

from search.query_decomposition import QueryDecomposer


@lru_cache
def get_decomposer():
    return QueryDecomposer()


@lru_cache
def get_llm_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


@lru_cache
def get_qdrant():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )