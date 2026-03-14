"""
utils/retriever.py
Queries the FAISS index and returns the top-K relevant document chunks.
"""

import logging
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config.config import RETRIEVER_TOP_K

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    query: str,
    vector_store: FAISS,
    top_k: int = RETRIEVER_TOP_K,
) -> List[Document]:
    """
    Retrieve the most semantically similar document chunks for a given query.

    Args:
        query:        The user's question or search string.
        vector_store: The populated FAISS vector store.
        top_k:        Number of chunks to return (default from config).

    Returns:
        A list of Document objects ordered by descending relevance.

    Raises:
        RuntimeError: If retrieval fails.
    """
    try:
        if not query.strip():
            raise ValueError("Query string must not be empty.")

        results: List[Document] = vector_store.similarity_search(
            query=query,
            k=top_k,
        )

        logger.info(
            "Retrieved %d chunk(s) for query: '%.60s...'",
            len(results),
            query,
        )
        return results

    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise RuntimeError(f"Retrieval failed: {exc}") from exc


def retrieve_with_scores(
    query: str,
    vector_store: FAISS,
    top_k: int = RETRIEVER_TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Retrieve chunks along with their similarity scores (useful for the
    Diagnostic Analyzer match-score feature).

    Returns:
        List of (Document, score) tuples; lower L2 distance = higher relevance.
    """
    try:
        results = vector_store.similarity_search_with_score(query=query, k=top_k)
        logger.info("Retrieved %d scored chunk(s).", len(results))
        return results

    except Exception as exc:
        logger.error("Scored retrieval failed: %s", exc)
        raise RuntimeError(f"Scored retrieval failed: {exc}") from exc
