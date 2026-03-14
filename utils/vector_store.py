"""
utils/vector_store.py
Builds and returns a FAISS vector store from document chunks.
The embedding model is provided by the caller (models/embeddings.py).
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def build_vector_store(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Embed document chunks and index them in an in-memory FAISS vector store.

    Args:
        chunks:          Chunked Document objects from the text splitter.
        embedding_model: Initialized HuggingFaceEmbeddings instance.

    Returns:
        A FAISS vector store ready for similarity search.

    Raises:
        RuntimeError: If the vector store cannot be built.
    """
    try:
        if not chunks:
            raise ValueError("Cannot build vector store from an empty chunk list.")

        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )

        logger.info(
            "FAISS vector store built with %d vectors.", len(chunks)
        )
        return vector_store

    except Exception as exc:
        logger.error("Failed to build vector store: %s", exc)
        raise RuntimeError(f"Failed to build vector store: {exc}") from exc
