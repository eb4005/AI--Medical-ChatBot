"""
models/embeddings.py
Initialises and returns the HuggingFace embedding model.
No API key required — the model runs locally via sentence-transformers.
"""

import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and return a HuggingFaceEmbeddings instance.

    The model is downloaded on first run and cached locally by
    sentence-transformers (~90 MB). Subsequent calls are instant.

    Returns:
        A configured HuggingFaceEmbeddings instance.

    Raises:
        RuntimeError: If the embedding model cannot be loaded.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model '%s' loaded successfully.", EMBEDDING_MODEL_NAME)
        return embeddings

    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        raise RuntimeError(f"Failed to load embedding model: {exc}") from exc
