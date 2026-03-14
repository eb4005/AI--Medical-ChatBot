"""
utils/text_splitter.py
Splits raw document pages into overlapping chunks for embedding.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of LangChain Documents into smaller, overlapping chunks.

    Uses RecursiveCharacterTextSplitter which respects natural language
    boundaries (paragraphs → sentences → words) before hard-cutting.

    Args:
        documents: Raw Document objects from the document loader.

    Returns:
        A list of chunked Document objects ready for embedding.

    Raises:
        RuntimeError: If splitting fails or produces no chunks.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Splitting produced zero chunks. Check document content.")

        logger.info(
            "Split %d document(s) into %d chunk(s) "
            "(chunk_size=%d, overlap=%d).",
            len(documents),
            len(chunks),
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )
        return chunks

    except Exception as exc:
        logger.error("Text splitting failed: %s", exc)
        raise RuntimeError(f"Text splitting failed: {exc}") from exc
