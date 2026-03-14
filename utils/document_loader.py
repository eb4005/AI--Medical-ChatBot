"""
utils/document_loader.py
Loads and extracts text from uploaded PDF documents.
"""

import logging
import tempfile
import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf_documents(uploaded_files: list) -> List[Document]:
    """
    Extract text from one or more Streamlit UploadedFile objects (PDFs).

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        A flat list of LangChain Document objects (one per PDF page).

    Raises:
        RuntimeError: If no documents could be loaded from the provided files.
    """
    all_documents: List[Document] = []

    for uploaded_file in uploaded_files:
        try:
            # Write the in-memory bytes to a temp file so PyPDFLoader can read it
            suffix = os.path.splitext(uploaded_file.name)[-1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # Tag each page with the original filename for traceability
            for doc in documents:
                doc.metadata["source_file"] = uploaded_file.name

            all_documents.extend(documents)
            logger.info(
                "Loaded %d pages from '%s'.", len(documents), uploaded_file.name
            )

        except Exception as exc:
            logger.error("Failed to load '%s': %s", uploaded_file.name, exc)

        finally:
            # Always clean up the temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if not all_documents:
        raise RuntimeError(
            "No documents could be loaded. Please upload valid PDF files."
        )

    return all_documents
