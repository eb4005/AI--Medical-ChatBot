"""
config/config.py
Centralized configuration — loads all API keys and RAG constants.
Keys are NEVER hardcoded; they are read from environment variables or
Streamlit secrets so this file is safe to commit to version control.
"""

import os
from dotenv import load_dotenv

# Load a local .env file when running outside Streamlit Cloud
load_dotenv()


def get_groq_api_key() -> str:
    """Return Groq API key from environment or Streamlit secrets."""
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


def get_tavily_api_key() -> str:
    """Return Tavily API key from environment or Streamlit secrets."""
    try:
        import streamlit as st
        return st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))
    except Exception:
        return os.getenv("TAVILY_API_KEY", "")


# ── RAG constants ──────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 800          # characters per text chunk
CHUNK_OVERLAP: int = 150       # overlap between consecutive chunks
RETRIEVER_TOP_K: int = 4       # number of chunks to retrieve per query

# ── Embedding model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── LLM defaults ──────────────────────────────────────────────────────────────
DEFAULT_GROQ_MODEL: str = "llama-3.1-8b-instant"
AVAILABLE_GROQ_MODELS: list[str] = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]
