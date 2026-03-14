"""
models/llm.py
Initialises and returns the ChatGroq LLM.
"""

import logging
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


def get_chatgroq_model(api_key: str, model_name: str = "llama-3.1-8b-instant") -> ChatGroq:
    """
    Initialize and return a ChatGroq model instance.

    Args:
        api_key:    Groq API key (injected at call-site from config or session state).
        model_name: One of the supported Groq model identifiers.

    Returns:
        A configured ChatGroq instance.

    Raises:
        RuntimeError: If the model cannot be initialized.
    """
    try:
        if not api_key:
            raise ValueError("Groq API key is empty. Please enter it in the sidebar.")

        model = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0.3,
            max_tokens=2048,
        )
        logger.info("ChatGroq model '%s' initialized successfully.", model_name)
        return model

    except Exception as exc:
        logger.error("Failed to initialize Groq model: %s", exc)
        raise RuntimeError(f"Failed to initialize Groq model: {exc}") from exc
