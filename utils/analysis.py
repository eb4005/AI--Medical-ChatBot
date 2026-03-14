"""
utils/analysis.py
Orchestration logic for the Diagnostic Case Analyzer (Tab 2).

Flow:
  1. Accept the patient's extracted features as a string.
  2. Query the FAISS vector store for relevant clinical guideline chunks.
  3. Build the CASE_ANALYZER_PROMPT and invoke the LLM.
  4. Parse the response to extract the numeric Match Score for st.metric.
  5. Return the full report text and the extracted score.
"""

import logging
import re
from typing import Tuple, Optional

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from utils.retriever import retrieve_with_scores
from utils.prompts import CASE_ANALYZER_PROMPT
from config.config import RETRIEVER_TOP_K

logger = logging.getLogger(__name__)


def run_diagnostic_analysis(
    patient_features: str,
    vector_store: FAISS,
    llm: ChatGroq,
    top_k: int = RETRIEVER_TOP_K,
) -> Tuple[str, Optional[int]]:
    """
    Orchestrate the full diagnostic case analysis pipeline.

    Args:
        patient_features: Free-text description of the patient's acoustic /
                          diagnostic markers (entered by the clinician in Tab 2).
        vector_store:     Populated FAISS index of clinical guideline documents.
        llm:              Initialised ChatGroq model.
        top_k:            Number of guideline chunks to retrieve.

    Returns:
        A tuple of (report_markdown: str, match_score: int | None).
        match_score is None if it cannot be parsed from the LLM response.

    Raises:
        RuntimeError: If retrieval or LLM invocation fails.
    """
    try:
        # ── Step 1: Retrieve relevant guideline chunks ─────────────────────────
        scored_chunks = retrieve_with_scores(
            query=patient_features,
            vector_store=vector_store,
            top_k=top_k,
        )

        if not scored_chunks:
            raise ValueError(
                "No relevant medical reports found. "
                "Please upload medical document(s) first."
            )

        # Concatenate chunk texts into a single context block
        clinical_guidelines = "\n\n---\n\n".join(
            doc.page_content for doc, _score in scored_chunks
        )

        # ── Step 2: Build the prompt ───────────────────────────────────────────
        filled_prompt = CASE_ANALYZER_PROMPT.format(
            clinical_guidelines=clinical_guidelines,
            patient_features=patient_features,
        )

        # ── Step 3: Invoke the LLM ────────────────────────────────────────────
        response = llm.invoke([HumanMessage(content=filled_prompt)])
        report_text: str = response.content

        # ── Step 4: Parse Match Score from the report ─────────────────────────
        match_score = _extract_match_score(report_text)

        logger.info(
            "Diagnostic analysis complete. Match score: %s", match_score
        )
        return report_text, match_score

    except Exception as exc:
        logger.error("Diagnostic analysis failed: %s", exc)
        raise RuntimeError(f"Diagnostic analysis failed: {exc}") from exc


def _extract_match_score(report_text: str) -> Optional[int]:
    """
    Parse the numeric match score from the LLM's structured report.

    Looks for patterns like:
        **Concern Level Score:** 72 / 100
        Concern Level Score: 72/100
        **Score:** 72

    Returns:
        Integer score 0–100, or None if not found.
    """
    try:
        pattern = r"\*{0,2}(?:Concern Level )?Score\*{0,2}[:\s]+(\d{1,3})\s*(?:/\s*100)?"
        match = re.search(pattern, report_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return max(0, min(100, score))   # clamp to [0, 100]
        return None
    except Exception:
        return None
