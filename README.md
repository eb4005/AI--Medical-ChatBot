# 🩺 MediAssist AI - Medical Report Interpreter

MediAssist AI is a modern, ChatGPT-style medical assistant built to help users seamlessly interpret their clinical reports and answer medical queries. Using Retrieval-Augmented Generation (RAG) and leveraging advanced LLMs via Groq, the application analyzes uploaded patient PDFs (blood tests, pathology reports, visit summaries) and provides structured, easy-to-understand insights.

**Disclaimer:** This tool is for decision-support and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 🔗 Live Demo

👉 **[Try MediAssist AI on Streamlit Cloud](https://ai--medical-chatbot-h9jqzpzsfxfmmcbgt7sraf.streamlit.app/)**

---

## 🌟 Features

*   **Beautiful Modern UI:** A premium, minimalist light theme inspired by ChatGPT with a stable, fixed-bottom chat composer.
*   **Intelligent Knowledge Base (RAG):** Upload multiple medical PDFs via the convenient "＋" menu. The app indexes documents locally and references them for accurate answers.
*   **Web Search Fallback:** When your uploaded reports lack sufficient context, MediAssist seamlessly falls back to Tavily Web Search to find the latest medical information.
*   **Conversational Assistant Modes:**
    *   **Standard Chat:** Ask open-ended questions about your attached reports.
    *   **Diagnostic Analysis:** Provide patient features/symptoms to receive a structured assessment and concern score.
    *   **Summarise Jargon:** Paste complex medical text to have it explained in simple, accessible language.
*   **Adjustable Detail Level:** Switch between "Concise" and "Detailed" response lengths.
*   **Powered by Groq:** Ultra-fast inference with support for state-of-the-art models (Llama 3, Mixtral, Gemma).

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.10+
*   API Keys for **Groq** and **Tavily**.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/eb4005/AI--Medical-ChatBot.git
    cd AI--Medical-ChatBot
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure environment variables:**
    *   Rename `.env.example` to `.env`.
    *   Add your API keys to the `.env` file:
        ```env
        GROQ_API_KEY="your_groq_api_key_here"
        TAVILY_API_KEY="your_tavily_api_key_here"
        DEFAULT_GROQ_MODEL="llama3-70b-8192"
        ```

### Running the App

Start the Streamlit development server:

```bash
streamlit run app.py
```

The app will open automatically in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure

*   `app.py`: Main Streamlit application and UI layout.
*   `config/`: Configuration logic for environment variables and model selection.
*   `models/`: LLM and embedding model initializations (Groq, HuggingFace BGE).
*   `utils/`: Core utilities for document loading (PyPDF2), text splitting, vector store management (FAISS), and prompt templates.
*   `test_rag.py`: A standalone script to verify the RAG pipeline.

---

## 🛠️ Built With

*   [Streamlit](https://streamlit.io/) - The web framework used.
*   [LangChain](https://python.langchain.com/) - Framework for developing LLM applications.
*   [Groq API](https://groq.com/) - Blazing-fast LLM inference.
*   [FAISS](https://faiss.ai/) - Vector search and clustering.
*   [Tavily](https://tavily.com/) - AI-optimized web search.