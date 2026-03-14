"""
app.py  —  MediAssist AI
Medical Report Interpretation Assistant
Fixed-bottom composer + scrollable conversation, ChatGPT-style.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from langchain_core.messages import HumanMessage

from config.config import (
    get_groq_api_key,
    get_tavily_api_key,
    AVAILABLE_GROQ_MODELS,
    DEFAULT_GROQ_MODEL,
)
from models.llm import get_chatgroq_model
from models.embeddings import get_embedding_model
from utils.document_loader import load_pdf_documents
from utils.text_splitter import split_documents
from utils.vector_store import build_vector_store
from utils.retriever import retrieve_relevant_chunks
from utils.web_search import web_search
from utils.prompts import CONCISE_PROMPT, DETAILED_PROMPT, SUMMARIZER_PROMPT
from utils.analysis import run_diagnostic_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM  (all tokens in :root, nothing hard-coded elsewhere)
# ─────────────────────────────────────────────────────────────────────────────
CHAT_HEIGHT = 520   # px for the scrollable message container

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Design tokens ───────────────────────────────────── */
:root {{
    --bg:      #F7F8FA;
    --surface: #FFFFFF;
    --surf2:   #F3F4F6;
    --txt:     #111827;
    --txt2:    #6B7280;
    --border:  #E5E7EB;
    --accent:  #10a37f;
    --purple:  #7C3AED;
    --radius:  14px;
    --rad-sm:  8px;
    --shadow:  0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
    --shadow-lg: 0 4px 24px rgba(0,0,0,.09);
}}

/* ── Reset / base ────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--txt) !important;
}}
.stApp {{ background-color: var(--bg) !important; }}
[data-testid="stSidebar"]   {{ display: none !important; }}
[data-testid="stHeader"]    {{ background: transparent !important; }}
footer, #MainMenu           {{ display: none !important; }}

.block-container {{
    max-width: 1180px !important;
    padding: 1rem 1.75rem 2rem !important;  /* bottom pad = composer height handled by CSS below */
}}

/* ─────────────────────────────────────────────────────
   FIXED BOTTOM COMPOSER  (Streamlit wraps st.chat_input
   in [data-testid="stBottom"] — we pin that to viewport)
───────────────────────────────────────────────────── */
[data-testid="stBottom"] {{
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 9000 !important;
    background: linear-gradient(to top, var(--bg) 80%, transparent) !important;
    padding: .75rem 1.75rem 1.1rem !important;
    display: flex !important;
    justify-content: center !important;
}}
[data-testid="stBottom"] > div {{
    max-width: 780px !important;
    width: 100% !important;
}}

/* styled pill composer */
div[data-testid="stChatInput"] {{
    padding: 0 !important;
}}

div[data-testid="stChatInput"] textarea {{
    padding-left: 12px !important; /* Back to normal padding */
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 30px !important;
    box-shadow: var(--shadow-lg) !important;
    color: var(--txt) !important;
    min-height: 52px !important;
}}

/* ── + button style (adjacent to composer) ──────────── */
[data-testid="stBottom"] [data-testid="stPopover"] button {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    width: 44px !important;
    height: 48px !important;
    font-size: 1.6rem !important;
    color: var(--txt2) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

/* Hide popover chevron and text for ChatGPT icon look */
[data-testid="stBottom"] [data-testid="stPopover"] button svg,
[data-testid="stBottom"] [data-testid="stPopover"] button div[data-testid="stMarkdownContainer"] p,
[data-testid="stBottom"] [data-testid="stPopover"] button span,
[data-testid="stBottom"] [data-testid="stPopover"] svg {{
    display: none !important;
}}

[data-testid="stBottom"] [data-testid="stPopover"] button:hover {{
    color: var(--txt) !important;
    background: var(--surf2) !important;
    border-radius: 50% !important;
}}


/* ── Header ──────────────────────────────────────────── */
.app-header {{
    padding: .4rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}}
.app-header h1 {{ font-size:1.35rem; font-weight:700; margin:0 0 .15rem; color:var(--txt) !important; }}
.app-header p  {{ font-size:.88rem; color:var(--txt2) !important; margin:0 0 .5rem; }}
.badge {{
    display:inline-flex; align-items:center; gap:.3rem;
    background:#ECFDF5; color:#065F46;
    border:1px solid #A7F3D0; border-radius:999px;
    padding:.18rem .65rem; font-size:.73rem; font-weight:500;
}}
.badge-off {{ background:#F3F4F6; color:var(--txt2) !important; border-color:var(--border); }}

/* ── Message scroll container ────────────────────────── */
/* give extra bottom breathing room so last message clears composer */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    padding-bottom: 0 !important;
}}

/* ── Chat messages ───────────────────────────────────── */
.stChatMessage {{ background: transparent !important; border:none !important; padding:.6rem 0 !important; }}
[data-testid="chatAvatarIcon-assistant"] {{ background: var(--accent) !important; border-radius:50%; }}
[data-testid="chatAvatarIcon-user"]      {{ background: var(--purple) !important; border-radius:50%; }}
.stChatMessage [data-testid="stMarkdownContainer"] p {{ color: var(--txt) !important; }}

/* ── Suggestion chips ────────────────────────────────── */
.chips {{ display:flex; flex-wrap:wrap; gap:.55rem; margin-top:.6rem; }}
.chip {{
    background:var(--surface); border:1px solid var(--border);
    border-radius:999px; padding:.4rem 1rem;
    font-size:.83rem; color:var(--txt);
    cursor:pointer; transition:all .15s;
}}
.chip:hover {{ background:var(--surf2); border-color:#C4B5FD; color:var(--purple); }}
.empty-state {{
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; min-height:340px; text-align:center; padding:2rem;
}}
.empty-state .icon {{ font-size:2.8rem; margin-bottom:.75rem; }}
.empty-state h2 {{ font-size:1.35rem; font-weight:600; margin-bottom:.45rem; color:var(--txt) !important; }}
.empty-state p  {{ color:var(--txt2) !important; font-size:.92rem; max-width:360px; margin:0 auto .65rem; }}

/* ── Cards / alert boxes ─────────────────────────────── */
.card {{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); box-shadow:var(--shadow); padding:1.1rem 1.3rem; margin-bottom:.85rem; }}
.card-title {{ font-size:.78rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--txt2) !important; margin-bottom:.65rem; }}
.abox {{ border-radius:var(--rad-sm); padding:.65rem .9rem; font-size:.85rem; margin:.35rem 0; }}
.abox-ok   {{ background:#ECFDF5; border-left:3px solid #10B981; color:#065F46 !important; }}
.abox-warn {{ background:#FFFBEB; border-left:3px solid #F59E0B; color:#92400E !important; }}
.abox-info {{ background:#EFF6FF; border-left:3px solid #3B82F6; color:#1E40AF !important; }}

/* ── Expander (Settings accordion) ──────────────────── */
[data-testid="stExpander"] {{
    border:1px solid var(--border) !important; border-radius:var(--radius) !important;
    background:var(--surface) !important; box-shadow:var(--shadow) !important; overflow:hidden !important;
}}
[data-testid="stExpander"] summary {{
    color:var(--txt) !important; font-weight:600 !important;
    font-size:.88rem !important; padding:.8rem 1rem !important;
    background:var(--surface) !important;
}}
[data-testid="stExpander"] summary:hover {{ background:var(--surf2) !important; }}
[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {{ padding:.9rem 1rem 1rem !important; }}

/* ── Native Streamlit widget polish ─────────────────── */
[data-baseweb="select"] > div {{
    background:var(--surface) !important; border-color:var(--border) !important;
    border-radius:var(--rad-sm) !important;
}}
[data-baseweb="select"] span {{ color:var(--txt) !important; }}
[data-baseweb="popover"] ul {{ background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:var(--rad-sm) !important; box-shadow:var(--shadow-lg) !important; }}
[data-baseweb="popover"] li {{ color:var(--txt) !important; background:var(--surface) !important; }}
[data-baseweb="popover"] li:hover {{ background:var(--surf2) !important; }}
[data-baseweb="popover"] [aria-selected="true"] {{ background:#EDE9FE !important; color:var(--purple) !important; }}

[data-testid="stFileUploadDropzone"] {{
    background:var(--surface) !important; border:2px dashed var(--border) !important;
    border-radius:var(--rad-sm) !important; color:var(--txt) !important; padding:1rem !important;
}}
[data-testid="stFileUploadDropzone"]:hover {{ border-color:#C4B5FD !important; background:#FAFAFF !important; }}
[data-testid="stFileUploadDropzone"] > div {{ color:var(--txt2) !important; }}

/* Popover (+ button attachment menu) */
[data-testid="stPopoverBody"] {{
    background:var(--surface) !important; border:1px solid var(--border) !important;
    border-radius:var(--radius) !important; box-shadow:var(--shadow-lg) !important;
    padding:.5rem !important; min-width:200px !important;
}}
[data-testid="stPopoverBody"] * {{ color:var(--txt) !important; }}
[data-testid="stPopoverBody"] [data-baseweb="select"] > div {{
    background:var(--surface) !important;
}}

button[data-testid="stPopoverButton"] {{
    border-radius:50% !important; width:36px !important; height:36px !important;
    padding:0 !important; font-size:1.3rem !important;
    border:none !important;
    background:transparent !important; color:var(--txt2) !important;
    box-shadow:none !important; transition:all .15s;
    display:flex !important; align-items:center !important; justify-content:center !important;
}}
button[data-testid="stPopoverButton"]:hover {{
    background:var(--surf2) !important; color:var(--txt) !important;
}}

.stButton > button {{
    border-radius:var(--rad-sm) !important;
    background:var(--surface) !important; border:1px solid var(--border) !important;
    color:var(--txt) !important; font-weight:500 !important; transition:all .15s ease !important;
}}
.stButton > button:hover {{ background:var(--surf2) !important; border-color:#C4B5FD !important; color:var(--purple) !important; }}
.stButton > button[kind="primary"] {{ background:var(--accent) !important; border-color:var(--accent) !important; color:#fff !important; }}

[data-testid="stToggle"] label {{ color:var(--txt) !important; }}
[data-testid="stCaptionContainer"] p {{ color:var(--txt2) !important; }}
h1,h2,h3,h4,h5,h6 {{ color:var(--txt) !important; }}
p, li, td, th, label {{ color:var(--txt) !important; }}
hr {{ border-color:var(--border) !important; }}
::placeholder {{ color:var(--txt2) !important; opacity:1 !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def _init() -> None:
    defaults = {
        "messages":       [],
        "vector_store":   None,
        "docs_processed": False,
        "doc_names":      [],
        "agent_mode":     "Standard Chat",
        "response_mode":  "Concise",
        "web_search":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _llm(api_key: str, model: str):
    return get_chatgroq_model(api_key=api_key, model_name=model)


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_documents(files):
    with st.spinner("Indexing reports…"):
        try:
            emb    = get_embedding_model()
            docs   = load_pdf_documents(files)
            chunks = split_documents(docs)
            vs     = build_vector_store(chunks, emb)
            st.session_state.vector_store   = vs
            st.session_state.docs_processed = True
            st.session_state.doc_names      = [f.name for f in files]
            names = ", ".join(st.session_state.doc_names)
            st.session_state.messages.append({
                "role":    "assistant",
                "content": (f"✅ **{len(files)} report(s) indexed successfully.**\n\n"
                            f"_{names}_\n\nYou can now ask questions about these documents."),
            })
        except Exception as exc:
            logger.error("Doc error: %s", exc)
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"⚠️ Processing failed: {exc}",
            })


# ─────────────────────────────────────────────────────────────────────────────
# RAG + WEB SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def handle_chat(cfg: dict, question: str):
    rag_ctx = ""; web_ctx = ""; sources = ""; web_needed = False

    if st.session_state.vector_store:
        try:
            chunks = retrieve_relevant_chunks(question, st.session_state.vector_store)
            if chunks:
                rag_ctx = "\n\n---\n\n".join(
                    f"[{c.metadata.get('source_file','doc')} p.{c.metadata.get('page',0)+1}]\n{c.page_content}"
                    for c in chunks)
                sources = "**📄 Sources**\n\n" + "\n\n---\n".join(
                    f"**{c.metadata.get('source_file','doc')}** (p.{c.metadata.get('page',0)+1})\n"
                    f"{c.page_content[:200]}…" for c in chunks)
            else:
                web_needed = True
        except Exception as e:
            logger.warning("RAG error: %s", e); web_needed = True
    else:
        web_needed = True

    kw = ["latest","recent","current","news","2024","2025","search","look up"]
    if question and any(k in question.lower() for k in kw):
        web_needed = True

    if cfg["web_search"] and cfg["tavily_key"] and web_needed:
        try:
            web_ctx = web_search(question, cfg["tavily_key"])
            sources += ("\n\n---\n\n" if sources else "") + "**🌐 Web Results**\n\n" + web_ctx
        except Exception as e:
            logger.warning("Web search error: %s", e)

    tpl = CONCISE_PROMPT if cfg["mode"] == "Concise" else DETAILED_PROMPT
    prompt = tpl.format(
        context=rag_ctx or "No document context.",
        web_results=web_ctx or "No web results.",
        question=question,
    )
    return prompt, sources


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _init()

    groq_key   = get_groq_api_key()
    tavily_key = get_tavily_api_key()
    cfg = {
        "groq_key":   groq_key,
        "tavily_key": tavily_key,
        "model":      DEFAULT_GROQ_MODEL,
        "mode":       st.session_state.response_mode,
        "web_search": st.session_state.web_search,
    }

    # ═══════════════════════════════════════════════════════════
    #  HEADER
    # ═══════════════════════════════════════════════════════════
    rag_cls = "badge" if st.session_state.docs_processed else "badge badge-off"
    rag_lbl = (f"✦ RAG Active — {len(st.session_state.doc_names)} report(s)"
               if st.session_state.docs_processed else "○ No reports loaded")
    st.markdown(f"""
    <div class="app-header">
        <h1>🩺 MediAssist AI</h1>
        <p>Upload medical reports, ask questions, get structured clinical insights.</p>
        <span class="{rag_cls}">{rag_lbl}</span>
    </div>
    """, unsafe_allow_html=True)

    if not groq_key:
        st.markdown('<div class="abox abox-warn">⚠️ <strong>Groq API key missing.</strong> Add it to <code>.env</code> and restart.</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    #  TWO-COLUMN LAYOUT
    # ═══════════════════════════════════════════════════════════
    col_chat, col_side = st.columns([2.4, 1], gap="large")

    # ───────────────────────────────────────────────────────────
    #  RIGHT PANEL: compact settings accordion only
    # ───────────────────────────────────────────────────────────
    with col_side:
        if st.session_state.docs_processed:
            st.markdown('<div class="abox abox-ok">✅ Reports ready</div>', unsafe_allow_html=True)
            for n in st.session_state.doc_names:
                st.caption(f"📎 {n}")

        with st.expander("⚙️  Assistant Settings", expanded=False):
            modes = ["Standard Chat", "Diagnostic Analysis", "Summarise Jargon"]
            st.session_state.agent_mode = st.selectbox(
                "Mode", modes, index=modes.index(st.session_state.agent_mode))
            details = ["Concise", "Detailed"]
            st.session_state.response_mode = st.selectbox(
                "Detail Level", details, index=details.index(st.session_state.response_mode))
            st.session_state.web_search = st.toggle(
                "Web Search Fallback", value=st.session_state.web_search,
                help="Searches the web when reports lack relevant content.")
            cfg["mode"]       = st.session_state.response_mode
            cfg["web_search"] = st.session_state.web_search
            st.divider()
            st.markdown("<p style='font-size:.75rem;color:var(--txt2);margin:0;'>"
                        "⚠️ <strong>Disclaimer:</strong> Decision-support only. "
                        "Not a substitute for professional medical advice.</p>",
                        unsafe_allow_html=True)

    # ───────────────────────────────────────────────────────────
    #  LEFT PANEL: scrollable messages + fixed composer
    # ───────────────────────────────────────────────────────────
    with col_chat:

        # ── Scrollable message container ──────────────────────
        chat_box = st.container(height=CHAT_HEIGHT, border=False)

        with chat_box:
            if not st.session_state.messages:
                st.markdown("""
                <div class="empty-state">
                    <div class="icon">🩺</div>
                    <h2>Medical Report Assistant</h2>
                    <p>Upload a report using the <strong>＋</strong> button below, or ask a clinical question.</p>
                    <div class="chips">
                        <div class="chip">📁 Upload a report</div>
                        <div class="chip">📝 Summarise jargon</div>
                        <div class="chip">🔬 Run analysis</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if msg.get("sources"):
                            with st.expander("📎 Sources", expanded=False):
                                st.markdown(msg["sources"])

        # ── Bottom fixed input area ──────────────────────────
        c1, c2 = st.columns([1, 15], vertical_alignment="bottom", gap="small")
        with c1:
            # + Popover (docked via CSS margin/negative horizontal hack)
            with st.popover("＋"):
                st.markdown("**📎 Attach medical reports**")
                uploaded = st.file_uploader(
                    "Upload PDFs", type=["pdf"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                    key="attach_uploader",
                )
                if uploaded:
                    if st.button("⚡ Process & Index", use_container_width=True):
                        process_documents(uploaded)
                        st.rerun()
                st.divider()
                st.session_state.web_search = st.toggle(
                    "🌐 Web Search",
                    value=st.session_state.web_search,
                    key="quick_websearch",
                )
                cfg["web_search"] = st.session_state.web_search
                st.divider()
                st.caption("⚠️ Medical AI assistant. For information only.")

        with c2:
            # Fixed chat input
            placeholder_map = {
                "Standard Chat":       "Ask about your medical reports…",
                "Diagnostic Analysis": "Describe symptoms or paste results…",
                "Summarise Jargon":   "Paste medical text to simplify…",
            }
            placeholder = placeholder_map.get(st.session_state.agent_mode, "Ask anything…")

            if not groq_key:
                st.chat_input("Add a Groq API key to start chatting.", disabled=True)
            else:
                user_input = st.chat_input(placeholder)

                if user_input:
                    st.session_state.messages.append({"role": "user", "content": user_input})

                    with chat_box:
                        with st.chat_message("user"):
                            st.markdown(user_input)

                        with st.chat_message("assistant"):
                            with st.spinner("Analysing…"):
                                try:
                                    llm = _llm(cfg["groq_key"], cfg["model"])
                                    answer = ""; sources = None

                                    if st.session_state.agent_mode == "Standard Chat":
                                        prompt, sources = handle_chat(cfg, user_input)
                                        answer = llm.invoke([HumanMessage(content=prompt)]).content

                                    elif st.session_state.agent_mode == "Diagnostic Analysis":
                                        if not st.session_state.vector_store:
                                            answer = ("⚠️ **No reports loaded.**\n\n"
                                                      "Use the **＋** button to upload and process a PDF first.")
                                        else:
                                            report, score = run_diagnostic_analysis(
                                                patient_features=user_input,
                                                vector_store=st.session_state.vector_store,
                                                llm=llm,
                                            )
                                            answer = (f"### 📊 Interpretation Report\n"
                                                      f"**Concern Level Score: {score}/100**\n\n{report}")

                                    elif st.session_state.agent_mode == "Summarise Jargon":
                                        filled = SUMMARIZER_PROMPT.format(medical_text=user_input)
                                        answer = llm.invoke([HumanMessage(content=filled)]).content

                                    st.markdown(answer)
                                    if sources:
                                        with st.expander("📎 Sources", expanded=False):
                                            st.markdown(sources)

                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": answer,
                                        "sources": sources,
                                    })

                                except Exception as exc:
                                    st.error(f"Error: {exc}")
                                    logger.error("Chat error: %s", exc)
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": f"⚠️ An error occurred: {exc}",
                                    })


if __name__ == "__main__":
    main()
