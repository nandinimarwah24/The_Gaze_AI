import html as html_lib
import json
import re
import socket
import subprocess
import time
from pathlib import Path
from memory.memory_manager import *
from memory.memory_extractor import extract_memory

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from faiss_store import (
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    INDEX_PATH,
    build_faiss_index,
    load_faiss_store,
    search_faiss,
)
from local_llm import (
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaError,
    answer_with_context,
)
from query import build_retrieval_query, remember_turn
from rag import DATA_DIR, MODEL_NAME, create_chunks_from_pdfs

# ─── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="The Gaze AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── CSS ───────────────────────────────────────────────────────────────────
STYLES = """

<style>

    :root {
        --primary: #2563EB;
    }

/* Light Mode */
[data-theme="light"] {
    --background: #FFFFFF;
    --surface: #F8FAFC;
    --text: #111111;
    --secondary: #374151;
    --border: #D1D5DB;
}

/* Dark Mode */
[data-theme="dark"] {
    --background: #0F172A;
    --surface: #1E293B;
    --text: #F8FAFC;
    --secondary: #CBD5E1;
    --border: #334155;
}
    /* Main App */
    .stApp {
        background-color: var(--background);
        color: var(--text);
        font-family: Times new roman, sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 2px solid var(--border) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    /* Layout */
    .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }

    /* Header */
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
    }

    .app-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primmary);
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .app-subtitle {
        font-size: 0.9rem;
        color: var(--secondary-text);
    }

    /* Empty State */
    .empty-state {
        background: var(--surface);
        border: 1px solid var(--primmary);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 2rem auto;
    }

    .empty-state strong {
        display: block;
        color: var(--text);
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }

    .empty-state span {
        color: var(--secondary-text);
        font-size: 0.95rem;
    }

    /* Chat Container */
    .conversation-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .message-row {
        display: flex;
        width: 100%;
    }

    .message-row.user {
        justify-content: flex-end;
    }

    .message-row.assistant {
        justify-content: flex-start;
    }

    .message {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: 12px;
        font-size: 15px;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
    }

    .message.user {
        background: var(--primary);
        color: white;
    }

    .message.assistant {
        background: var(--surface);
        color: var(--text);
        border: 1px solid var(--border);
    }

    .message-label {
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 6px;
        color: var(--secondary-text);
    }

    /* Buttons */
    .stButton button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
    }

    /* File Uploader */
    .stFileUploader {
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px;
        background: white;
    }

    /* Chat Input */
    .stChatInput {
        border-top: 1px solid var(--border);
    }

    /* Text Inputs */
    .stTextInput input,
    .stTextArea textarea {
        border: 2px solid var(--primary);
        border-radius: 8px;
    }

</style>

"""
# ─── Ollama auto-start ─────────────────────────────────────────────────────
def _ollama_reachable(host: str = "localhost", port: int = 11434) -> bool:
    """Return True if Ollama is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def ensure_ollama_running() -> str:
    """
    If Ollama is not reachable, attempt to start it with `ollama serve`.
    Returns a short status string: 'online' | 'started' | 'offline'.
    """
    if _ollama_reachable():
        return "online"
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Give it up to 8 seconds to come up
        for _ in range(8):
            time.sleep(1)
            if _ollama_reachable():
                return "started"
        return "offline"
    except FileNotFoundError:
        return "offline"


# ─── Helpers ───────────────────────────────────────────────────────────────
def safe_filename(filename: str) -> str:
    name = Path(filename).name
    clean = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    return clean if clean.lower().endswith(".pdf") else f"{clean}.pdf"


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME, local_files_only=True)


@st.cache_resource(show_spinner=False)
def load_vector_store(index_mtime, chunks_mtime):
    del index_mtime, chunks_mtime
    return load_faiss_store()


def get_vector_store():
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    return load_vector_store(
        INDEX_PATH.stat().st_mtime_ns,
        CHUNKS_PATH.stat().st_mtime_ns,
    )


def rebuild_index() -> int:
    chunks = create_chunks_from_pdfs()
    texts = [c["text"] for c in chunks]
    model = load_embedding_model()
    embeddings = model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float32)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(EMBEDDINGS_PATH, embeddings)
    build_faiss_index(embeddings)
    load_vector_store.clear()
    return len(chunks)


def render_conversation(messages: list, pending_question: str | None = None):
    """Render all messages as styled HTML bubbles — no raw HTML leaks."""
    parts = ['<div class="conversation-wrap">']

    def bubble(role: str, text: str) -> str:
        label = "You" if role == "user" else "The Gaze"
        safe = html_lib.escape(text).replace("\n", "<br>")
        return (
            f'<div class="msg-row {role}">'
            f'<div class="msg-bubble {role}">'
            f'<div class="msg-label">{label}</div>'
            f'<div class="msg-content">{safe}</div>'
            f"</div></div>"
        )

    for msg in messages:
        parts.append(bubble(msg["role"], msg["content"]))

    if pending_question:
        parts.append(bubble("user", pending_question))

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ─── Inject CSS ────────────────────────────────────────────────────────────
st.markdown(STYLES, unsafe_allow_html=True)

# ─── Session state init ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_done" not in st.session_state:
    st.session_state.upload_done = False

# ─── Ensure Ollama is running (once per session) ───────────────────────────
if "ollama_status" not in st.session_state:
    st.session_state.ollama_status = ensure_ollama_running()


# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h3 class="sidebar-title">Your Library</h3>', unsafe_allow_html=True)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    st.markdown(
        f'<div class="doc-count">{len(pdf_files)} PDF{"s" if len(pdf_files) != 1 else ""} indexed</div>',
        unsafe_allow_html=True,
    )

    uploads = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help="Files are saved locally and indexed with FAISS.",
    )

    if st.button("Add to library", disabled=not uploads):
        saved = 0
        for uf in uploads:
            dest = DATA_DIR / safe_filename(uf.name)
            dest.write_bytes(uf.getbuffer())
            saved += 1

        with st.spinner("Indexing your PDFs…"):
            try:
                chunk_count = rebuild_index()
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
            else:
                st.success(f"✓ {saved} PDF(s) added — {chunk_count} chunks ready.")
                st.session_state.messages = []
                st.rerun()

    st.markdown(
        '<p class="sidebar-note">Upload PDFs, click <strong>Add to library</strong>, '
        "then ask questions. Memory lasts this session only.</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Ollama status ──
    _status = st.session_state.get("ollama_status", "offline")
    if _status == "online":
        st.caption("🟢 Ollama connected")
    elif _status == "started":
        st.caption("🟡 Ollama started — ready")
    else:
        st.caption("🔴 Ollama offline — install from ollama.com")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ─── Header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="gaze-header">
        <div class="gaze-mark">✦ · ✦ · ✦</div>
        <h1 class="gaze-title">The Gaze AI</h1>
        <div class="gaze-subtitle">A quiet place to look deeper into your documents</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Conversation area ─────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="dreamy-sphere"></div>
            <strong>What would you like to uncover?</strong>
            <span>Begin a quiet dialogue with your documents. The Gaze listens.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    render_conversation(st.session_state.messages)


# ─── Chat input ────────────────────────────────────────────────────────────
question = st.chat_input("Ask The Gaze about your documents…")

if question:
    # Save any personal information from the user's message
    memory = extract_memory(question)
    for key, value in memory.items():
         update_memory(key, value)
    # Optimistically show user message while we wait
    render_conversation(st.session_state.messages, pending_question=question)

    try:
        embedding_model = load_embedding_model()
        chunks, index = get_vector_store()

        if chunks is None or index is None:
            raise ValueError("No PDFs indexed yet — upload a PDF from the sidebar first.")

        retrieval_query = build_retrieval_query(question, st.session_state.messages)

        with st.spinner("The Gaze is reading…"):
            results = search_faiss(
                retrieval_query, embedding_model, chunks, index, top_k=3
            )
            answer = answer_with_context(
                question,
                results,
                chat_history=st.session_state.messages,
                model=DEFAULT_LLM_MODEL,
                base_url=DEFAULT_OLLAMA_URL,
                timeout=120,
            )

    except (OllamaError, OSError, ValueError) as exc:
        # If it looks like a connection error, retry auto-start once
        err_str = str(exc)
        if "Could not connect" in err_str or "ollama" in err_str.lower():
            retry_status = ensure_ollama_running()
            st.session_state.ollama_status = retry_status
            if retry_status in ("online", "started"):
                try:
                    results = search_faiss(
                        retrieval_query, embedding_model, chunks, index, top_k=3
                    )
                    answer = answer_with_context(
                        question,
                        results,
                        chat_history=st.session_state.messages,
                        model=DEFAULT_LLM_MODEL,
                        base_url=DEFAULT_OLLAMA_URL,
                        timeout=120,
                    )
                except (OllamaError, OSError, ValueError) as retry_exc:
                    answer = f"I couldn't complete that request. {retry_exc}"
            else:
                answer = (
                    "Ollama is not running. Please start it by opening a terminal "
                    "and running: ollama serve"
                )
        else:
            answer = f"I couldn't complete that request. {exc}"

    remember_turn(st.session_state.messages, question, answer, max_turns=5)
    st.rerun()