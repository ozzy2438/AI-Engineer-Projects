"""
DocMind AI - Advanced Multi-Agent Document Intelligence System
Built with Streamlit, Agno, OpenAI, and Milvus

Features:
- Multi-document PDF upload and management
- 6 specialized AI agents (Q&A, Summarizer, Researcher, Analyst, Translator, Data Extractor)
- PDF table extraction to CSV with pdfplumber
- Chart/graph image analysis via GPT-4o Vision
- Streaming responses with real-time token display
- Model selection (GPT-4o-mini, GPT-4o, GPT-4-turbo)
- AI-generated follow-up question suggestions
- Document insights with auto-summary and quick actions
- Conversation export to Markdown
- Temperature/creativity control
- Modern gradient UI with professional dark theme
"""

import streamlit as st
import os
import base64
from datetime import datetime
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.milvus import Milvus

from agents import create_agent, AGENT_CONFIGS, get_agent_options
from data_extractor import (
    extract_all,
    analyze_chart_with_vision,
    ExtractionResult,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEMP_DIR = Path("temp_docs")
TEMP_DIR.mkdir(exist_ok=True)

AVAILABLE_MODELS = {
    "gpt-4o-mini": "GPT-4o Mini  (Fast & Efficient)",
    "gpt-4o": "GPT-4o  (Balanced)",
    "gpt-4-turbo": "GPT-4 Turbo  (Most Capable)",
}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Gradient header */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 2rem;
    border-radius: 1rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}
.main-header h1 { color: white !important; margin: 0; font-size: 2rem; }
.main-header p  { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0; font-size: 1rem; }

/* Sidebar */
.stSidebar > div:first-child {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}

/* Agent info card */
.agent-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 0.75rem;
    padding: 1rem;
    margin: 0.5rem 0;
}
.agent-card h4 { margin: 0 0 0.3rem; color: #a78bfa; }
.agent-card p  { margin: 0; font-size: 0.85rem; opacity: 0.8; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: rgba(74,74,106,0.3);
    border-radius: 0.75rem;
    padding: 1rem;
    margin-bottom: 0.5rem;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Document card */
.doc-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin: 0.3rem 0;
}

/* Stat card */
.stat-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border-radius: 0.75rem;
    padding: 1.25rem;
    text-align: center;
    border: 1px solid rgba(102,126,234,0.3);
}
.stat-card h2 { margin: 0; color: #a78bfa; }
.stat-card p  { margin: 0.25rem 0 0; opacity: 0.7; font-size: 0.9rem; }

/* Feature card (landing page) */
.feature-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(118,75,162,0.12));
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 0.75rem;
    padding: 1.5rem;
    text-align: center;
    height: 100%;
}
.feature-card h3 { color: #a78bfa; margin: 0.5rem 0; }
.feature-card p  { font-size: 0.9rem; opacity: 0.8; }

/* PDF preview */
.pdf-preview-container {
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 0.75rem;
    padding: 0.5rem;
    background: rgba(0,0,0,0.2);
    margin: 0.5rem 0;
}

/* Sidebar divider */
.sidebar-divider { border-top: 1px solid rgba(255,255,255,0.1); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "messages": [],
    "documents_loaded": False,
    "agent": None,
    "knowledge_base": None,
    "uploaded_file_names": [],
    "processed_files": [],
    "current_agent_type": "qa",
    "current_model": "gpt-4o-mini",
    "current_temperature": 0.7,
    "query_count": 0,
    "session_start": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "document_summary": None,
    "suggestions": [],
    "last_suggestion_for": "",
    "extraction_result": None,
    "analyzed_images": {},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        try:
            key = st.secrets["OPENAI_API_KEY"]
        except (FileNotFoundError, KeyError):
            pass
    return key


def get_vector_db() -> Milvus:
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    return Milvus(
        collection="docmind_rag_collection",
        uri=uri,
        embedder=OpenAIEmbedder(),
    )


def save_uploaded_files(files) -> List[str]:
    paths = []
    for f in files:
        p = TEMP_DIR / f.name
        p.write_bytes(f.getvalue())
        paths.append(str(p))
    return paths


def display_pdf_preview(pdf_path: str, height: int = 250):
    try:
        b64 = base64.b64encode(Path(pdf_path).read_bytes()).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="{height}px" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Preview error: {e}")


def process_documents(file_paths: List[str]) -> bool:
    try:
        vector_db = get_vector_db()
        kb_path = file_paths[0] if len(file_paths) == 1 else str(TEMP_DIR)
        kb = PDFKnowledgeBase(path=kb_path, vector_db=vector_db)
        kb.load(recreate=True)

        agent = create_agent(
            agent_type=st.session_state.current_agent_type,
            knowledge_base=kb,
            model_id=st.session_state.current_model,
            temperature=st.session_state.current_temperature,
        )
        st.session_state.knowledge_base = kb
        st.session_state.agent = agent
        st.session_state.documents_loaded = True
        return True
    except Exception as e:
        st.error(f"Processing error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False


def rebuild_agent():
    if st.session_state.knowledge_base is not None:
        st.session_state.agent = create_agent(
            agent_type=st.session_state.current_agent_type,
            knowledge_base=st.session_state.knowledge_base,
            model_id=st.session_state.current_model,
            temperature=st.session_state.current_temperature,
        )


def export_conversation() -> str:
    lines = [
        "# DocMind AI - Conversation Export",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Documents:** {', '.join(st.session_state.processed_files)}",
        f"**Agent:** {AGENT_CONFIGS[st.session_state.current_agent_type]['icon']} "
        f"{AGENT_CONFIGS[st.session_state.current_agent_type]['name']}",
        f"**Model:** {st.session_state.current_model}",
        "",
        "---",
        "",
    ]
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.extend([f"### {role}", msg["content"], ""])
    return "\n".join(lines)


def generate_follow_up_questions(context: str) -> List[str]:
    """Generate 3 short follow-up questions using the OpenAI API directly."""
    try:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate exactly 3 short follow-up questions (each under 10 words). "
                        "Return ONLY the questions, one per line, no numbering or bullet points."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context[:600]}"},
            ],
            max_tokens=150,
            temperature=0.8,
        )
        raw = resp.choices[0].message.content.strip().split("\n")
        return [q.strip() for q in raw if q.strip()][:3]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 DocMind AI")
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- API Key ---
    api_key = get_openai_api_key()
    if not api_key:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to get started.",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.success("API Key configured", icon="✅")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- Document Upload (multi-file) ---
    st.markdown("### Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        help="Upload one or more PDF files to analyse.",
    )

    if uploaded_files:
        current_names = sorted(f.name for f in uploaded_files)
        if current_names != sorted(st.session_state.processed_files):
            st.session_state.documents_loaded = False
            st.session_state.agent = None
            st.session_state.document_summary = None
            st.session_state.extraction_result = None
            st.session_state.analyzed_images = {}

        for uf in uploaded_files:
            size_mb = round(len(uf.getvalue()) / 1024 / 1024, 2)
            st.markdown(
                f'<div class="doc-card">📄 <strong>{uf.name}</strong>'
                f"<br/><small>{size_mb} MB</small></div>",
                unsafe_allow_html=True,
            )

        if not st.session_state.documents_loaded:
            if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
                with st.spinner("Indexing documents..."):
                    paths = save_uploaded_files(uploaded_files)
                    if process_documents(paths):
                        st.session_state.processed_files = [f.name for f in uploaded_files]
                        st.session_state.messages = []
                        st.session_state.suggestions = []
                        st.rerun()
        else:
            st.success(f"{len(st.session_state.processed_files)} document(s) ready", icon="✅")

        # Preview first file
        if uploaded_files:
            with st.expander("Preview first PDF", expanded=False):
                tmp = TEMP_DIR / uploaded_files[0].name
                if tmp.exists():
                    display_pdf_preview(str(tmp))
    else:
        if st.session_state.documents_loaded:
            st.session_state.documents_loaded = False
            st.session_state.messages = []
            st.session_state.agent = None
            st.session_state.processed_files = []
            st.session_state.document_summary = None
            st.session_state.extraction_result = None
            st.session_state.analyzed_images = {}
            st.session_state.suggestions = []
            st.rerun()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- Settings ---
    st.markdown("### Settings")

    selected_model = st.selectbox(
        "Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS[x],
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.current_model),
    )
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        rebuild_agent()

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.current_temperature,
        step=0.1,
        help="Lower = more precise, Higher = more creative",
    )
    if temperature != st.session_state.current_temperature:
        st.session_state.current_temperature = temperature
        rebuild_agent()

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- Agent Selection ---
    st.markdown("### Agent Mode")

    agent_options = get_agent_options()
    for agent_key, agent_label in agent_options.items():
        is_active = st.session_state.current_agent_type == agent_key
        if st.button(
            agent_label,
            key=f"agent_btn_{agent_key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if agent_key != st.session_state.current_agent_type:
                st.session_state.current_agent_type = agent_key
                rebuild_agent()
                st.rerun()

    active = AGENT_CONFIGS[st.session_state.current_agent_type]
    st.markdown(
        f'<div class="agent-card"><h4>{active["icon"]} {active["name"]}</h4>'
        f'<p>{active["description"]}</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # --- Session Stats ---
    st.markdown("### Session")
    c1, c2 = st.columns(2)
    c1.metric("Queries", st.session_state.query_count)
    c2.metric("Docs", len(st.session_state.processed_files))

    if st.session_state.messages:
        st.download_button(
            "📥 Export Chat",
            data=export_conversation(),
            file_name=f"docmind_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.document_summary = None
            st.session_state.suggestions = []
            st.rerun()


# ---------------------------------------------------------------------------
# Main Content Area
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="main-header">'
    "<h1>🧠 DocMind AI</h1>"
    "<p>Advanced Multi-Agent Document Intelligence System</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---- Landing page (no documents yet) ----
if not st.session_state.documents_loaded:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="feature-card"><h3>📄 Multi-Document</h3>'
            "<p>Upload multiple PDFs and cross-reference information across all of them.</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="feature-card"><h3>🤖 6 AI Agents</h3>'
            "<p>Q&A, Summarizer, Researcher, Analyst, Translator and Data Extractor.</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="feature-card"><h3>🔬 Deep Research</h3>'
            "<p>Combine document knowledge with live web search for comprehensive analysis.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown(
            '<div class="feature-card"><h3>⚡ Streaming</h3>'
            "<p>Watch answers appear in real-time with token-by-token streaming.</p></div>",
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            '<div class="feature-card"><h3>💡 Smart Suggestions</h3>'
            "<p>AI-generated follow-up questions after every answer to guide exploration.</p></div>",
            unsafe_allow_html=True,
        )
    with col6:
        st.markdown(
            '<div class="feature-card"><h3>📊 Data Extraction</h3>'
            "<p>Extract tables to CSV, analyze charts with GPT-4o Vision, and scrape structured data.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info("👈 Upload PDF documents in the sidebar to get started.")

# ---- Document loaded – show tabs ----
else:
    tab_chat, tab_insights, tab_data = st.tabs(["💬 Chat", "📊 Insights", "🗃️ Data Extraction"])

    # ================================================================
    # CHAT TAB
    # ================================================================
    with tab_chat:
        # Render history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Suggested follow-up questions
        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "assistant"
        ):
            last_resp = st.session_state.messages[-1]["content"]
            sig = last_resp[:120]
            if st.session_state.last_suggestion_for != sig:
                st.session_state.suggestions = generate_follow_up_questions(last_resp[:600])
                st.session_state.last_suggestion_for = sig

            if st.session_state.suggestions:
                st.markdown("**💡 Suggested follow-ups:**")
                cols = st.columns(len(st.session_state.suggestions))
                for idx, (col, q) in enumerate(zip(cols, st.session_state.suggestions)):
                    with col:
                        if st.button(q, key=f"sugg_{idx}", use_container_width=True):
                            st.session_state.messages.append({"role": "user", "content": q})
                            st.session_state.suggestions = []
                            st.rerun()

        # Chat input
        if prompt := st.chat_input("Ask anything about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.suggestions = []

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                with st.spinner("Thinking..."):
                    try:
                        # Attempt streaming first
                        stream = st.session_state.agent.run(prompt, stream=True)
                        for chunk in stream:
                            if chunk.content:
                                # Agno streams may yield cumulative or delta text
                                if len(chunk.content) >= len(full_response):
                                    full_response = chunk.content
                                else:
                                    full_response += chunk.content
                                placeholder.markdown(full_response + " ▌")
                        placeholder.markdown(full_response)
                    except Exception:
                        # Fallback to non-streaming
                        try:
                            resp = st.session_state.agent.run(prompt)
                            full_response = resp.content
                            placeholder.markdown(full_response)
                        except Exception as e2:
                            full_response = f"Error: {e2}"
                            st.error(full_response)
                            import traceback
                            st.error(traceback.format_exc())

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            st.session_state.query_count += 1
            st.rerun()

    # ================================================================
    # INSIGHTS TAB
    # ================================================================
    with tab_insights:
        st.markdown("### Document Insights")

        # Stats row
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f'<div class="stat-card"><h2>{len(st.session_state.processed_files)}</h2>'
                "<p>Documents</p></div>",
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f'<div class="stat-card"><h2>{st.session_state.query_count}</h2>'
                "<p>Queries</p></div>",
                unsafe_allow_html=True,
            )
        with s3:
            icon = AGENT_CONFIGS[st.session_state.current_agent_type]["icon"]
            st.markdown(
                f'<div class="stat-card"><h2>{icon}</h2>'
                "<p>Active Agent</p></div>",
                unsafe_allow_html=True,
            )
        with s4:
            model_short = st.session_state.current_model.replace("gpt-", "").upper()
            st.markdown(
                f'<div class="stat-card"><h2>{model_short}</h2>'
                "<p>Model</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Auto-summary section
        st.markdown("### Document Summary")
        if st.session_state.document_summary:
            st.markdown(st.session_state.document_summary)
        else:
            if st.button("🔄 Generate Summary", type="primary"):
                with st.spinner("Generating comprehensive summary..."):
                    try:
                        summarizer = create_agent(
                            agent_type="summarizer",
                            knowledge_base=st.session_state.knowledge_base,
                            model_id=st.session_state.current_model,
                            temperature=0.3,
                        )
                        summary_resp = summarizer.run(
                            "Provide a comprehensive summary of all the documents in the "
                            "knowledge base. Include: Executive Summary, Key Points, "
                            "Main Sections overview, and Notable Data/Statistics."
                        )
                        st.session_state.document_summary = summary_resp.content
                        st.rerun()
                    except Exception as e:
                        st.error(f"Summary error: {e}")

        st.markdown("---")

        # Quick Actions
        st.markdown("### Quick Actions")
        q1, q2, q3 = st.columns(3)

        with q1:
            if st.button("📋 Key Takeaways", use_container_width=True):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": "What are the top 5 key takeaways from this document?",
                    }
                )
                st.session_state.current_agent_type = "summarizer"
                rebuild_agent()
                st.rerun()

        with q2:
            if st.button("❓ Generate FAQ", use_container_width=True):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Generate a FAQ with 10 questions and answers "
                            "based on this document."
                        ),
                    }
                )
                st.session_state.current_agent_type = "qa"
                rebuild_agent()
                st.rerun()

        with q3:
            if st.button("🔍 Critical Analysis", use_container_width=True):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Provide a critical analysis of this document. "
                            "What are its strengths, weaknesses, and gaps?"
                        ),
                    }
                )
                st.session_state.current_agent_type = "analyst"
                rebuild_agent()
                st.rerun()

        q4, q5, q6 = st.columns(3)

        with q4:
            if st.button("🌍 Translate Summary", use_container_width=True):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Translate the main summary and key points of this document "
                            "into Turkish (Turkce)."
                        ),
                    }
                )
                st.session_state.current_agent_type = "translator"
                rebuild_agent()
                st.rerun()

        with q5:
            if st.button("🔬 Research Context", use_container_width=True):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Research the main topic of this document on the web and "
                            "provide additional context, recent developments, and how "
                            "this document fits into the broader landscape."
                        ),
                    }
                )
                st.session_state.current_agent_type = "researcher"
                rebuild_agent()
                st.rerun()

        with q6:
            if st.button("📊 Data & Statistics", use_container_width=True):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Extract and list all important data points, statistics, "
                            "numbers, and quantitative information from this document."
                        ),
                    }
                )
                st.session_state.current_agent_type = "qa"
                rebuild_agent()
                st.rerun()

    # ================================================================
    # DATA EXTRACTION TAB
    # ================================================================
    with tab_data:
        st.markdown("### Data Extraction")
        st.markdown(
            "Extract tables as CSV and analyze charts/graphs from your PDF documents. "
            "Perfect for data analysts who need structured data from reports."
        )

        # Run extraction
        if st.session_state.extraction_result is None:
            if st.button("⚡ Extract Tables & Charts", type="primary", use_container_width=True):
                with st.spinner("Scanning PDF for tables and charts..."):
                    # Extract from all uploaded PDFs
                    all_tables = []
                    all_images = []
                    total_pages = 0
                    file_names = []

                    for fname in st.session_state.processed_files:
                        pdf_path = str(TEMP_DIR / fname)
                        result = extract_all(pdf_path)
                        # Tag tables/images with file name for multi-doc
                        for t in result.tables:
                            t._file_name = fname
                        for img in result.images:
                            img._file_name = fname
                        all_tables.extend(result.tables)
                        all_images.extend(result.images)
                        total_pages += result.page_count
                        file_names.append(fname)

                    combined = ExtractionResult(
                        file_name=", ".join(file_names),
                        tables=all_tables,
                        images=all_images,
                        page_count=total_pages,
                    )
                    st.session_state.extraction_result = combined
                    st.rerun()
        else:
            result = st.session_state.extraction_result

            # Summary stats
            st.markdown("---")
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.markdown(
                    f'<div class="stat-card"><h2>{len(result.tables)}</h2>'
                    "<p>Tables Found</p></div>",
                    unsafe_allow_html=True,
                )
            with ec2:
                st.markdown(
                    f'<div class="stat-card"><h2>{len(result.images)}</h2>'
                    "<p>Charts/Images Found</p></div>",
                    unsafe_allow_html=True,
                )
            with ec3:
                st.markdown(
                    f'<div class="stat-card"><h2>{result.page_count}</h2>'
                    "<p>Pages Scanned</p></div>",
                    unsafe_allow_html=True,
                )

            # Re-extract button
            if st.button("🔄 Re-extract", use_container_width=False):
                st.session_state.extraction_result = None
                st.session_state.analyzed_images = {}
                st.rerun()

            st.markdown("---")

            # ---- TABLES SECTION ----
            if result.tables:
                st.markdown("### Extracted Tables")

                for i, table in enumerate(result.tables):
                    file_label = getattr(table, "_file_name", "")
                    header = f"📋 {file_label} — {table.label}" if file_label else f"📋 {table.label}"
                    with st.expander(header, expanded=(i == 0)):
                        st.dataframe(table.dataframe, use_container_width=True)

                        col_csv, col_info = st.columns([1, 2])
                        with col_csv:
                            csv_data = table.to_csv()
                            st.download_button(
                                "📥 Download CSV",
                                data=csv_data,
                                file_name=f"table_p{table.page_number}_t{table.table_index + 1}.csv",
                                mime="text/csv",
                                key=f"csv_download_{i}",
                                use_container_width=True,
                            )
                        with col_info:
                            rows, cols = table.dataframe.shape
                            st.caption(f"{rows} rows x {cols} columns")

                # Download ALL tables as one CSV
                if len(result.tables) > 1:
                    st.markdown("---")
                    import io
                    combined_csv = io.StringIO()
                    for j, tbl in enumerate(result.tables):
                        file_label = getattr(tbl, "_file_name", "")
                        combined_csv.write(f"# {file_label} - {tbl.label}\n")
                        combined_csv.write(tbl.to_csv())
                        combined_csv.write("\n\n")
                    st.download_button(
                        "📥 Download ALL Tables (Combined CSV)",
                        data=combined_csv.getvalue(),
                        file_name="all_tables_combined.csv",
                        mime="text/csv",
                        key="csv_download_all",
                        use_container_width=True,
                        type="primary",
                    )
            else:
                st.info("No tables were detected in the uploaded PDF(s).")

            st.markdown("---")

            # ---- CHARTS / IMAGES SECTION ----
            if result.images:
                st.markdown("### Charts & Graphs")
                st.markdown(
                    "Click **Analyze with GPT-4o Vision** to extract numerical data from a chart. "
                    "This sends the image to GPT-4o for analysis."
                )

                for i, img in enumerate(result.images):
                    file_label = getattr(img, "_file_name", "")
                    header = (
                        f"🖼️ {file_label} — {img.label} ({img.width}x{img.height}px)"
                        if file_label
                        else f"🖼️ {img.label} ({img.width}x{img.height}px)"
                    )
                    with st.expander(header, expanded=False):
                        # Show the image
                        st.image(img.image_bytes, use_container_width=True)

                        img_key = f"{file_label}_{img.page_number}_{img.image_index}"

                        if img_key in st.session_state.analyzed_images:
                            analysis, extracted_df = st.session_state.analyzed_images[img_key]
                            st.markdown("#### Analysis")
                            st.markdown(analysis)

                            if extracted_df is not None and not extracted_df.empty:
                                st.markdown("#### Extracted Data")
                                st.dataframe(extracted_df, use_container_width=True)
                                st.download_button(
                                    "📥 Download Chart Data CSV",
                                    data=extracted_df.to_csv(index=False),
                                    file_name=f"chart_p{img.page_number}_i{img.image_index + 1}.csv",
                                    mime="text/csv",
                                    key=f"chart_csv_{i}",
                                    use_container_width=True,
                                )
                        else:
                            if st.button(
                                "🔍 Analyze with GPT-4o Vision",
                                key=f"analyze_img_{i}",
                                use_container_width=True,
                                type="primary",
                            ):
                                with st.spinner("Analyzing chart with GPT-4o Vision..."):
                                    analysis, extracted_df = analyze_chart_with_vision(img)
                                    st.session_state.analyzed_images[img_key] = (
                                        analysis,
                                        extracted_df,
                                    )
                                    st.rerun()
            else:
                st.info(
                    "No charts or significant images were detected in the uploaded PDF(s). "
                    "Small icons and logos are automatically filtered out."
                )
