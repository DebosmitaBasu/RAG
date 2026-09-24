import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

LLM_MODEL = os.getenv(
    "GEMINI_LLM_MODEL",
    "gemini-3.6-flash"
)

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# CHECK API KEY
# ============================================================

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing from your .env file.")
    st.stop()


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAGly - AI Knowledge Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
    <style>

    .stApp {
        background: #0b0d12;
        color: #f5f5f5;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit branding */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: #10131b;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    /* Brand */

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 13px;

        display: flex;
        align-items: center;
        justify-content: center;

        background: linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6
        );

        font-size: 22px;
        font-weight: 700;

        box-shadow:
            0 8px 30px rgba(99,102,241,0.30);
    }

    .brand-name {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    .brand-subtitle {
        color: #8f96a8;
        font-size: 12px;
        margin-left: 54px;
        margin-bottom: 25px;
    }

    /* Section headings */

    .section-title {
        color: #9da5b8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Hero */

    .hero {
        text-align: center;
        padding: 40px 20px 25px 20px;
    }

    .hero-badge {
        display: inline-block;

        padding: 6px 13px;

        border-radius: 999px;

        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);

        color: #a5b4fc;

        font-size: 11px;
        font-weight: 700;

        margin-bottom: 16px;
    }

    .hero h1 {
        font-size: 42px;
        line-height: 1.1;
        letter-spacing: -1.5px;
        margin: 0;
        font-weight: 800;

        background: linear-gradient(
            90deg,
            #ffffff,
            #c7d2fe
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #9299aa;
        max-width: 650px;
        margin: 14px auto 0 auto;
        font-size: 15px;
        line-height: 1.6;
    }

    /* Online status */

    .status-card {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;

        width: fit-content;

        margin: 18px auto;

        padding: 7px 13px;

        border-radius: 999px;

        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.18);

        color: #86efac;

        font-size: 12px;
    }

    .status-dot {
        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: #22c55e;

        box-shadow:
            0 0 10px rgba(34,197,94,0.7);
    }

    /* Empty state */

    .empty-card {
        margin: 20px auto;

        max-width: 820px;

        padding: 35px;

        border-radius: 22px;

        background: rgba(255,255,255,0.025);

        border: 1px solid rgba(255,255,255,0.08);

        text-align: center;
    }

    .empty-icon {
        font-size: 38px;
        margin-bottom: 10px;
    }

    .empty-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .empty-text {
        color: #8f96a8;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Feature cards */

    .feature-card {
        min-height: 125px;

        padding: 20px;

        border-radius: 17px;

        background: rgba(255,255,255,0.025);

        border: 1px solid rgba(255,255,255,0.07);
    }

    .feature-icon {
        font-size: 23px;
        margin-bottom: 8px;
    }

    .feature-title {
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .feature-text {
        color: #81899b;
        font-size: 12px;
        line-height: 1.5;
    }

    /* Source cards */

    .source-card {
        display: flex;
        align-items: center;
        gap: 9px;

        padding: 9px 11px;

        margin-bottom: 7px;

        border-radius: 10px;

        background: rgba(255,255,255,0.035);

        border: 1px solid rgba(255,255,255,0.055);

        font-size: 12px;

        color: #c5cad6;

        overflow: hidden;
    }

    .source-icon {
        font-size: 16px;
    }

    .source-name {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* Metrics */

    .metric-card {
        padding: 14px 16px;

        border-radius: 14px;

        background: rgba(255,255,255,0.025);

        border: 1px solid rgba(255,255,255,0.07);
    }

    .metric-number {
        font-size: 22px;
        font-weight: 750;
    }

    .metric-label {
        color: #81899b;
        font-size: 11px;
        margin-top: 2px;
    }

    /* Retrieved sources */

    .result-source {
        padding: 11px 14px;

        margin-top: 7px;

        border-radius: 11px;

        background: rgba(99,102,241,0.06);

        border: 1px solid rgba(99,102,241,0.14);

        color: white;

        font-size: 12px;
    }

    /* Chat messages */

    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.018);

        border: 1px solid rgba(255,255,255,0.055);

        border-radius: 18px;

        margin-bottom: 12px;
    }

    /* Buttons */

    .stButton > button {
        border-radius: 11px;

        border: 1px solid rgba(255,255,255,0.09);

        background: rgba(255,255,255,0.045);

        color: #e8eaf0;

        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: rgba(129,140,248,0.55);

        background: rgba(99,102,241,0.12);

        color: white;
    }

    /* Primary button */

    button[kind="primary"] {
        background: linear-gradient(
            135deg,
            #6366f1,
            #7c3aed
        ) !important;

        border: none !important;
    }

    /* Text areas */

    .stTextArea textarea {
        background: rgba(255,255,255,0.055) !important;

        border: 1px solid rgba(255,255,255,0.12) !important;

        border-radius: 12px !important;

        color: #eef0f6 !important;
    }

    .stTextArea textarea::placeholder {
        color: #5a6278 !important;
    }

    /* File uploader */

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.025);

        border-radius: 14px;
    }

    /* Chat input — typed text must be visible */

    [data-testid="stChatInput"] {
        border-radius: 16px;
    }

    [data-testid="stChatInput"] textarea {
        color: #000000 !important;
        caret-color: #a5b4fc !important;
        background: rgba(255,255,255,0.08) !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #9aa3b8 !important;
        opacity: 1 !important;
    }

    /* User message bubble — indigo-tinted so it stands out */

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(99,102,241,0.10) !important;
        border: 1px solid rgba(99,102,241,0.22) !important;
    }

    /* Assistant message bubble — subtle green-tinted */

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(255,255,255,0.030) !important;
        border: 1px solid rgba(255,255,255,0.075) !important;
    }

    /* All text inside chat messages — bright white */

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #eef0f6 !important;
    }

    /* Markdown rendered inside messages */

    [data-testid="stChatMessage"] .stMarkdown p,
    [data-testid="stChatMessage"] .stMarkdown li,
    [data-testid="stChatMessage"] .stMarkdown h1,
    [data-testid="stChatMessage"] .stMarkdown h2,
    [data-testid="stChatMessage"] .stMarkdown h3 {
        color: #eef0f6 !important;
    }

    /* Code blocks inside messages */

    [data-testid="stChatMessage"] code {
        color: #c7d2fe !important;
        background: rgba(99,102,241,0.12) !important;
    }

    /* Divider */

    hr {
        border-color: rgba(255,255,255,0.06);
    }

    </style>
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "sources" not in st.session_state:
    st.session_state.sources = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "document_count" not in st.session_state:
    st.session_state.document_count = 0


# ============================================================
# MODELS
# ============================================================

@st.cache_resource
def get_embedding_model():

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )


embedding_model = get_embedding_model()

llm = get_llm()


# ============================================================
# PDF LOADER
# ============================================================

def load_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_type"] = "PDF"
            doc.metadata["source_name"] = uploaded_file.name

            if "page" in doc.metadata:
                doc.metadata["page_number"] = doc.metadata["page"] + 1

        return documents
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# WEBSITE LOADER
# ============================================================

def load_website(url):

    loader = WebBaseLoader(
        web_path=url,
        header_template={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )

    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"] = "Website"
        doc.metadata["source_name"] = url

    return documents


# ============================================================
# CHUNKING
# ============================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


# ============================================================
# VECTOR DATABASE
# ============================================================

def create_vector_database(chunks):

    # Clear any previous in-memory collection
    if st.session_state.vector_db is not None:
        try:
            st.session_state.vector_db.delete_collection()
        except Exception:
            pass
        st.session_state.vector_db = None

    # Use an in-memory (ephemeral) client — no disk folder, no file locks,
    # no schema-corruption issues. The DB lives in st.session_state anyway.
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="rag_documents",
    )

    return vector_db


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="brand">
            <div class="brand-icon">✦</div>
            <div class="brand-name">RAGly</div>
        </div>

        <div class="brand-subtitle">
            AI Knowledge Assistant
        </div>
        """
    )

    st.html('<div class="section-title">Add Sources</div>')

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.html('<div class="section-title">Website URLs</div>')

    url_input = st.text_area(
        "Website URLs",
        placeholder=(
            "Paste one URL per line\n\n"
            "https://example.com\n"
            "https://example.com/about"
        ),
        height=115,
        label_visibility="collapsed",
    )

    process_button = st.button(
        "✦  Process Sources",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.html('<div class="section-title">Your Sources</div>')

    if st.session_state.sources:

        for source in st.session_state.sources:

            if source.startswith("http"):
                icon = "🌐"
            else:
                icon = "📄"

            st.html(
                f"""
                <div class="source-card">
                    <span class="source-icon">
                        {icon}
                    </span>

                    <span class="source-name">
                        {source}
                    </span>
                </div>
                """
            )

    else:

        st.caption("Processed sources will appear here.")

    st.divider()

    if st.session_state.vector_db:

        st.html('<div class="section-title">Knowledge Base</div>')

        col1, col2 = st.columns(2)

        with col1:

            st.html(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {st.session_state.document_count}
                    </div>

                    <div class="metric-label">
                        Sources
                    </div>
                </div>
                """
            )

        with col2:

            st.html(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {st.session_state.chunk_count}
                    </div>

                    <div class="metric-label">
                        Chunks
                    </div>
                </div>
                """
            )

    st.divider()

    if st.button(
        "🗑  Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.rerun()


# ============================================================
# PROCESS SOURCES
# ============================================================

if process_button:

    all_documents = []
    source_names = []

    # --------------------------------------------------------
    # PDFs
    # --------------------------------------------------------

    if uploaded_files:

        with st.spinner("Reading your PDF documents..."):

            for uploaded_file in uploaded_files:

                try:
                    docs = load_pdf(uploaded_file)
                    all_documents.extend(docs)
                    source_names.append(uploaded_file.name)

                except Exception as error:

                    st.error(
                        f"Could not read {uploaded_file.name}: {error}"
                    )

    # --------------------------------------------------------
    # WEBSITES
    # --------------------------------------------------------

    urls = [
        url.strip()
        for url in url_input.splitlines()
        if url.strip()
    ]

    if urls:

        with st.spinner("Reading website content..."):

            for url in urls:

                try:
                    docs = load_website(url)
                    all_documents.extend(docs)
                    source_names.append(url)

                except Exception as error:

                    st.error(
                        f"Could not read {url}: {error}"
                    )

    # --------------------------------------------------------
    # NO SOURCES
    # --------------------------------------------------------

    if not all_documents:

        st.warning(
            "Please upload at least one PDF or enter a website URL."
        )

    else:

        # ----------------------------------------------------
        # CHUNKING
        # ----------------------------------------------------

        with st.spinner("Splitting sources into chunks..."):

            chunks = split_documents(all_documents)

        # ----------------------------------------------------
        # EMBEDDINGS + CHROMA
        # ----------------------------------------------------

        with st.spinner(
            "Creating embeddings and building knowledge base..."
        ):

            vector_db = create_vector_database(chunks)

        # ----------------------------------------------------
        # SAVE STATE
        # ----------------------------------------------------

        st.session_state.vector_db = vector_db
        st.session_state.sources = source_names
        st.session_state.chunk_count = len(chunks)
        st.session_state.document_count = len(source_names)
        st.session_state.messages = []

        st.success(
            f"Knowledge base ready! {len(chunks)} chunks processed."
        )

        st.rerun()


# ============================================================
# HERO
# ============================================================

if not st.session_state.messages:

    st.html(
        """
        <div class="hero">

            <div class="hero-badge">
                ✦ RETRIEVAL-AUGMENTED GENERATION
            </div>

            <h1>
                Ask your knowledge anything.
            </h1>

            <p>
                Upload documents or connect websites.
                RAGly retrieves relevant information
                and uses AI to answer your questions.
            </p>

            <div class="status-card">
                <span class="status-dot"></span>
                AI system online
            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # EMPTY KNOWLEDGE BASE
    # --------------------------------------------------------

    if not st.session_state.vector_db:

        st.html(
            """
            <div class="empty-card">

                <div class="empty-icon">
                    📚
                </div>

                <div class="empty-title">
                    Your knowledge base is empty
                </div>

                <div class="empty-text">
                    Upload a PDF or paste a website
                    URL from the sidebar to create
                    your personal AI knowledge base.
                </div>

            </div>
            """
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.html(
                """
                <div class="feature-card">

                    <div class="feature-icon">
                        📄
                    </div>

                    <div class="feature-title">
                        PDF Knowledge
                    </div>

                    <div class="feature-text">
                        Upload research papers,
                        notes, reports and documents.
                    </div>

                </div>
                """
            )

        with col2:

            st.html(
                """
                <div class="feature-card">

                    <div class="feature-icon">
                        🌐
                    </div>

                    <div class="feature-title">
                        Website Knowledge
                    </div>

                    <div class="feature-text">
                        Connect webpages and ask
                        questions about their content.
                    </div>

                </div>
                """
            )

        with col3:

            st.html(
                """
                <div class="feature-card">

                    <div class="feature-icon">
                        🧠
                    </div>

                    <div class="feature-title">
                        Grounded Answers
                    </div>

                    <div class="feature-text">
                        Answers are generated using
                        retrieved source information.
                    </div>

                </div>
                """
            )

    else:

        st.html(
            """
            <div class="empty-card">

                <div class="empty-icon">
                    ✨
                </div>

                <div class="empty-title">
                    Your knowledge base is ready
                </div>

                <div class="empty-text">
                    Ask anything about your uploaded
                    documents and connected websites.
                </div>

            </div>
            """
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.html("<br><b>📚 Sources used</b>")

            for source in message["sources"]:

                st.html(
                    f"""
                    <div class="result-source">
                        {source}
                    </div>
                    """
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input("Ask a question about your knowledge base...")


# ============================================================
# RAG
# ============================================================

if question:

    if st.session_state.vector_db is None:

        st.warning(
            "Please process at least one source before asking a question."
        )

        st.stop()

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # --------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------

    retriever = st.session_state.vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
        },
    )

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching your knowledge base..."):

            retrieved_docs = retriever.invoke(question)

        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context_parts = []
        source_display = []
        seen_sources = set()

        for index, doc in enumerate(retrieved_docs):

            source = doc.metadata.get(
                "source_name",
                "Unknown source",
            )

            source_type = doc.metadata.get(
                "source_type",
                "",
            )

            page_number = doc.metadata.get("page_number")

            context_parts.append(
                f"""
SOURCE {index + 1}

Source:
{source}

Content:
{doc.page_content}
"""
            )

            if source_type == "PDF":

                if page_number:
                    display = f"📄 {source} · Page {page_number}"
                else:
                    display = f"📄 {source}"

            else:

                display = f"🌐 {source}"

            if display not in seen_sources:

                source_display.append(display)
                seen_sources.add(display)

        context = "\n\n".join(context_parts)

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are RAGly, a helpful Retrieval-Augmented Generation assistant.

Answer the user's question using ONLY the retrieved context below.

Rules:

1. Use only the provided context.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the answer cannot be found in the context, say:

"I could not find the answer in the provided sources."

5. Keep the answer clear and useful.
6. Use bullet points when appropriate.

Retrieved Context:

{context}

User Question:

{question}
"""

        # ----------------------------------------------------
        # GENERATE WITH STREAMING
        # ----------------------------------------------------

        def stream_text(stream):
            """Yield plain-text strings from LLM stream chunks.

            gemini-3.6-flash returns chunk.content as a list of dicts
            e.g. [{'type': 'text', 'text': '...', ...}] rather than a
            plain string.  This helper normalises both formats so that
            st.write_stream always receives str objects.
            """
            for chunk in stream:
                content = chunk.content
                if isinstance(content, str):
                    yield content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                yield text

        try:
            response_stream = llm.stream(prompt)
            answer = st.write_stream(stream_text(response_stream))

        except Exception as error:

            answer = (
                "An error occurred while generating the answer:\n\n"
                f"{error}"
            )
            st.markdown(answer)

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if source_display:

            st.html("<br><b>📚 Sources used</b>")

            for source in source_display:

                st.html(
                    f"""
                    <div class="result-source">
                        {source}
                    </div>
                    """
                )

    # --------------------------------------------------------
    # SAVE MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_display,
        }
    )