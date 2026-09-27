import streamlit as st

from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import BM25Search
from app.retrieval.role_filter import filter_by_role


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Adaptive Enterprise Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD DOCUMENTS
# ============================================================

@st.cache_resource
def initialize_retrieval():

    documents = load_text_documents()

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100
    )

    search_engine = BM25Search(chunks)

    return documents, chunks, search_engine


documents, chunks, search_engine = initialize_retrieval()


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Adaptive Enterprise Knowledge Assistant")

st.markdown(
    "**Hybrid RAG • Role-Aware Retrieval • "
    "Hallucination Detection**"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("👤 User Settings")

    user_role = st.selectbox(
        "Select User Role",
        [
            "Employee",
            "Manager",
            "HR",
            "IT",
            "Admin"
        ]
    )

    st.divider()

    st.subheader("📊 System Status")

    st.success("✓ Document Loading")
    st.success("✓ Metadata Extraction")
    st.success("✓ Document Chunking")
    st.success("✓ BM25 Retrieval")
    st.success("✓ Role Filtering")

    st.warning("⏳ Vector Search")
    st.warning("⏳ Hybrid RAG")
    st.warning("⏳ Ollama LLM")
    st.warning("⏳ Hallucination Detection")

    st.divider()

    st.caption("M.Tech AI/ML Project")
    st.caption("Enterprise Knowledge Assistant")


# ============================================================
# PROJECT METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Documents",
        len(documents)
    )

with col2:
    st.metric(
        "Chunks",
        len(chunks)
    )

with col3:
    st.metric(
        "User Role",
        user_role
    )

with col4:
    st.metric(
        "Retrieval",
        "BM25"
    )


st.divider()


# ============================================================
# QUESTION
# ============================================================

st.subheader("🔎 Ask the Enterprise Assistant")

query = st.text_input(
    "Enter your question",
    placeholder=(
        "Example: How many days can employees "
        "work from home?"
    )
)


# ============================================================
# SEARCH
# ============================================================

if st.button(
    "Search",
    type="primary",
    use_container_width=True
):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ----------------------------------------------------
        # STEP 1: BM25 RETRIEVAL
        # ----------------------------------------------------

        raw_results = search_engine.search(
            query,
            top_k=5
        )


        # ----------------------------------------------------
        # STEP 2: ROLE FILTERING
        # ----------------------------------------------------

        filtered_results = filter_by_role(
            raw_results,
            user_role
        )


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.subheader(
            "📄 Retrieved Evidence"
        )


        if not filtered_results:

            st.error(
                "No authorized documents were found "
                "for this question and user role."
            )

        else:

            # ------------------------------------------------
            # SHOW TOP RESULTS
            # ------------------------------------------------

            for i, result in enumerate(
                filtered_results,
                start=1
            ):

                document = result["document"]

                st.markdown(
                    f"### Result {i}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write(
                        f"**Source:** "
                        f"{document['source']}"
                    )

                with col2:

                    st.write(
                        f"**Department:** "
                        f"{document['department']}"
                    )

                with col3:

                    st.write(
                        f"**Score:** "
                        f"{result['score']:.4f}"
                    )


                st.write(
                    f"**Allowed Roles:** "
                    f"{', '.join(document['allowed_roles'])}"
                )


                st.info(
                    document["text"]
                )


                st.divider()


            # ------------------------------------------------
            # ACCESS STATUS
            # ------------------------------------------------

            st.subheader(
                "🔐 Access Control"
            )

            st.success(
                f"Results filtered for authorized role: "
                f"**{user_role}**"
            )


# ============================================================
# CURRENT PIPELINE
# ============================================================

st.divider()

st.subheader(
    "⚙️ Current Implementation Pipeline"
)

st.code(
"""
Enterprise Documents
        ↓
Document Loader                 ✓
        ↓
Metadata Extraction             ✓
        ↓
Document Chunking               ✓
        ↓
BM25 Keyword Retrieval          ✓
        ↓
Role-Based Filtering            ✓
        ↓
Vector Retrieval                ⏳
        ↓
Hybrid RAG                      ⏳
        ↓
Local LLM / Ollama              ⏳
        ↓
Hallucination Detection         ⏳
        ↓
Answer + Citation + Confidence
""",
    language="text"
)


# ============================================================
# METHODOLOGY PLAN
# ============================================================

st.divider()

st.subheader(
    "📌 Methodology Plan"
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
"""
### Phase 1 — Data Preparation

✓ Document collection  
✓ Text extraction  
✓ Metadata extraction  
✓ Chunking  

### Phase 2 — Retrieval

✓ BM25  
⏳ Embeddings  
⏳ FAISS  
⏳ Hybrid retrieval  
⏳ Reranking
"""
    )


with col2:

    st.markdown(
"""
### Phase 3 — Generation

⏳ Local LLM / Ollama  
⏳ Context-aware answer generation  

### Phase 4 — Verification

⏳ Evidence verification  
⏳ Hallucination detection  
⏳ Confidence scoring  

### Phase 5 — Deployment

⏳ FastAPI  
⏳ Docker  
⏳ GitHub Actions CI/CD
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Adaptive Enterprise Knowledge Assistant | "
    "M.Tech AI/ML Project"
)