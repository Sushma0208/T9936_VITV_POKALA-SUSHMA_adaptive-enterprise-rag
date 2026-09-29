import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Adaptive Enterprise Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)


API_URL = "http://localhost:8000"


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Adaptive Enterprise Knowledge Assistant")

st.markdown(
    "**Hybrid RAG • Role-Aware Retrieval • "
    "Evidence-Based Hallucination Detection**"
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

    st.success("✓ FastAPI Backend")
    st.success("✓ Hybrid Retrieval")
    st.success("✓ Role-Aware Filtering")
    st.success("✓ Ollama LLM")
    st.success("✓ Evidence Verification")

    st.divider()

    st.caption("M.Tech AI/ML Project")
    st.caption("Enterprise Knowledge Assistant")


# ============================================================
# PROJECT METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Retrieval",
        "Hybrid"
    )

with col2:
    st.metric(
        "LLM",
        "Llama 3.2"
    )

with col3:
    st.metric(
        "Embedding",
        "Nomic"
    )

with col4:
    st.metric(
        "User Role",
        user_role
    )


st.divider()


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("🔎 Ask the Enterprise Assistant")

query = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: How many days can employees "
        "work from home?"
    ),
    height=100
)


# ============================================================
# ASK ASSISTANT
# ============================================================

if st.button(
    "🤖 Ask Assistant",
    type="primary",
    use_container_width=True
):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching enterprise knowledge..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": query,
                        "user_role": user_role
                    },
                    timeout=180
                )

                response.raise_for_status()

                result = response.json()

            except requests.exceptions.RequestException as error:

                st.error(
                    "Unable to connect to the FastAPI backend."
                )

                st.code(str(error))

                st.stop()


        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader("💬 Answer")

        answer = result.get(
            "answer",
            "No answer returned."
        )

        if answer == (
            "Insufficient evidence in the available documents."
        ):

            st.warning(answer)

        else:

            st.success(answer)


        # ====================================================
        # VERIFICATION
        # ====================================================

        verification = result.get(
            "verification",
            {}
        )

        supported = verification.get(
            "supported",
            False
        )

        confidence = verification.get(
            "confidence",
            0.0
        )

        reason = verification.get(
            "reason",
            "No verification information available."
        )


        st.subheader("🛡️ Evidence Verification")

        col1, col2 = st.columns(2)

        with col1:

            if supported:

                st.success(
                    "✓ Evidence Supported"
                )

            else:

                st.error(
                    "✗ Evidence Not Supported"
                )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.1f}%"
            )

        st.caption(
            f"Verification: {reason}"
        )


        # ====================================================
        # SOURCES
        # ====================================================

        st.subheader("📚 Retrieved Sources")

        sources = result.get(
            "sources",
            []
        )

        if not sources:

            st.info(
                "No authorized sources were retrieved."
            )

        else:

            for index, source in enumerate(
                sources,
                start=1
            ):

                document = source.get(
                    "document",
                    {}
                )

                source_name = document.get(
                    "source",
                    "Unknown"
                )

                organization = document.get(
                    "organization",
                    "Unknown"
                )

                document_type = document.get(
                    "document_type",
                    "Unknown"
                )

                page_number = document.get(
                    "page_number",
                    "N/A"
                )

                access_type = document.get(
                    "access_type",
                    "Unknown"
                )

                score = source.get(
                    "score",
                    0.0
                )

                with st.expander(
                    f"{index}. {organization} — "
                    f"{document_type}"
                ):

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.write(
                            f"**Source:** {source_name}"
                        )

                    with col2:

                        st.write(
                            f"**Page:** {page_number}"
                        )

                    with col3:

                        st.write(
                            f"**Hybrid Score:** "
                            f"{score:.4f}"
                        )

                    st.write(
                        f"**Access:** {access_type}"
                    )

                    st.info(
                        document.get(
                            "text",
                            ""
                        )
                    )


# ============================================================
# ARCHITECTURE
# ============================================================

st.divider()

st.subheader("🏗️ System Architecture")

st.code(
"""
Enterprise Documents
        ↓
Document Processing
        ↓
Hybrid Retrieval
   ↙           ↘
BM25         Vector
   ↘           ↙
    Result Fusion
          ↓
   Role-Based Filter
          ↓
       Reranking
          ↓
         LLM
          ↓
 Evidence Verification
     ↙           ↘
Supported      Unsupported
     ↓             ↓
Answer +       Insufficient
Sources        Evidence
""",
    language="text"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Adaptive Enterprise Knowledge Assistant | "
    "M.Tech AI/ML Project"
)