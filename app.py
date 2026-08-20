import streamlit as st

from src.recommender import (
    initialize_system,
    reranked_search
)

from src.generator import generate_rag_answer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Research Paper Recommender",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 30px;
    }

    .paper-title {
        font-size: 21px;
        font-weight: 600;
    }

    .score {
        font-size: 16px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 RAG Research Paper Recommender</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Find relevant research papers using semantic retrieval, '
    'Cross-Encoder reranking, and local LLM generation.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD SYSTEM
# ============================================================

@st.cache_resource
def load_system():

    return initialize_system()


with st.spinner("Loading recommendation system..."):

    df, embedding_model, embeddings, reranker = load_system()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Search Settings")

    top_k = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    candidate_k = st.slider(
        "Candidate papers",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    )

    st.divider()

    st.write("### 📊 Dataset")

    st.write(
        f"**Papers:** {len(df)}"
    )

    st.write(
        "**Retrieval:** Semantic Similarity"
    )

    st.write(
        "**Reranking:** Cross-Encoder"
    )

    st.write(
        "**Generation:** Llama 3.2 3B"
    )

    st.write(
        "**LLM:** Local / Ollama"
    )


# ============================================================
# SEARCH BOX
# ============================================================

query = st.text_input(
    "🔎 Enter your research topic",
    placeholder="Example: RAG applications in healthcare and medical diagnosis"
)


search = st.button(
    "🔍 Find Papers",
    type="primary",
    use_container_width=True
)


# ============================================================
# SEARCH
# ============================================================

if search:

    if not query.strip():

        st.warning(
            "Please enter a research topic."
        )

    else:

        # ====================================================
        # RETRIEVAL + RERANKING
        # ====================================================

        with st.spinner(
            "Searching and reranking papers..."
        ):

            results = reranked_search(
                query=query,
                df=df,
                embedding_model=embedding_model,
                embeddings=embeddings,
                reranker=reranker,
                candidate_k=candidate_k,
                top_k=top_k
            )

        st.success(
            f"Found {len(results)} recommended papers."
        )


        # ====================================================
        # LLM GENERATED ANSWER
        # ====================================================

        st.divider()

        st.subheader("🤖 Research Assistant Answer")

        with st.spinner(
            "Generating answer using Llama 3.2..."
        ):

            try:

                answer = generate_rag_answer(
                    query=query,
                    papers=results
                )

                st.markdown(answer)

            except Exception as e:

                st.error(
                    "Could not generate the LLM answer."
                )

                st.caption(
                    f"Error: {e}"
                )


        # ====================================================
        # RECOMMENDED PAPERS
        # ====================================================

        st.divider()

        st.subheader("📚 Recommended Research Papers")


        for i, (_, paper) in enumerate(
            results.iterrows(),
            start=1
        ):

            st.markdown(
                f'<div class="paper-title">'
                f'{i}. {paper["title"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.write(
                    f"📅 **Year:** {paper['year']}"
                )

            with col2:

                st.write(
                    f"📚 **Citations:** {paper['citations']}"
                )

            with col3:

                st.markdown(
                    f'<div class="score">'
                    f'🎯 Score: {paper["rerank_score"]:.3f}'
                    f'</div>',
                    unsafe_allow_html=True
                )


            # =================================================
            # CLUSTER / DOMAIN
            # =================================================

            if "cluster_name" in paper:

                cluster = paper["cluster_name"]

                if cluster:

                    st.write(
                        f"🏷️ **Domain:** {cluster}"
                    )


            # =================================================
            # AUTHORS
            # =================================================

            if "authors" in paper:

                authors = paper["authors"]

                if authors:

                    st.write(
                        f"👥 **Authors:** {authors}"
                    )


            # =================================================
            # DOI
            # =================================================

            if "doi" in paper:

                doi = paper["doi"]

                if doi:

                    st.write(
                        f"🔗 **DOI:** {doi}"
                    )


            # =================================================
            # PAPER URL
            # =================================================

            if "paper_url" in paper:

                url = paper["paper_url"]

                if url:

                    st.link_button(
                        "📄 Open Paper",
                        url
                    )


            st.divider()