from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from src.generator import generate_rag_answer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Supports both locations in case you haven't moved the CSV yet
DATA_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "rag_papers_clean_471.csv",
    PROJECT_ROOT / "rag_papers_clean_471.csv",
]

EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "paper_embeddings.npy"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the cleaned 471-paper dataset."""

    data_path = None

    for path in DATA_PATHS:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(
            "Could not find rag_papers_clean_471.csv"
        )

    df = pd.read_csv(data_path)

    # Fill missing values
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna("")

    print(f"Loaded dataset: {df.shape}")

    return df


# ============================================================
# CREATE PAPER TEXT
# ============================================================

def prepare_text(df):
    """
    Create the text representation used for semantic retrieval.
    """

    if "text" in df.columns:
        text = df["text"].fillna("").astype(str)

    else:
        title = df.get(
            "title",
            pd.Series("", index=df.index)
        ).fillna("").astype(str)

        abstract = df.get(
            "abstract_text",
            pd.Series("", index=df.index)
        ).fillna("").astype(str)

        text = title + " " + abstract

    return text.tolist()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    """
    Sentence Transformer used for semantic retrieval.
    """

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD / CREATE EMBEDDINGS
# ============================================================

def get_embeddings(df, model):
    """
    Load previously saved embeddings if available.
    Otherwise generate and save them.
    """

    EMBEDDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if EMBEDDINGS_PATH.exists():

        embeddings = np.load(
            EMBEDDINGS_PATH
        )

        # Make sure embeddings match dataset size
        if len(embeddings) == len(df):

            print("Loaded saved embeddings.")

            return embeddings

    print("Creating embeddings...")

    paper_text = prepare_text(df)

    embeddings = model.encode(
        paper_text,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    print("Embeddings saved.")

    return embeddings


# ============================================================
# LOAD CROSS-ENCODER
# ============================================================

def load_reranker():
    """
    Cross-Encoder used for final reranking.
    """

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


# ============================================================
# INITIALIZE SYSTEM
# ============================================================

def initialize_system():
    """
    Load dataset, embedding model, embeddings,
    and reranking model.
    """

    df = load_data()

    embedding_model = load_embedding_model()

    embeddings = get_embeddings(
        df,
        embedding_model
    )

    reranker = load_reranker()

    return (
        df,
        embedding_model,
        embeddings,
        reranker
    )


# ============================================================
# SEARCH + RERANK
# ============================================================

def reranked_search(
    query,
    df,
    embedding_model,
    embeddings,
    reranker,
    candidate_k=20,
    top_k=5
):
    """
    Retrieve candidate papers using semantic similarity
    and rerank them using a Cross-Encoder.
    """

    # --------------------------------------------------------
    # STEP 1: Encode query
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    # --------------------------------------------------------
    # STEP 2: Semantic similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    # --------------------------------------------------------
    # STEP 3: Select candidate papers
    # --------------------------------------------------------

    candidate_indices = (
        similarities
        .argsort()[::-1][:candidate_k]
    )

    candidates = df.iloc[
        candidate_indices
    ].copy()

    # --------------------------------------------------------
    # STEP 4: Cross-Encoder reranking
    # --------------------------------------------------------

    paper_text = prepare_text(candidates)

    pairs = [
        [query, text]
        for text in paper_text
    ]

    rerank_scores = reranker.predict(
        pairs
    )

    candidates["rerank_score"] = rerank_scores

    # --------------------------------------------------------
    # STEP 5: Sort by reranking score
    # --------------------------------------------------------

    candidates = candidates.sort_values(
        "rerank_score",
        ascending=False
    )

    # --------------------------------------------------------
    # STEP 6: Return top results
    # --------------------------------------------------------

    return candidates.head(top_k).reset_index(
        drop=True
    )

# ============================================================
# SIMPLE TEST + LLM GENERATION
# ============================================================

if __name__ == "__main__":

    print("\nInitializing Research Paper Recommender...\n")

    (
        df,
        embedding_model,
        embeddings,
        reranker
    ) = initialize_system()

    query = (
        "RAG applications in healthcare "
        "and medical diagnosis"
    )

    # --------------------------------------------------------
    # RETRIEVAL + RERANKING
    # --------------------------------------------------------

    results = reranked_search(
        query=query,
        df=df,
        embedding_model=embedding_model,
        embeddings=embeddings,
        reranker=reranker,
        candidate_k=20,
        top_k=5
    )

    # --------------------------------------------------------
    # DISPLAY PAPERS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RECOMMENDED PAPERS")
    print("=" * 70)

    columns = [
        "title",
        "year",
        "citations",
        "rerank_score"
    ]

    columns = [
        col
        for col in columns
        if col in results.columns
    ]

    print(
        results[columns].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # LLM GENERATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING RESEARCH ANSWER...")
    print("=" * 70)

    answer = generate_rag_answer(
        query=query,
        papers=results
    )

    print("\n" + "=" * 70)
    print("RAG GENERATED ANSWER")
    print("=" * 70)

    print(answer)

