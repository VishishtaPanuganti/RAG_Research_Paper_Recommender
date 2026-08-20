# RAG Research Paper Recommender — Project Report

## 1. Project Overview

**Project:** RAG Research Paper Recommender

**Goal:** Build a research-paper recommendation system that accepts a research topic, retrieves relevant papers using semantic similarity, reranks them with a Cross-Encoder, provides domain information from clustering, and generates a grounded research answer using a local LLM.

The project evolved from exploratory NLP/ML notebook experiments into a working end-to-end Streamlit application.

### Final pipeline

```text
Research papers
      ↓
Data cleaning + duplicate removal
      ↓
Text / abstract preparation
      ↓
Sentence embeddings
      ↓
PCA + clustering experiments
      ↓
Semantic retrieval
      ↓
Top candidate papers
      ↓
Cross-Encoder reranking
      ↓
Top-K papers
      ↓
Local LLM (Llama 3.2 3B via Ollama)
      ↓
Grounded research answer
      ↓
Streamlit application
```

---

# 2. Dataset Preparation

The original dataset contained:

```text
Original shape: (500, 11)
```

Duplicate removal produced:

```text
After duplicate removal: (473, 11)
Rows removed: 27
```

After additional cleaning/processing, the final usable corpus became:

```text
Final dataset: 471 papers
```

Important fields included:

```text
title
year
authors
citations
doi
paper_url
abstract
abstract_text
text
```

Additional recommendation metadata included cluster fields such as `cluster`, `cluster_name`, `kmeans_cluster`, `agg_cluster`, and `pca_cluster`.

---

# 3. NLP / Text Representation

The paper information was converted into text suitable for semantic retrieval.

Sentence Transformers were used to create dense embeddings. Important notebook objects included:

```text
model
embeddings
clean_embeddings
embeddings_2d
embeddings_pca50
loaded_embeddings
```

The embeddings allow papers and queries to be compared using cosine similarity instead of relying only on exact keyword matching.

The final embedding model is:

```text
all-MiniLM-L6-v2
```

---

# 4. Clustering Experiments

Clustering was explored to discover semantic groups in the research-paper corpus. It was ultimately used for organization/domain labeling rather than as the main ranking mechanism.

## 4.1 HDBSCAN

One HDBSCAN experiment produced:

```text
Number of clusters: 2
Noise points: 445
HDBSCAN Silhouette Score: 0.190952
```

At that stage there were 473 papers, so approximately 94.08% were classified as noise.

Several parameter combinations were tested. Representative results included:

| min_cluster_size | min_samples | clusters | noise | noise % | silhouette |
|---:|---:|---:|---:|---:|---:|
| 8 | 5 | 2 | 445 | 94.08% | 0.190952 |
| 10 | 5 | 2 | 445 | 94.08% | 0.190952 |
| 20 | 3 | 2 | 411 | 86.89% | 0.139368 |
| 10 | 2 | 3 | 390 | 82.45% | 0.136770 |
| 5 | 2 | 2 | 88 | 18.60% | 0.121379 |
| 8 | 2 | 2 | 185 | 39.11% | 0.054020 |

**Conclusion:** A high silhouette score alone was not sufficient. Excessive noise made HDBSCAN unsuitable as the primary clustering solution.

---

# 5. PCA Experiments

PCA was used to reduce embedding dimensionality.

Final PCA representation:

```text
Shape: (471, 50)
Variance explained: 0.72644365
```

Therefore, the first 50 components retained approximately **72.64% of the variance**.

Earlier experimentation with 473 papers gave approximately the same result:

```text
(473, 50)
Variance explained: 0.7260155
```

---

# 6. K-Means + PCA Experiments

K-Means was tested on PCA-50 embeddings.

| K | Silhouette |
|---:|---:|
| 2 | 0.086478 |
| 3 | 0.080669 |
| 4 | 0.091598 |
| 5 | 0.094722 |
| 6 | 0.094947 |
| 7 | 0.075267 |
| 8 | 0.064543 |
| 9 | 0.068965 |
| 10 | 0.067420 |

An earlier experiment found:

```text
Best K = 6
Silhouette = 0.094947
```

A later final configuration used:

```text
Best K = 5
Silhouette = 0.09655945003032684
```

The five interpreted domains were:

1. Education, Conversational & Agentic RAG
2. Healthcare & Biomedical RAG
3. Core RAG & Retrieval Methods
4. RAG Evaluation & Advanced Techniques
5. Industry, Cybersecurity & Code RAG

The silhouette scores were relatively low, showing that the embedding space did not contain extremely clean clusters. However, the clusters were still semantically useful for organizing the corpus and displaying domain labels.

---

# 7. Semantic Retrieval

Semantic retrieval was implemented using query embeddings and cosine similarity.

Example query:

```text
RAG applications in healthcare and medical diagnosis
```

The retrieval stage:

1. Encodes the query.
2. Computes cosine similarity against all paper embeddings.
3. Selects a candidate pool.
4. Sends the candidates to the Cross-Encoder.

Default candidate pool:

```text
candidate_k = 20
```

---

# 8. Multiple Query Tests

Several research topics were tested to check whether retrieval respected different domains.

### Healthcare

```text
RAG for healthcare and medical diagnosis
```

Example distribution:

```text
Healthcare RAG: 17
Specialized & Industry RAG: 2
Core RAG & Retrieval: 1
```

### Cybersecurity

```text
RAG for cybersecurity
```

Example distribution:

```text
Specialized & Industry RAG: 16
Education / Conversational / Agentic RAG: 2
Core RAG & Retrieval: 2
```

### Education

```text
RAG applications in education
```

Example distribution:

```text
Education / Conversational / Agentic RAG: 9
Healthcare RAG: 6
Core RAG & Retrieval: 4
Specialized & Industry RAG: 1
```

### Evaluation

```text
evaluation methods for RAG systems
```

Example distribution:

```text
Core RAG & Retrieval: 9
Healthcare RAG: 8
Specialized & Industry RAG: 2
Education / Conversational / Agentic RAG: 1
```

### Knowledge Graph RAG

```text
knowledge graph based retrieval augmented generation
```

Example distribution:

```text
Evaluation & Advanced RAG: 15
Core RAG & Retrieval: 5
```

These tests showed that semantic retrieval could distinguish important RAG research areas.

---

# 9. Cross-Encoder Reranking

The major ranking improvement was Cross-Encoder reranking.

Final ranking pipeline:

```text
Query
  ↓
Embedding retrieval
  ↓
Top 20 candidate papers
  ↓
Cross-Encoder scoring
  ↓
Sort by rerank score
  ↓
Top 5 recommendations
```

Model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The rerank score is a ranking signal and is **not a probability or percentage**.

For the healthcare query, the final recommendations included:

1. Enhancing medical AI with retrieval-augmented generation: A mini narrative review — 2025, 79 citations, score 7.470868
2. Retrieval-Augmented Generation (RAG) in Healthcare: A Comprehensive Review — 2025, 60 citations, score 6.281657
3. Improving Large Language Model Applications in the Medical and Nursing Domains With Retrieval-Augmented Generation: Scoping Review — 2025, 22 citations, score 4.167648
4. Bridging AI and Healthcare: A Scoping Review of Retrieval-Augmented Generation—Ethics, Bias, Transparency, Improvements, and Applications — 2025, 15 citations, score 4.023738
5. Evidence-based artificial intelligence: Implementing retrieval-augmented generation models to enhance clinical decision support in plastic surgery — 2025, 26 citations, score 3.269176

All five were in the `Healthcare & Biomedical RAG` domain.

---

# 10. Retrieval Evaluation

The recommendation system was evaluated using:

- Precision@5
- MRR
- nDCG@5

## Original reranking evaluation

| Query | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|
| Healthcare & Biomedical RAG | 1.0 | 1.0 | 1.000000 |
| Industry, Cybersecurity & Code RAG | 1.0 | 1.0 | 1.000000 |
| Education, Conversational & Agentic RAG | 0.8 | 1.0 | 0.982892 |
| RAG Evaluation & Advanced Techniques | 1.0 | 1.0 | 1.000000 |
| Core RAG & Retrieval Methods | 1.0 | 1.0 | 1.000000 |

Mean:

```text
Precision@5 = 0.96
MRR         = 1.00
nDCG@5      = 0.996578
```

These scores were obtained on manually defined evaluation queries and relevance labels. They should not be presented as universal performance guarantees.

An earlier evaluation configuration produced:

```text
Mean Precision@5 = 0.88
```

Both results are part of the experiment history.

---

# 11. Query Expansion Experiment

Query expansion was tested to see whether related terms would improve retrieval.

Example:

```text
Original query:
RAG for medical diagnosis
```

Expansion terms included:

```text
medical
healthcare
rag
ai
clinical
llms
applications
diagnostic
```

The expanded query combined the original query with related terms.

## Query expansion evaluation

| Query | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|
| Healthcare & Biomedical RAG | 1.0 | 1.0 | 1.000000 |
| Industry, Cybersecurity & Code RAG | 1.0 | 1.0 | 1.000000 |
| Education, Conversational & Agentic RAG | 0.8 | 1.0 | 0.982892 |
| RAG Evaluation & Advanced Techniques | 0.0 | 0.0 | 0.000000 |
| Core RAG & Retrieval Methods | 0.6 | 0.5 | 0.732829 |

Mean:

```text
Precision@5 = 0.68
MRR         = 0.70
nDCG@5      = 0.743144
```

### Conclusion

Query expansion performed worse than the original semantic retrieval + Cross-Encoder reranking pipeline. Therefore it was kept as an experiment but **not selected as the default method**.

This demonstrated the importance of evaluating an additional technique instead of assuming that more query terms will improve retrieval.

---

# 12. Experiments That Did Not Work Well

### HDBSCAN

The main problem was excessive noise. An example configuration classified 445 of 473 papers as noise (94.08%). It was therefore not selected as the primary clustering approach.

### K-Means

Silhouette scores were only around 0.08–0.10. Nevertheless, the clusters were semantically interpretable and useful for domain labels.

### Query expansion

Performance decreased to:

```text
Precision@5 = 0.68
MRR         = 0.70
nDCG@5      = 0.743144
```

The original query + retrieval + reranking approach was therefore preferred.

---

# 13. Coding Mistakes and Debugging History

## Error 1 — `model` not defined

An early reranking call produced:

```text
NameError: name 'model' is not defined
```

Cause: the embedding model variable was not available in the current notebook/kernel state.

Fix: reload/reinitialize the embedding model before searching.

## Error 2 — `embedding_model` not defined

Query expansion initially produced:

```text
NameError: name 'embedding_model' is not defined
```

Cause: a different variable name was used from the currently initialized model.

Fix: initialize and use the correct embedding model.

## Error 3 — Empty results

At one point the retrieval function returned:

```text
[]
```

The model and embedding state were checked and restored.

## Error 4 — Notebook state dependency

The NLP notebook contained many variations and experiments. Some variables existed only after specific cells had been executed.

Lesson: final application code should explicitly initialize its dependencies instead of relying on notebook execution order.

This led to moving the final recommendation logic into `src/recommender.py`.

---

# 14. Transition from Notebook to Python

The final recommendation pipeline was moved into Python scripts.

Main source files:

```text
src/
├── recommender.py
└── generator.py
```

`recommender.py` handles:

- dataset loading,
- embedding model loading,
- embedding generation/loading,
- semantic retrieval,
- candidate selection,
- Cross-Encoder reranking,
- Top-K recommendations.

Saved embeddings are reused when they match the dataset size.

The recommender was successfully tested with:

```powershell
python src\recommender.py
```

It loaded:

```text
Dataset: (471, 11)
```

and generated recommendations successfully.

---

# 15. LLM Generation / RAG Stage

After the retrieval and reranking system was working, an LLM generation stage was added.

The architecture became:

```text
User query
    ↓
Semantic retrieval
    ↓
Cross-Encoder reranking
    ↓
Top-K papers
    ↓
Paper context
    ↓
LLM
    ↓
Generated research answer
```

The generation code was placed in:

```text
src/generator.py
```

---

# 16. OpenAI API Experiment

The OpenAI Python package was initially used for generation.

Installed package:

```text
openai 3.3.1
```

There were environment-variable/credential issues during setup. After correcting the environment configuration, the API could authenticate and list available models.

Generation then returned:

```text
429 insufficient_quota
```

with a message indicating that the current API quota/billing limit had been exceeded.

### Decision

The project was switched to a local LLM using Ollama instead of depending on paid API quota.

---

# 17. Ollama / Local LLM

Ollama was tested locally with:

```text
llama3.2:3b
```

An initial standalone test was:

```text
Explain RAG in one sentence.
```

The model incorrectly interpreted RAG as "Reverse Auction". This demonstrated that the model needed clear domain context and prompting.

After integrating the retrieved research-paper context, the generator successfully produced a research-oriented answer.

The local setup now uses:

```text
Llama 3.2 3B
Ollama
```

---

# 18. Final LLM Generation Test

For the query:

```text
RAG applications in healthcare and medical diagnosis
```

the system generated a research answer using the retrieved papers.

The generated response discussed RAG in medical AI, healthcare applications, patient data analysis, medical literature review, clinical decision support, and potential medical diagnosis applications.

The important design principle is that the answer is generated from retrieved paper context rather than functioning as an unrestricted chatbot.

---

# 19. Streamlit Application

A Streamlit interface was created for the final recommender.

The application provides:

- research query input,
- number of recommendations,
- candidate-pool size,
- semantic retrieval,
- Cross-Encoder reranking,
- generated LLM answer,
- paper title,
- year,
- citations,
- reranking score,
- domain/cluster,
- authors,
- DOI,
- Open Paper button.

The Open Paper functionality was manually tested and confirmed to work.

For the healthcare query, the UI successfully displayed:

```text
Found 5 recommended papers.
```

---

# 20. Streamlit + LLM Integration

The Streamlit application now performs the complete pipeline:

```text
User enters topic
        ↓
reranked_search()
        ↓
Top-K research papers
        ↓
generate_rag_answer()
        ↓
Ollama / Llama 3.2 3B
        ↓
Research Assistant Answer
        ↓
Recommended Research Papers
```

The complete Streamlit application was tested successfully.

Local LLM generation takes longer than retrieval because the model is running locally, especially on CPU-based hardware. This is expected.

---

# 21. Streamlit Import Issue

When connecting the generator to the recommender, this import was initially used:

```python
from generator import generate_rag_answer
```

This worked when directly running `src/recommender.py`, but failed when Streamlit imported the module as part of the `src` package.

The import was corrected to:

```python
from src.generator import generate_rag_answer
```

After the correction, Streamlit worked successfully.

**Lesson:** imports must match the project package structure and execution method.

---

# 22. Requirements

A `requirements.txt` file was created for the Python dependencies used by the project.

Main packages include:

```text
numpy
pandas
scikit-learn
sentence-transformers
streamlit
openai
```

Ollama is a local model runtime/application rather than a normal Python package dependency.

The virtual environment is local development infrastructure and should not be uploaded to GitHub.

---

# 23. Git / Repository Preparation

Before GitHub upload, Git was accidentally operating at an incorrect directory level. `git status` showed unrelated Windows folders such as:

```text
AppData/
Documents/
Downloads/
.ssh/
.vscode/
```

This is dangerous because those files are unrelated to the project and may contain private information.

The correct project root is:

```text
C:\Users\pvish\OneDrive\Desktop\P
```

Only project files should be tracked.

Recommended project structure:

```text
RAG-Research-Paper-Recommender/
├── data/
├── notebooks/
├── src/
├── app.py
├── README.md
├── PROJECT_REPORT.md
├── requirements.txt
└── .gitignore
```

The virtual environment, caches, secrets, API keys, and unrelated personal files must not be committed.

---

# 24. Security Notes

An OpenAI API key was used during development.

API keys must never be placed in:

- Python source files,
- notebooks,
- README files,
- project reports,
- Git commits,
- GitHub repositories.

Use environment variables or a secure secret mechanism instead.

If a real API key is accidentally exposed, it should be revoked/rotated.

The final working application uses Ollama locally and therefore does not depend on OpenAI API quota.

---

# 25. Final Pipeline Selection

### Experiments retained for analysis

- HDBSCAN parameter search
- K-Means parameter search
- PCA experiments
- Different cluster counts
- Multiple research queries
- Query expansion
- Retrieval evaluation
- OpenAI generation experiment
- Ollama generation experiment
- Notebook debugging and variable-state fixes

### Selected final pipeline

```text
all-MiniLM-L6-v2
        ↓
Cosine Similarity Retrieval
        ↓
Top 20 Candidates
        ↓
Cross-Encoder Reranking
        ↓
Top-K Recommendations
        ↓
Cluster / Domain Metadata
        ↓
Llama 3.2 3B via Ollama
        ↓
Streamlit UI
```

Query expansion is not part of the default ranking pipeline because it reduced evaluation performance.

---

# 26. Current Project Status

## Completed

- [x] Collected research papers
- [x] Removed duplicate papers
- [x] Cleaned the dataset
- [x] Finalized 471-paper corpus
- [x] Generated sentence embeddings
- [x] Saved/loaded embeddings
- [x] Tested HDBSCAN
- [x] Tested K-Means
- [x] Tested PCA
- [x] Compared different K values
- [x] Interpreted semantic clusters
- [x] Created meaningful cluster names
- [x] Implemented semantic retrieval
- [x] Implemented Cross-Encoder reranking
- [x] Tested multiple research queries
- [x] Evaluated Precision@5
- [x] Evaluated MRR
- [x] Evaluated nDCG@5
- [x] Tested query expansion
- [x] Compared query expansion with the original approach
- [x] Moved final recommender logic from notebook to Python
- [x] Created `src/recommender.py`
- [x] Created `src/generator.py`
- [x] Tested OpenAI generation
- [x] Diagnosed OpenAI quota limitation
- [x] Switched to local Ollama
- [x] Tested Llama 3.2 3B
- [x] Connected retrieved papers to LLM generation
- [x] Created Streamlit application
- [x] Connected Streamlit to recommender
- [x] Connected Streamlit to LLM generator
- [x] Fixed Python package import issue
- [x] Tested Open Paper functionality
- [x] Created `requirements.txt`
- [x] Verified complete end-to-end application

### Current best practical approach

```text
Semantic Embedding Retrieval
          +
Cross-Encoder Reranking
          +
Cluster/domain information
          +
Llama 3.2 3B local generation
          ↓
RAG Research Assistant
          +
Top-K Research Paper Recommendations
```

---

# 27. Key Findings

1. Duplicate removal reduced the corpus from 500 to 473 papers, followed by further cleaning to a final 471-paper corpus.
2. PCA with 50 components retained approximately 72.64% variance.
3. HDBSCAN produced excessive noise in several configurations and was not selected as the primary clustering method.
4. K-Means produced relatively low silhouette scores (~0.09–0.10), but the clusters were still semantically interpretable.
5. Cross-Encoder reranking produced strong practical recommendations and became the main ranking method.
6. The main evaluation achieved Mean Precision@5 = 0.96, Mean MRR = 1.00, and Mean nDCG@5 = 0.996578 on the manually defined evaluation set.
7. Query expansion reduced performance to Precision@5 = 0.68, MRR = 0.70, and nDCG@5 = 0.743144, so it was not selected as default.
8. OpenAI generation was blocked by insufficient API quota, so the project moved to local Llama 3.2 3B through Ollama.
9. The final Streamlit application successfully performs retrieval, reranking, LLM generation, and paper display in one workflow.

---

# 28. Lessons Learned

1. A high clustering silhouette score does not automatically mean a better practical clustering solution.
2. Noise percentage matters for density-based clustering.
3. PCA is useful for dimensionality reduction and visualization, but can affect clustering quality.
4. Semantic retrieval and reranking serve different purposes.
5. Candidate retrieval followed by Cross-Encoder reranking is effective for final recommendations.
6. Query expansion must be evaluated empirically because additional terms can introduce irrelevant concepts.
7. Notebook execution order can create hidden state dependencies.
8. Production code should explicitly load models and data.
9. Evaluation metrics are meaningful relative to the chosen queries and relevance labels.
10. Failed experiments are valuable because they justify the final architecture.
11. Local LLMs can remove dependence on paid API quotas for a prototype.
12. Package imports must match the project's execution structure.
13. API keys and secrets must never be committed to GitHub.

---

# 29. Future Improvements

Possible future work:

- Add a larger and more diverse paper corpus.
- Improve metadata quality.
- Add abstract-based explanations for recommendations.
- Add publication-year and citation filters.
- Add domain filtering.
- Add user feedback such as relevant/not relevant.
- Add hybrid keyword + semantic retrieval.
- Experiment with stronger embedding models.
- Evaluate additional Cross-Encoder models.
- Create a larger manually labeled evaluation set.
- Improve local LLM generation speed.
- Add streaming LLM output.
- Add automated tests.
- Improve UI/UX.
- Deploy the Streamlit application publicly.

---

# 30. Final Architecture

```text
                    ┌────────────────────────┐
                    │ Research Paper Data    │
                    │       471 papers       │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Cleaning + Deduplication│
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Sentence Embeddings    │
                    │ all-MiniLM-L6-v2       │
                    └───────────┬────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
        ┌─────────────────┐         ┌──────────────────┐
        │ PCA + Clustering│         │ Semantic Retrieval│
        │ Domain Analysis │         │ Cosine Similarity │
        └────────┬────────┘         └────────┬─────────┘
                 │                           │
                 │                           ▼
                 │                  ┌──────────────────┐
                 │                  │ Top 20 Candidates│
                 │                  └────────┬─────────┘
                 │                           │
                 │                           ▼
                 │                  ┌──────────────────┐
                 │                  │ Cross-Encoder    │
                 │                  │ Reranking         │
                 │                  └────────┬─────────┘
                 │                           │
                 └──────────────┬────────────┘
                                ▼
                     ┌──────────────────────┐
                     │ Top-K Research Papers│
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Llama 3.2 3B         │
                     │ Ollama / Local LLM   │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Grounded RAG Answer   │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Streamlit Application │
                     └──────────────────────┘
```

---

# 31. Conclusion

The project progressed from exploratory NLP and clustering experiments into a working end-to-end RAG research-paper recommendation system.

The experiments were important because multiple approaches were tested rather than assuming the first method would work. HDBSCAN produced excessive noise, K-Means produced relatively weak but interpretable clusters, PCA retained approximately 72.64% variance with 50 components, and query expansion reduced performance on the evaluation set.

The final practical system therefore uses semantic embedding retrieval followed by Cross-Encoder reranking, with clustering used for research-domain organization and explanation.

A local Llama 3.2 3B model running through Ollama was integrated as the generation component. The retrieved papers are supplied as context so that the system can generate a research-oriented answer alongside the recommended papers.

The final Streamlit application is working end-to-end:

```text
User query
    ↓
Semantic retrieval
    ↓
Cross-Encoder reranking
    ↓
Top-K papers
    ↓
Llama 3.2 3B via Ollama
    ↓
Generated research answer
    ↓
Paper recommendations + metadata + open-paper links
```

**Current status: WORKING END-TO-END.**

---

# Appendix — Important Experimental Numbers

```text
Initial dataset                  : 500 rows
After duplicate removal          : 473 rows
Final cleaned corpus             : 471 papers

Embedding model                  : all-MiniLM-L6-v2
Reranker                         : cross-encoder/ms-marco-MiniLM-L-6-v2

PCA components                   : 50
Variance explained               : 0.72644365 (~72.64%)

HDBSCAN example:
Clusters                         : 2
Noise points                     : 445
Noise percentage                 : 94.08%
Silhouette                       : 0.190952

K-Means PCA experiment:
Best K (earlier experiment)      : 6
Silhouette                       : 0.094947

Final K-Means experiment:
Best K                           : 5
Silhouette                       : 0.096559

Earlier Precision@5              : 0.88

Final reranking evaluation:
Mean Precision@5                 : 0.96
Mean MRR                         : 1.00
Mean nDCG@5                      : 0.996578

Query expansion evaluation:
Mean Precision@5                 : 0.68
Mean MRR                         : 0.70
Mean nDCG@5                      : 0.743144

Candidate retrieval              : 20 papers
Default recommendations          : 5 papers

Local LLM                        : Llama 3.2 3B
LLM runtime                      : Ollama
Interface                        : Streamlit
```

# Appendix — Current Project Structure

```text
RAG-Research-Paper-Recommender/
├── data/
│   └── processed/
│       ├── paper_embeddings.npy
│       └── rag_papers_clean_471.csv
│
├── notebooks/
│   └── NLP / exploratory notebooks
│
├── src/
│   ├── recommender.py
│   └── generator.py
│
├── app.py
├── README.md
├── PROJECT_REPORT.md
├── requirements.txt
└── .gitignore
```

The virtual environment and generated caches should remain excluded from Git.

**Final status: retrieval → reranking → local LLM generation → Streamlit UI is fully working.**
