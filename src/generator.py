import ollama


def generate_rag_answer(query, papers):

    context = ""

    for i, paper in papers.iterrows():
        context += f"""
Paper {i + 1}
Title: {paper['title']}
Year: {paper['year']}
Authors: {paper['authors']}
Abstract: {paper['abstract_text']}
DOI: {paper['doi']}

"""

    prompt = f"""
You are a research assistant specializing in Artificial Intelligence.

IMPORTANT:
RAG means Retrieval-Augmented Generation.
It does NOT mean Reverse Auction.

Answer the user's research question using ONLY the research
papers provided below.

User question:
{query}

Retrieved research papers:
{context}

Instructions:
- Explain the answer clearly.
- Base your answer on the retrieved papers.
- Mention relevant paper titles when appropriate.
- Do not invent facts or references.
- If the papers do not provide enough information, say so.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

