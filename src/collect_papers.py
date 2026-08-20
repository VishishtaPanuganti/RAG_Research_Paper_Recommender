import requests
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

topic = "retrieval augmented generation"
url = "https://api.openalex.org/works"

total_papers = 500
per_page = 100

# -----------------------------
# Collect papers
# -----------------------------

paper_data = []

for page in range(1, (total_papers // per_page) + 1):

    print(f"Collecting page {page}...")

    params = {
        "search": topic,
        "per-page": per_page,
        "page": page
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    for paper in data["results"]:

        authors = []

        for author in paper["authorships"]:
            author_name = author["author"]["display_name"]
            authors.append(author_name)

        paper_data.append({
            "title": paper["title"],
            "year": paper["publication_year"],
            "authors": ", ".join(authors),
            "citations": paper["cited_by_count"],
            "doi": paper["doi"],
            "paper_url": paper["primary_location"]["landing_page_url"],
            "abstract": paper["abstract_inverted_index"]
        })

print(f"\nTotal papers collected: {len(paper_data)}")

# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame(paper_data)

# -----------------------------
# Save dataset
# -----------------------------

df.to_csv("data/raw/papers.csv", index=False)

print("Dataset saved successfully!")
print("Shape:", df.shape)