import pandas as pd
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ------------------------------
# CONFIGURATION
# ------------------------------

CSV_FILE = "SampleData.csv"
NUM_CLUSTERS = 2
MIN_CLUSTER_SIZE = 1
KB_FOLDER = "/Users/durgaprasad/Documents/KB_Articles"

# ------------------------------
# LOAD DATA
# ------------------------------

df = pd.read_csv(CSV_FILE)
df = df.fillna("")

print("Loaded tickets:", len(df))

# ------------------------------
# CREATE RAW TEXT FROM FIELDS
# ------------------------------

df["ticket_text"] = (
    "Subject: " + df["Subject"] + "\n" +
    "Topic: " + df["Topic"] + "\n" +
    "SubTopic: " + df["SubTopic"] + "\n" +
    "Product: " + df["Product"] + "\n" +
    "Severity: " + df["Severity Level"] + "\n\n" +
    "Description: " + df["Description"] + "\n\n" +
    "Troubleshooting: " + df["Troubleshooting Steps Taken"] + "\n\n" +
    "Resolution: " + df["Resolution"]
)

# ------------------------------
# EXTRACT CORE ISSUE USING LLM
# ------------------------------

print("Extracting core issues using Llama...")

issue_summaries = []

for text in df["ticket_text"]:

    prompt = f"""
Analyze the support ticket below and extract the core issue.

Return ONLY a short issue statement (max 10 words).

Ticket:
{text}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    issue = response["message"]["content"].strip().lower()
    issue_summaries.append(issue)

df["issue_summary"] = issue_summaries

print("Issue extraction completed")

# ------------------------------
# GENERATE EMBEDDINGS
# ------------------------------

print("Generating embeddings...")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embed_model.encode(
    df["issue_summary"].tolist(),
    batch_size=64,
    show_progress_bar=True
)

# ------------------------------
# CLUSTER ISSUES
# ------------------------------

print("Clustering issues...")

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)

df["cluster"] = kmeans.fit_predict(embeddings)

# ------------------------------
# GENERATE KB ARTICLES
# ------------------------------

print("Generating KB articles...")

for cluster_id in sorted(df["cluster"].unique()):

    cluster_df = df[df["cluster"] == cluster_id]

    if len(cluster_df) < MIN_CLUSTER_SIZE:
        continue

    sample_tickets = cluster_df["ticket_text"].tolist()[:20]

    combined = "\n\n".join(sample_tickets)

    prompt = f"""
You are a technical support documentation expert.

The following support tickets describe the SAME issue.

Tickets:
{combined}

Create a Knowledge Base article with sections:

Title
Problem
Environment
Cause
Resolution
Steps to Resolve
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    article = response["message"]["content"]

    filename = f"{KB_FOLDER}/kb_cluster_{cluster_id}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(article)

    print(f"Created KB article: {filename}")

print("Pipeline finished successfully.")