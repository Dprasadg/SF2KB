import pandas as pd
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ---------------------------
# 1. Load CSV
# ---------------------------
df = pd.read_csv("SampleData.csv")
df = df.fillna("")

# ---------------------------
# 2. Combine Important Fields
# ---------------------------
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

tickets = df["ticket_text"].tolist()

# ---------------------------
# 3. Generate Embeddings
# ---------------------------
print("Generating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(tickets, batch_size=32)

# ---------------------------
# 4. Cluster Tickets
# ---------------------------
num_clusters = 5  # adjust later (15 for large datasets)

print("Clustering tickets...")

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(embeddings)

# ---------------------------
# 5. Generate KB per Cluster
# ---------------------------
print("Generating KB articles...")

for cluster_id in df["cluster"].unique():

    cluster_tickets = df[df["cluster"] == cluster_id]["ticket_text"].tolist()

    # sample max 20 tickets
    sample = cluster_tickets[:20]

    combined_text = "\n\n".join(sample)

    prompt = f"""
You are a support knowledge base writer.

Analyze the following support tickets and determine the common issue.

Tickets:
{combined_text}

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

    filename = f"kb_cluster_{cluster_id}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(article)

    print(f"KB article created: {filename}")

print("Pipeline completed.")