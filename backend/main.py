from backend.ingestion.fetch_cases import load_case_records_from_csv
from backend.preprocessing.clean_text import clean_text
from backend.preprocessing.pii_removal import remove_pii

from backend.embeddings.embedder import HybridEmbedder
from backend.embeddings.vector_store import FAISSStore

from backend.deduplication.dedup import deduplicate
from backend.clustering.cluster import cluster_embeddings

from backend.kb.generator import generate_kb
from backend.kb.validator import is_duplicate_kb
from backend.kb.kb_store import save_kb
from backend.config import CSV_FILE_PATH, ENABLE_PRE_CLUSTER_DEDUP

# 1. Load
print("\n[STEP 1] Loading data...")
records = load_case_records_from_csv(str(CSV_FILE_PATH))
print(f"Loaded {len(records)} records")

# 2. Clean + PII
print("\n[STEP 2] Cleaning + Removing PII...")
texts = [remove_pii(clean_text(r["issue_text"])) for r in records]

# 3. Embeddings
print("\n[STEP 3] Generating embeddings...")
embedder = HybridEmbedder()
embeddings = embedder.encode(texts)
print(f"Embedding shape: {embeddings.shape}")

# 4. Dedup
print("\n[STEP 4] Deduplication...")
if ENABLE_PRE_CLUSTER_DEDUP:
    unique_idx = deduplicate(texts, embeddings)
    records = [records[i] for i in unique_idx]
    texts = [texts[i] for i in unique_idx]
    embeddings = embeddings[unique_idx]
    print(f"Dedup enabled: kept {len(texts)} records")
else:
    print("Dedup disabled before clustering (recommended for small datasets)")

# 5. Cluster
print("\n[STEP 5] Clustering...")
labels = cluster_embeddings(embeddings)
print(f"Cluster labels: {labels}")

# 6. Vector store
print("\n[STEP 6] Initializing FAISS store...")
dim = embeddings.shape[1]
vector_store = FAISSStore(dim)

# 7. Group clusters
clusters = {}
for i, label in enumerate(labels):
    if label == -1:
        print(f"[INFO] Skipping noise point index {i}")
        continue
    clusters.setdefault(label, []).append(records[i])

print(f"\nClusters found: {len(clusters)}")

# 8. Process clusters
print("\n[STEP 7] Processing clusters...")

created = 0
skipped = 0
failed = 0

for cluster_id, cluster_records in clusters.items():
    print(f"\n--- Processing Cluster {cluster_id} ---")
    print(f"Cluster size: {len(cluster_records)}")

    cluster_texts = [remove_pii(clean_text(r.get("issue_text", ""))) for r in cluster_records]
    resolution_hints = [r.get("resolution", "") for r in cluster_records if r.get("resolution")]
    next_step_hints = [r.get("next_steps", "") for r in cluster_records if r.get("next_steps")]

    cluster_text_combined = "\n\n".join([f"CASE:\n{t}" for t in cluster_texts])
    cluster_vec = embedder.encode([cluster_text_combined])[0]

    # Duplicate check
    is_dup = is_duplicate_kb(cluster_vec, vector_store)
    print(f"Duplicate KB? {is_dup}")

    if is_dup:
        print("Skipping due to duplicate")
        skipped += 1
        continue

    # Generate KB using issue text plus hints from solution-oriented columns.
    kb_input_texts = list(cluster_texts)
    kb_input_texts.extend([f"resolution_hint: {h}" for h in resolution_hints[:3]])
    kb_input_texts.extend([f"next_steps_hint: {h}" for h in next_step_hints[:3]])
    kb = generate_kb(kb_input_texts)

    print(f"Generated KB Output: {kb}")

    if not kb:
        print("KB generation failed")
        failed += 1
        continue

    # Save KB
    print("Saving KB...")
    save_kb(kb)

    # Store vector + metadata
    vector_store.add(
        [cluster_vec],
        [{
            "title": kb.get("title", ""),
            "resolution": kb.get("resolution", "")
        }]
    )

    print("KB successfully created")
    created += 1

# Final summary
print("\n========== FINAL SUMMARY ==========")
print(f"Clusters processed: {len(clusters)}")
print(f"KB Created: {created}")
print(f"KB Skipped (duplicates): {skipped}")
print(f"KB Failed: {failed}")