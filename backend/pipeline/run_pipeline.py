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
from backend.config import ENABLE_PRE_CLUSTER_DEDUP
from backend.processing.aggregator import aggregate_resolution_steps, combine_issue_texts
from backend.response.formatter import build_structured_resolution
from backend.rag.retrieval import build_kb_retrieval


def run_pipeline(csv_path: str) -> dict:
    """
    Run the full SF → KB pipeline on the given CSV file.

    Args:
        csv_path: Absolute or relative path to the uploaded CSV file.

    Returns:
        A summary dict with keys: loaded, clusters, created, skipped, failed.
    """

    # 1. Load
    print("\n[STEP 1] Loading data...")
    records = load_case_records_from_csv(str(csv_path))
    print(f"Loaded {len(records)} records")

    if not records:
        print("[WARN] No records found in CSV.")
        return {"loaded": 0, "clusters": 0, "created": 0, "skipped": 0, "failed": 0}

    # 2. Clean + PII
    print("\n[STEP 2] Cleaning + Removing PII...")
    texts = [remove_pii(clean_text(r["issue_text"])) for r in records]

    # 3. Embeddings
    print("\n[STEP 3] Generating embeddings...")
    embedder = HybridEmbedder()
    embeddings = embedder.encode(texts)
    print(f"Embedding shape: {embeddings.shape}")

    # 4. Dedup (optional, off by default for small datasets)
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

    # 6. Vector store — loads existing index from disk so duplicate detection
    #    covers ALL previously created KBs, not only those from this run.
    print("\n[STEP 6] Initializing FAISS store...")
    dim = embeddings.shape[1]
    vector_store = FAISSStore(dim, load_existing=True)
    print(f"Existing KB vectors in store: {vector_store.ntotal}")

    # 7. Group clusters (label -1 = noise, skip)
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            print(f"[INFO] Skipping noise point index {i}")
            continue
        clusters.setdefault(int(label), []).append(records[i])

    print(f"\nClusters found: {len(clusters)}")

    # 8. Process clusters → generate KB articles
    print("\n[STEP 7] Processing clusters...")
    created = 0
    skipped = 0
    failed = 0
    
    # Load existing KBs for duplicate checking (cluster vec based)
    from backend.kb.kb_store import load_kbs
    existing_kbs = load_kbs()

    for cluster_id, cluster_records in clusters.items():
        print(f"\n--- Processing Cluster {cluster_id} ({len(cluster_records)} cases) ---")

        cluster_texts = [
            remove_pii(clean_text(r.get("issue_text", ""))) for r in cluster_records
        ]
        resolution_hints = [
            r["resolution"] for r in cluster_records if r.get("resolution")
        ]
        next_step_hints = [
            r["next_steps"] for r in cluster_records if r.get("next_steps")
        ]

        cluster_text_combined = combine_issue_texts(cluster_texts)
        cluster_vec = embedder.encode([cluster_text_combined])[0]

        if is_duplicate_kb(cluster_vec, existing_kbs):
            print("Skipping: duplicate KB article already exists")
            skipped += 1
            continue

        # Aggregate ALL resolution hints with frequency ranking
        resolution_agg = aggregate_resolution_steps(resolution_hints)
        print(f"[RESOLUTION SYNTHESIS] {len(resolution_agg['primary'])} primary steps, {len(resolution_agg['secondary'])} secondary steps")
        
        # Build structured resolution context
        structured_resolution = build_structured_resolution(
            resolution_agg["primary"],
            resolution_agg["secondary"],
            next_step_hints,
        )
        
        # Prepare input for KB generation
        kb_input_texts = list(cluster_texts)
        if structured_resolution:
            kb_input_texts.append(structured_resolution)
        kb = generate_kb(kb_input_texts)

        print(f"Generated KB: {kb}")

        if not kb:
            print("KB generation failed")
            failed += 1
            continue

        kb_retrieval_text = build_kb_retrieval(
            kb,
            cluster_texts,
            resolution_hints,
            next_step_hints,
        )
        kb_vec = embedder.encode([kb_retrieval_text])[0]
        kb_metadata = dict(kb)
        kb_metadata["_retrieval_text"] = kb_retrieval_text
        kb_metadata["_cluster_vec"] = cluster_vec.tolist()  # Store issue-only vec for dedup

        save_kb(kb)
        vector_store.add([kb_vec], [kb_metadata])
        existing_kbs.append(kb_metadata)  # Update in-memory list for next checks
        print("KB successfully created")
        created += 1

    summary = {
        "loaded": len(records),
        "clusters": len(clusters),
        "created": created,
        "skipped": skipped,
        "failed": failed,
    }

    print("\n========== FINAL SUMMARY ==========")
    for key, val in summary.items():
        print(f"  {key}: {val}")

    return summary

