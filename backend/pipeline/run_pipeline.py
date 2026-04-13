from backend.clustering.cluster import cluster_embeddings
from backend.deduplication.dedup import deduplicate
from backend.embeddings.embedder import HybridEmbedder
from backend.embeddings.vector_store import FAISSStore
from backend.ingestion.fetch_cases import load_case_records_from_csv
from backend.kb.generator import classify_kb_template, generate_kb
from backend.kb.kb_store import load_kbs, save_kb
from backend.kb.validator import is_duplicate_kb, validate_kb_template
from backend.preprocessing.clean_text import clean_text
from backend.preprocessing.pii_removal import remove_pii
from backend.processing.aggregator import aggregate_resolution_steps, combine_issue_texts
from backend.rag.retrieval import build_kb_retrieval
from backend.response.formatter import build_structured_resolution
from backend.config import ENABLE_PRE_CLUSTER_DEDUP


def _sanitize_case_text(value: str) -> str:
    return remove_pii(clean_text(value or ""))


def run_pipeline(csv_path: str) -> dict:
    print("\n[STEP 1] Loading data...")
    records = load_case_records_from_csv(str(csv_path))
    print(f"Loaded {len(records)} records")

    if not records:
        print("[WARN] No records found in CSV.")
        return {"loaded": 0, "clusters": 0, "created": 0, "skipped": 0, "failed": 0}

    print("\n[STEP 2] Cleaning + Removing PII...")
    texts = [_sanitize_case_text(record["issue_text"]) for record in records]

    print("\n[STEP 3] Generating embeddings...")
    embedder = HybridEmbedder()
    embeddings = embedder.encode(texts)
    print(f"Embedding shape: {embeddings.shape}")

    print("\n[STEP 4] Deduplication...")
    if ENABLE_PRE_CLUSTER_DEDUP:
        unique_idx = deduplicate(texts, embeddings)
        records = [records[i] for i in unique_idx]
        texts = [texts[i] for i in unique_idx]
        embeddings = embeddings[unique_idx]
        print(f"Dedup enabled: kept {len(texts)} records")
    else:
        print("Dedup disabled before clustering (recommended for small datasets)")

    print("\n[STEP 5] Clustering...")
    labels = cluster_embeddings(embeddings)
    print(f"Cluster labels: {labels}")

    print("\n[STEP 6] Initializing FAISS store...")
    dim = embeddings.shape[1]
    vector_store = FAISSStore(dim, load_existing=True)
    print(f"Existing KB vectors in store: {vector_store.ntotal}")

    clusters = {}
    for index, label in enumerate(labels):
        if label == -1:
            print(f"[INFO] Skipping noise point index {index}")
            continue
        clusters.setdefault(int(label), []).append(records[index])

    print(f"\nClusters found: {len(clusters)}")
    print("\n[STEP 7] Processing clusters...")

    created = 0
    skipped = 0
    failed = 0
    existing_kbs = load_kbs()

    for cluster_id, cluster_records in clusters.items():
        print(f"\n--- Processing Cluster {cluster_id} ({len(cluster_records)} cases) ---")

        cluster_texts = [_sanitize_case_text(record.get("issue_text", "")) for record in cluster_records]
        resolution_hints = []
        for record in cluster_records:
            if record.get("resolution"):
                resolution_hints.append(_sanitize_case_text(record["resolution"]))
            if record.get("troubleshooting"):
                resolution_hints.append(_sanitize_case_text(record["troubleshooting"]))

        next_step_hints = [
            _sanitize_case_text(record["next_steps"])
            for record in cluster_records
            if record.get("next_steps")
        ]
        root_cause_hints = [
            _sanitize_case_text(record["root_cause"])
            for record in cluster_records
            if record.get("root_cause")
        ]

        cluster_text_combined = combine_issue_texts(cluster_texts)
        cluster_vec = embedder.encode([cluster_text_combined])[0]

        if is_duplicate_kb(cluster_vec, existing_kbs):
            print("Skipping: duplicate KB article already exists")
            skipped += 1
            continue

        resolution_agg = aggregate_resolution_steps(resolution_hints)
        print(
            f"[RESOLUTION SYNTHESIS] {len(resolution_agg['primary'])} primary steps, {len(resolution_agg['secondary'])} secondary steps"
        )

        structured_resolution = build_structured_resolution(
            resolution_agg["primary"],
            resolution_agg["secondary"],
            next_step_hints,
        )

        template_type = classify_kb_template(cluster_texts, resolution_hints, next_step_hints)
        print(f"[TEMPLATE] Using template: {template_type}")

        kb_input_texts = list(cluster_texts)
        if structured_resolution:
            kb_input_texts.append(structured_resolution)
        if root_cause_hints:
            unique_root_causes = list(dict.fromkeys(root_cause_hints))
            kb_input_texts.append(
                "ROOT CAUSE OBSERVATIONS:\n" + "\n".join(f"- {item}" for item in unique_root_causes)
            )

        kb = generate_kb(kb_input_texts, template_type=template_type)
        print(f"Generated KB: {kb}")

        if not kb:
            print("KB generation failed")
            failed += 1
            continue

        kb["template_type"] = template_type
        known_fix_steps = resolution_agg["primary"] if resolution_agg["primary"] else resolution_agg["secondary"][:1]
        is_valid_template, template_issues = validate_kb_template(
            kb,
            template_type=template_type,
            known_fix_steps=known_fix_steps,
        )
        if not is_valid_template:
            print("KB failed Smarsh template validation:")
            for issue in template_issues:
                print(f"  - {issue}")
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
        kb_metadata["_cluster_vec"] = cluster_vec.tolist()

        save_kb(kb_metadata)
        vector_store.add([kb_vec], [kb_metadata])
        existing_kbs.append(kb_metadata)
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
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return summary
