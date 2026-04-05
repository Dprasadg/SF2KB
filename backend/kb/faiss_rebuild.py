"""
Safe FAISS rebuild utility for regenerating vectors from existing KB articles.

This module provides a command to rebuild the FAISS index from all existing KB articles,
applying the new rich retrieval text embedding strategy. Useful when KB embedding strategy
changes or after bulk KB imports.

Features:
- Dry-run mode (preview changes without modifying FAISS)
- Automatic backup of existing FAISS files
- Clear progress reporting
- Transactional rebuild (backup-first approach)
"""

import json
import shutil
from pathlib import Path
from typing import Optional

from backend.config import KB_STORE_PATH, DATA_DIR
from backend.embeddings.embedder import HybridEmbedder
from backend.embeddings.vector_store import FAISSStore
from backend.rag.retrieval import build_kb_retrieval


def rebuild_faiss_from_kb(dry_run: bool = False) -> dict:
    """
    Rebuild FAISS index from existing KB articles with rich retrieval embeddings.
    
    Process:
    1. Load all KBs from kb_articles.json
    2. Create new FAISS index with all KB vectors
    3. Backup existing FAISS files
    4. Persist new index (unless dry_run=True)
    
    Args:
        dry_run: If True, preview changes without modifying FAISS files
        
    Returns:
        Summary dict with keys: processed, skipped, errors, faiss_path, backup_path
    """
    
    # Load KB articles
    if not Path(KB_STORE_PATH).exists():
        print("[ERROR] KB store not found at", KB_STORE_PATH)
        return {"processed": 0, "skipped": 0, "errors": 0, "faiss_path": None}
    
    with open(KB_STORE_PATH) as f:
        kbs = json.load(f)
    
    print(f"\n[FAISS REBUILD] Found {len(kbs)} KB articles to process")
    
    if len(kbs) == 0:
        print("[WARN] No KB articles found. Skipping rebuild.")
        return {"processed": 0, "skipped": 0, "errors": 0, "faiss_path": None}
    
    # Initialize embedder and vector store (fresh, not loading existing)
    embedder = HybridEmbedder()
    dim = embedder.encode(["test"]).shape[1]
    vector_store = FAISSStore(dim, load_existing=False)  # Fresh store
    
    processed = 0
    skipped = 0
    errors = 0
    
    print("\n[PROGRESS] Encoding KB articles...")
    for i, kb in enumerate(kbs, 1):
        try:
            if not kb.get("title"):
                print(f"  [{i}/{len(kbs)}] SKIP: no title")
                skipped += 1
                continue
            
            # Build rich retrieval text (new strategy)
            retrieval_text = build_kb_retrieval(kb)
            
            if not retrieval_text.strip():
                print(f"  [{i}/{len(kbs)}] SKIP: empty retrieval text")
                skipped += 1
                continue
            
            # Encode and prepare metadata
            kb_vec = embedder.encode([retrieval_text])[0]
            kb_metadata = dict(kb)
            kb_metadata["_retrieval_text"] = retrieval_text
            # Note: _cluster_vec not available for old KBs (only in new uploads)
            
            # Add to new index
            vector_store.add([kb_vec], [kb_metadata])
            processed += 1
            
            if i % 10 == 0:
                print(f"  [{i}/{len(kbs)}] Processed {processed} articles")
        
        except Exception as e:
            print(f"  [{i}/{len(kbs)}] ERROR: {str(e)}")
            errors += 1
    
    # Backup existing FAISS if present
    faiss_dir = Path(DATA_DIR) / "faiss"
    backup_path = None
    
    if faiss_dir.exists() and any(faiss_dir.glob("*")):
        if dry_run:
            backup_path = str(faiss_dir.parent / "faiss_backup_old")
            print(f"\n[DRY RUN] Would backup existing FAISS to: {backup_path}")
        else:
            backup_path = str(faiss_dir.parent / "faiss_backup_old")
            if Path(backup_path).exists():
                shutil.rmtree(backup_path)
            shutil.copytree(faiss_dir, backup_path)
            print(f"\n[BACKUP] Existing FAISS backed up to: {backup_path}")
    
    # Persist new FAISS (unless dry_run)
    if dry_run:
        print(f"\n[DRY RUN] Would persist {processed} KB vectors to: {faiss_dir}")
        print("[DRY RUN] No files modified. Re-run without --dry-run to apply changes.")
    else:
        vector_store.index_path = str(faiss_dir / "index.faiss")
        vector_store.metadata_path = str(faiss_dir / "metadata.json")
        vector_store.save()
        print(f"\n[SUCCESS] New FAISS index persisted with {processed} KB vectors")
    
    summary = {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "faiss_path": str(faiss_dir),
        "backup_path": backup_path,
        "dry_run": dry_run,
    }
    
    print("\n[SUMMARY]")
    print(f"  Processed: {summary['processed']}")
    print(f"  Skipped:   {summary['skipped']}")
    print(f"  Errors:    {summary['errors']}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rebuild FAISS index from existing KB articles"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying FAISS files"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FAISS Index Rebuild Tool")
    print("=" * 60)
    
    summary = rebuild_faiss_from_kb(dry_run=args.dry_run)
    
    if summary["errors"] > 0:
        print(f"\n[WARNING] {summary['errors']} articles failed to process")
    
    exit(0 if summary["errors"] == 0 else 1)
