from pathlib import Path
import threading
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from pydantic import BaseModel
import json

from backend.pipeline.run_pipeline import run_pipeline
from backend.search.rag import search_kb
from backend.embeddings.embedder import HybridEmbedder
from backend.embeddings.vector_store import FAISSStore
from backend.config import CORS_ORIGINS, KB_STORE_PATH, DATA_DIR, MAX_UPLOAD_BYTES
from backend.kb.kb_store import load_kbs, update_kb_fields

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# UPLOAD DIR
# ------------------------------
UPLOAD_DIR = Path(DATA_DIR) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_LOCK = threading.Lock()

# ------------------------------
# Runtime search dependencies are loaded lazily so importing the API module
# does not unexpectedly trigger model downloads or FAISS initialization.
# ------------------------------
embedder = None
vector_store = None


def _get_search_runtime():
    global embedder, vector_store

    if embedder is None:
        embedder = HybridEmbedder()

    if vector_store is None:
        dim = embedder.model.get_sentence_embedding_dimension()
        # Loads existing FAISS index from disk so cross-run duplicate detection works.
        vector_store = FAISSStore(dim, load_existing=True)

    return embedder, vector_store


# ------------------------------
# MODELS
# ------------------------------
class QueryRequest(BaseModel):
    query: str


class KBApprovalRequest(BaseModel):
    approved: bool = True


def _public_kb(kb: dict) -> dict:
    """Remove internal-only and runtime metadata before returning KBs to clients."""
    if not isinstance(kb, dict):
        return {}

    hidden_fields = {"internal_to_smarsh"}
    return {
        key: value
        for key, value in kb.items()
        if not key.startswith("_") and key not in hidden_fields
    }


def _is_approved(kb: dict) -> bool:
    return str(kb.get("validation_state", "")).strip().lower() == "validated"


def _sort_approved_first(items: list[dict]) -> list[dict]:
    return sorted(
        items,
        key=lambda kb: (
            0 if _is_approved(kb) else 1,
            str(kb.get("approved_at") or ""),
            str(kb.get("title") or "").lower(),
        ),
        reverse=False,
    )


# ------------------------------
# 1. PROCESS CSV
# ------------------------------
@app.post("/process-cases")
def process_cases(file: UploadFile = File(...)):
    # Sanitize filename to prevent path traversal attacks.
    safe_name = Path(file.filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if Path(safe_name).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"

    bytes_written = 0
    with file_path.open("wb") as buffer:
        while chunk := file.file.read(1024 * 1024):
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                file_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Uploaded file is too large.")
            buffer.write(chunk)

    try:
        with PIPELINE_LOCK:
            result = run_pipeline(str(file_path))
            # Reload the global vector store so newly generated KB articles are
            # immediately searchable without restarting the server.
            if vector_store is not None:
                vector_store.reload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    return {"status": "success", "result": result}


# ------------------------------
# 2. SEARCH KB
# ------------------------------
@app.post("/search")
def search(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    active_embedder, active_vector_store = _get_search_runtime()
    raw_results = search_kb(
        request.query,
        active_embedder,
        active_vector_store,
        include_scores=True,
    )

    results = []
    for item in raw_results:
        kb = item.get("kb", {}) if isinstance(item, dict) else item
        if not isinstance(kb, dict) or not _is_approved(kb):
            continue

        results.append(
            {
                "kb": _public_kb(kb),
                "score": item.get("score") if isinstance(item, dict) else None,
                "semantic_score": item.get("semantic_score") if isinstance(item, dict) else None,
                "keyword_score": item.get("keyword_score") if isinstance(item, dict) else None,
                "matched_terms": item.get("matched_terms", []) if isinstance(item, dict) else [],
            }
        )

    return {"results": results}


# ------------------------------
# 3. GET ALL KB
# ------------------------------
@app.get("/kb")
def get_kb(approval: str = Query("all", regex="^(all|approved|pending)$")):
    kb_path = Path(KB_STORE_PATH)
    if not kb_path.exists():
        return {"kb": []}

    all_kbs = [_public_kb(kb) for kb in load_kbs(include_runtime=False)]

    if approval == "approved":
        all_kbs = [kb for kb in all_kbs if _is_approved(kb)]
    elif approval == "pending":
        all_kbs = [kb for kb in all_kbs if not _is_approved(kb)]

    return {"kb": _sort_approved_first(all_kbs)}


@app.post("/kb/{kb_id}/approval")
def set_kb_approval(kb_id: str, request: KBApprovalRequest):
    approved = bool(request.approved)
    validation_state = "Validated" if approved else "Not Validated"
    updates = {
        "validation_state": validation_state,
        "approved_at": datetime.now(timezone.utc).isoformat() if approved else None,
    }

    updated = update_kb_fields(kb_id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail="KB article not found")

    return {"status": "success", "kb": _public_kb(updated)}


# ------------------------------
# 4. DASHBOARD
# ------------------------------
@app.get("/dashboard")
def dashboard():
    kb_path = Path(KB_STORE_PATH)
    if not kb_path.exists():
        return {"total_kb": 0, "titles": []}

    with kb_path.open(encoding="utf-8") as f:
        kb_data = json.load(f)

    return {
        "total_kb": len(kb_data),
        "titles": [k.get("title") for k in kb_data],
    }
