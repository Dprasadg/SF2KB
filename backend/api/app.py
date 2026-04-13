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
from backend.cache.kb_cache import KBMetadataCache
from utils.logger import get_logger

logger = get_logger(__name__)

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

# Initialize KB metadata cache
kb_cache = KBMetadataCache(KB_STORE_PATH)

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
    try:
        # Sanitize filename to prevent path traversal attacks.
        safe_name = Path(file.filename or "").name
        if not safe_name:
            logger.warning(f"Invalid filename provided: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid filename.")
        if Path(safe_name).suffix.lower() != ".csv":
            logger.warning(f"Non-CSV file upload attempted: {safe_name}")
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
        logger.info(f"Processing CSV upload: {safe_name}")

        bytes_written = 0
        with file_path.open("wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    file_path.unlink(missing_ok=True)
                    logger.warning(f"File upload exceeded max size: {bytes_written} > {MAX_UPLOAD_BYTES}")
                    raise HTTPException(status_code=413, detail="Uploaded file is too large.")
                buffer.write(chunk)

        logger.debug(f"File saved: {file_path} ({bytes_written} bytes)")

        try:
            with PIPELINE_LOCK:
                result = run_pipeline(str(file_path))
                # Reload the global vector store and cache so newly generated KB articles are
                # immediately searchable without restarting the server.
                if vector_store is not None:
                    vector_store.reload()
                    logger.debug("Vector store reloaded after pipeline")
                kb_cache.reload()
                logger.debug("KB cache reloaded after pipeline")
        except Exception as exc:
            logger.error(f"Pipeline failed for {safe_name}: {exc}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

        logger.info(f"CSV processing complete: {safe_name} - {result}")
        return {"status": "success", "result": result}
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"File upload failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(exc)}") from exc


# ------------------------------
# 2. SEARCH KB
# ------------------------------
@app.post("/search")
def search(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty.")

        logger.debug(f"Search query: '{request.query}'")
        active_embedder, active_vector_store = _get_search_runtime()
        raw_results = search_kb(
            request.query,
            active_embedder,
            active_vector_store,
            include_scores=True,
        )
        logger.debug(f"RAG returned {len(raw_results) if raw_results else 0} results")

        results = []
        for item in raw_results:
            kb = item.get("kb", {}) if isinstance(item, dict) else item
            if not isinstance(kb, dict):
                continue

            # Use cached KB state instead of iterating live JSON
            # Try matching by kb_id first, then by title
            kb_id = str(kb.get("kb_id", "")).strip() if kb.get("kb_id") else ""
            title = str(kb.get("title", "")).strip() if kb.get("title") else ""
            kb_live = kb_cache.get_by_id_or_title(kb_id, title) if (kb_id or title) else None
            
            if not kb_live:
                kb_live = kb

            if not _is_approved(kb_live):
                logger.debug(f"Filtering out unapproved KB: {kb_live.get('title', 'Unknown')}")
                continue

            results.append(
                {
                    "kb": _public_kb(kb_live),
                    "score": item.get("score") if isinstance(item, dict) else None,
                    "semantic_score": item.get("semantic_score") if isinstance(item, dict) else None,
                    "keyword_score": item.get("keyword_score") if isinstance(item, dict) else None,
                    "matched_terms": item.get("matched_terms", []) if isinstance(item, dict) else [],
                }
            )

        logger.info(f"Search '{request.query}' returned {len(results)} approved results")
        return {"results": results}
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Search failed with exception: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(exc)}",
        ) from exc


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
    try:
        approved = bool(request.approved)
        validation_state = "Validated" if approved else "Not Validated"
        updates = {
            "validation_state": validation_state,
            "approved_at": datetime.now(timezone.utc).isoformat() if approved else None,
        }

        logger.info(f"Setting KB {kb_id} approval: {validation_state}")
        updated = update_kb_fields(kb_id, updates)
        if updated is None:
            logger.warning(f"KB not found for approval update: {kb_id}")
            raise HTTPException(status_code=404, detail="KB article not found")

        # Update cache with new approval state
        kb_cache.update_kb(updated)
        logger.info(f"KB {kb_id} approval updated successfully and cached")
        return {"status": "success", "kb": _public_kb(updated)}
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Approval update failed for KB {kb_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Approval update failed: {str(exc)}") from exc


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


# ------------------------------
# 5. HEALTH CHECK
# ------------------------------
@app.get("/health")
def health():
    """Check system health: KB storage, FAISS index, embedder, and services."""
    health_status = {
        "status": "healthy",
        "checks": {
            "kb_store": None,
            "faiss_index": None,
            "embedder": None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Check KB store
        kb_path = Path(KB_STORE_PATH)
        if kb_path.exists():
            with kb_path.open(encoding="utf-8") as f:
                kbs = json.load(f)
                health_status["checks"]["kb_store"] = {
                    "status": "ok",
                    "kb_count": len(kbs) if isinstance(kbs, list) else 0,
                }
        else:
            health_status["checks"]["kb_store"] = {
                "status": "warning",
                "message": "KB store file not found",
            }
    except Exception as exc:
        health_status["checks"]["kb_store"] = {
            "status": "error",
            "error": str(exc),
        }
        health_status["status"] = "degraded"
        logger.warning(f"KB store health check failed: {exc}")

    try:
        # Check FAISS index
        embedder, vector_store = _get_search_runtime()
        if vector_store is not None:
            # Try a simple search to verify FAISS is responsive
            try:
                test_vector = embedder.embed("health check")
                # Just check that FAISS can access the index
                if hasattr(vector_store, "index") and vector_store.index is not None:
                    health_status["checks"]["faiss_index"] = {
                        "status": "ok",
                        "message": "FAISS index responsive",
                    }
                else:
                    health_status["checks"]["faiss_index"] = {
                        "status": "warning",
                        "message": "FAISS index not initialized",
                    }
            except Exception as exc:
                health_status["checks"]["faiss_index"] = {
                    "status": "error",
                    "error": str(exc),
                }
                health_status["status"] = "degraded"
                logger.warning(f"FAISS health check failed: {exc}")
    except Exception as exc:
        health_status["checks"]["faiss_index"] = {
            "status": "error",
            "error": str(exc),
        }
        health_status["status"] = "degraded"
        logger.warning(f"FAISS initialization failed: {exc}")

    try:
        # Check embedder
        if embedder is not None:
            test_embedding = embedder.embed("test")
            if test_embedding is not None and len(test_embedding) > 0:
                health_status["checks"]["embedder"] = {
                    "status": "ok",
                    "embedding_dim": len(test_embedding),
                }
            else:
                health_status["checks"]["embedder"] = {
                    "status": "warning",
                    "message": "Embedder returned empty embedding",
                }
        else:
            health_status["checks"]["embedder"] = {
                "status": "warning",
                "message": "Embedder not initialized",
            }
    except Exception as exc:
        health_status["checks"]["embedder"] = {
            "status": "error",
            "error": str(exc),
        }
        health_status["status"] = "degraded"
        logger.warning(f"Embedder health check failed: {exc}")

    logger.info(f"Health check: {health_status['status']}")
    return health_status
