from pathlib import Path
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from pydantic import BaseModel
import json

from backend.pipeline.run_pipeline import run_pipeline
from backend.search.rag import search_kb
from backend.embeddings.embedder import HybridEmbedder
from backend.embeddings.vector_store import FAISSStore
from backend.config import CORS_ORIGINS, KB_STORE_PATH, DATA_DIR, MAX_UPLOAD_BYTES, LOG_DIR
from backend.kb.kb_store import load_kbs, update_kb_fields
from backend.kb.validator import validate_kb_template
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
    top_k: int = 10
    candidate_k: int = 40
    template_type: Optional[str] = None   # "solution" | "how_to" | "qa" | None
    applies_to: Optional[str] = None      # e.g. "Digital Safe" | "Enterprise Archive" | None
    validation_state: Optional[str] = None
    visibility: Optional[str] = None
    approval_status: str = "approved"    # approved | pending | needs_edits | all


class KBApprovalRequest(BaseModel):
    approved: bool = True


class KBUpdateRequest(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    symptoms: Optional[list[str]] = None
    applies_to: Optional[list[str]] = None
    resolution: Optional[list[str]] = None
    cause: Optional[str] = None
    additional_info: Optional[str] = None
    keyword_variations: Optional[list[str]] = None
    objective: Optional[str] = None
    steps: Optional[list[str]] = None
    answer: Optional[str] = None
    approve: bool = False


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


def _validation_issues_for(kb: dict) -> list[str]:
    try:
        _, issues = validate_kb_template(kb, template_type=kb.get("template_type"))
        return issues
    except Exception:
        return ["Unable to validate KB template"]


def _review_status(kb: dict) -> str:
    if _is_approved(kb):
        return "approved"
    issues = _validation_issues_for(kb)
    return "needs_edits" if issues else "pending_review"


def _enrich_kb_for_review(kb: dict) -> dict:
    public = _public_kb(kb)
    issues = _validation_issues_for(kb)
    public["validation_issues"] = issues
    public["review_status"] = "approved" if _is_approved(kb) else ("needs_edits" if issues else "pending_review")
    return public


def _kb_matches_search_filters(kb: dict, request: QueryRequest) -> bool:
    if request.template_type:
        kb_template = (kb.get("template_type") or "solution").strip().lower()
        if kb_template != request.template_type.strip().lower():
            return False

    if request.applies_to:
        applies_to_list = [str(a).lower() for a in (kb.get("applies_to") or [])]
        filter_val = request.applies_to.strip().lower()
        if not any(filter_val in a for a in applies_to_list):
            return False

    if request.validation_state:
        if str(kb.get("validation_state") or "").strip().lower() != request.validation_state.strip().lower():
            return False

    if request.visibility:
        if str(kb.get("visibility") or "").strip().lower() != request.visibility.strip().lower():
            return False

    approval_status = str(request.approval_status or "approved").strip().lower()
    status = _review_status(kb)
    if approval_status == "approved" and status != "approved":
        return False
    if approval_status == "pending" and status != "pending_review":
        return False
    if approval_status == "needs_edits" and status != "needs_edits":
        return False

    return True


def _normalize_update_payload(payload: KBUpdateRequest) -> dict:
    updates = {}
    fields = [
        "title",
        "summary",
        "symptoms",
        "applies_to",
        "resolution",
        "cause",
        "additional_info",
        "keyword_variations",
        "objective",
        "steps",
        "answer",
    ]
    for field in fields:
        value = getattr(payload, field)
        if value is not None:
            updates[field] = value
    return updates


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

        top_k = max(1, min(int(request.top_k or 10), 50))
        candidate_k = max(top_k, min(int(request.candidate_k or 40), 200))
        logger.debug(
            f"Search query: '{request.query}' top_k={top_k} candidate_k={candidate_k} "
            f"template={request.template_type} applies_to={request.applies_to} approval_status={request.approval_status}"
        )

        active_embedder, active_vector_store = _get_search_runtime()
        raw_results = search_kb(
            request.query,
            active_embedder,
            active_vector_store,
            top_k=top_k,
            candidate_k=candidate_k,
            include_scores=True,
            log_dir=str(LOG_DIR),
        )
        logger.debug(f"RAG returned {len(raw_results) if raw_results else 0} results")

        results = []
        for item in raw_results:
            kb = item.get("kb", {}) if isinstance(item, dict) else item
            if not isinstance(kb, dict):
                continue

            # Use cached KB state instead of iterating live JSON
            kb_id = str(kb.get("kb_id", "")).strip() if kb.get("kb_id") else ""
            title = str(kb.get("title", "")).strip() if kb.get("title") else ""
            kb_live = kb_cache.get_by_id_or_title(kb_id, title) if (kb_id or title) else None

            if not kb_live:
                kb_live = kb

            if not _kb_matches_search_filters(kb_live, request):
                continue

            enriched_kb = _enrich_kb_for_review(kb_live)

            results.append(
                {
                    "kb": enriched_kb,
                    "score": item.get("score") if isinstance(item, dict) else None,
                    "semantic_score": item.get("semantic_score") if isinstance(item, dict) else None,
                    "keyword_score": item.get("keyword_score") if isinstance(item, dict) else None,
                    "bm25_score": item.get("bm25_score") if isinstance(item, dict) else None,
                    "matched_fields": item.get("matched_fields", []) if isinstance(item, dict) else [],
                    "matched_terms": item.get("matched_terms", []) if isinstance(item, dict) else [],
                }
            )

        logger.info(f"Search '{request.query}' returned {len(results)} approved results")
        return {"results": results, "total": len(results)}

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
def get_kb(approval: str = Query("all", pattern="^(all|approved|pending|needs_edits)$")):
    kb_path = Path(KB_STORE_PATH)
    if not kb_path.exists():
        return {"kb": []}

    all_kbs_raw = [kb for kb in load_kbs(include_runtime=False) if isinstance(kb, dict)]
    all_kbs = [_enrich_kb_for_review(kb) for kb in all_kbs_raw]

    if approval == "approved":
        all_kbs = [kb for kb in all_kbs if kb.get("review_status") == "approved"]
    elif approval == "pending":
        all_kbs = [kb for kb in all_kbs if kb.get("review_status") == "pending_review"]
    elif approval == "needs_edits":
        all_kbs = [kb for kb in all_kbs if kb.get("review_status") == "needs_edits"]

    return {"kb": _sort_approved_first(all_kbs)}


@app.patch("/kb/{kb_id}")
def update_kb(kb_id: str, request: KBUpdateRequest):
    try:
        kb_id = str(kb_id or "").strip()
        if not kb_id:
            raise HTTPException(status_code=400, detail="Invalid KB id")

        all_kbs = [kb for kb in load_kbs(include_runtime=False) if isinstance(kb, dict)]
        existing = next((kb for kb in all_kbs if str(kb.get("kb_id", "")).strip() == kb_id), None)
        if not existing:
            raise HTTPException(status_code=404, detail="KB article not found")

        updates = _normalize_update_payload(request)
        merged = dict(existing)
        merged.update(updates)

        _, issues = validate_kb_template(merged, template_type=merged.get("template_type"))
        approve_requested = bool(request.approve)
        if approve_requested and not issues:
            updates["validation_state"] = "Validated"
            updates["approved_at"] = datetime.now(timezone.utc).isoformat()
        elif approve_requested and issues:
            updates["validation_state"] = "Not Validated"
            updates["approved_at"] = None

        if not updates and not approve_requested:
            raise HTTPException(status_code=400, detail="No editable fields provided")

        updated = update_kb_fields(kb_id, updates)
        if updated is None:
            raise HTTPException(status_code=404, detail="KB article not found")

        kb_cache.update_kb(updated)
        response_kb = _enrich_kb_for_review(updated)
        return {
            "status": "success",
            "kb": response_kb,
            "approval_applied": bool(approve_requested and not issues),
            "validation_issues": response_kb.get("validation_issues", []),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"KB update failed for {kb_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"KB update failed: {str(exc)}") from exc


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
