from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import json

from backend.pipeline.run_pipeline import run_pipeline
from backend.search.rag import search_kb
from backend.embeddings.embedder import HybridEmbedder
from backend.embeddings.vector_store import FAISSStore
from backend.config import KB_STORE_PATH, DATA_DIR

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# UPLOAD DIR
# ------------------------------
UPLOAD_DIR = Path(DATA_DIR) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------
# INIT (load once at startup)
# HybridEmbedder is now stateless — no fit() required.
# ------------------------------
embedder = HybridEmbedder()
dim = embedder.model.get_sentence_embedding_dimension()
# Loads existing FAISS index from disk so cross-run duplicate detection works.
vector_store = FAISSStore(dim, load_existing=True)


# ------------------------------
# MODELS
# ------------------------------
class QueryRequest(BaseModel):
    query: str


# ------------------------------
# 1. PROCESS CSV
# ------------------------------
@app.post("/process-cases")
async def process_cases(file: UploadFile = File(...)):
    # Sanitize filename to prevent path traversal attacks.
    safe_name = Path(file.filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = UPLOAD_DIR / safe_name

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = run_pipeline(str(file_path))
        # Reload the global vector store so newly generated KB articles are
        # immediately searchable without restarting the server.
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

    results = search_kb(request.query, embedder, vector_store)
    return {"results": results}


# ------------------------------
# 3. GET ALL KB
# ------------------------------
@app.get("/kb")
def get_kb():
    kb_path = Path(KB_STORE_PATH)
    if not kb_path.exists():
        return {"kb": []}

    with kb_path.open(encoding="utf-8") as f:
        return {"kb": json.load(f)}


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