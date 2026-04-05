import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
BASE_DIR = PROJECT_ROOT
DATA_DIR = Path(os.getenv("SF2KB_DATA_DIR", PROJECT_ROOT / "data"))

# Input file
CSV_FILE_PATH = Path(
    os.getenv("SF2KB_CSV_FILE_PATH", PROJECT_ROOT / "SFExportedData" / "SampleData.csv")
)

# Logging
LOG_DIR = Path(os.getenv("SF2KB_LOG_DIR", DATA_DIR / "logs"))
LOG_FILE = Path(os.getenv("SF2KB_LOG_FILE", LOG_DIR / "sf2kb.log"))

# Knowledge base store
KB_STORE_PATH = Path(os.getenv("SF2KB_KB_STORE_PATH", PROJECT_ROOT / "KB_Articles" / "kb_articles.json"))

# thresholds
SIMILARITY_THRESHOLD = float(os.getenv("SF2KB_SIMILARITY_THRESHOLD", "0.95"))
KB_DUPLICATE_THRESHOLD = 0.85
MIN_CLUSTER_SIZE = int(os.getenv("SF2KB_MIN_CLUSTER_SIZE", "4"))
HDBSCAN_MIN_SAMPLES = int(os.getenv("SF2KB_HDBSCAN_MIN_SAMPLES", "1"))
ENABLE_PRE_CLUSTER_DEDUP = os.getenv("SF2KB_ENABLE_PRE_CLUSTER_DEDUP", "false").lower() == "true"

# scoring (TF-IDF removed; weights retained for RAG keyword scoring only)
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# embeddings
EMBEDDING_MODEL = os.getenv("SF2KB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# vector store
FAISS_DIR = Path(os.getenv("SF2KB_FAISS_DIR", DATA_DIR / "faiss"))
FAISS_INDEX_PATH = Path(os.getenv("SF2KB_FAISS_INDEX_PATH", FAISS_DIR / "index.faiss"))
FAISS_METADATA_PATH = Path(
    os.getenv("SF2KB_FAISS_METADATA_PATH", FAISS_DIR / "metadata.json")
)