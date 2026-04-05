import faiss
import numpy as np
import json
from pathlib import Path
from typing import Any

from backend.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH

class FAISSStore:
    def __init__(self, dim: int, load_existing: bool = False):
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")

        self.dim = dim
        self.index_path = Path(FAISS_INDEX_PATH)
        self.metadata_path = Path(FAISS_METADATA_PATH)

        if load_existing and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if self.index.d != self.dim:
                raise ValueError(
                    f"Existing FAISS index dimension ({self.index.d}) does not match expected ({self.dim})."
                )
            if self.metadata_path.exists():
                with self.metadata_path.open("r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = []
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []
            if load_existing:
                # Persist an empty index/metadata so the FAISS directory is initialized.
                self._save()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        """Number of KB vectors currently stored (across all runs)."""
        return self.index.ntotal

    def reload(self) -> None:
        """Reload index and metadata from disk (e.g. after the pipeline writes new vectors)."""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if self.index.d != self.dim:
                raise ValueError(
                    f"Existing FAISS index dimension ({self.index.d}) does not match expected ({self.dim})."
                )

            if self.metadata_path.exists():
                with self.metadata_path.open("r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = []
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

    def reset(self) -> None:
        """Wipe the in-memory index and metadata and persist the empty state."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self._save()

    def add(self, vectors: Any, metadata_list: list[dict[str, Any]]) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array-like object.")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")
        if len(metadata_list) != vectors.shape[0]:
            raise ValueError("metadata_list length must match number of vectors.")

        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.metadata.extend(metadata_list)

        self._save()

    def search(self, vector: Any, k: int = 3) -> list[dict[str, Any]]:
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        if self.index.ntotal == 0:
            return []

        vector = np.asarray(vector, dtype="float32")
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.ndim != 2 or vector.shape[0] != 1:
            raise ValueError("vector must be a 1D array-like or a single-row 2D array-like.")
        if vector.shape[1] != self.dim:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dim}, got {vector.shape[1]}")

        faiss.normalize_L2(vector)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            idx = int(idx)
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    "score": float(scores[0][i]),
                    "metadata": self.metadata[idx]
                })

        return results

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))

        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)