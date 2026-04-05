from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
from typing import Sequence

from backend.config import EMBEDDING_MODEL


class HybridEmbedder:
    """
    Pure semantic embedder using a SentenceTransformer model.
    Stateless — no fit() step required, safe for API/multi-request use.
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    @staticmethod
    def _sanitize_texts(texts: Sequence) -> list:
        return [str(t) if t is not None else "" for t in texts]

    def encode(self, texts: Sequence) -> np.ndarray:
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)

        clean_texts = self._sanitize_texts(texts)
        embeddings = self.model.encode(
            clean_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return normalize(embeddings).astype(np.float32, copy=False)