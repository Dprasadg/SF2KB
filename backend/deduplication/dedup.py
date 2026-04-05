import numpy as np
from backend.config import SIMILARITY_THRESHOLD


def deduplicate(texts, embeddings) -> list[int]:
    if not 0 <= SIMILARITY_THRESHOLD <= 1:
        raise ValueError("SIMILARITY_THRESHOLD must be in [0, 1].")

    embedding_matrix = np.asarray(embeddings, dtype=np.float32)
    if embedding_matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D array-like object.")

    if texts is not None and len(texts) != embedding_matrix.shape[0]:
        raise ValueError("texts and embeddings must have the same length.")

    if embedding_matrix.shape[0] == 0:
        return []

    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    # Avoid division by zero for degenerate vectors.
    normalized_embeddings = embedding_matrix / np.clip(norms, 1e-12, None)

    unique_idx: list[int] = []
    unique_vecs: list[np.ndarray] = []

    for i, vec in enumerate(normalized_embeddings):
        if not unique_vecs:
            unique_idx.append(i)
            unique_vecs.append(vec)
            continue

        sims = np.dot(np.asarray(unique_vecs), vec)

        if float(np.max(sims)) < SIMILARITY_THRESHOLD:
            unique_idx.append(i)
            unique_vecs.append(vec)

    return unique_idx