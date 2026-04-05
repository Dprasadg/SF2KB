import hdbscan
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from backend.config import HDBSCAN_MIN_SAMPLES, MIN_CLUSTER_SIZE


def cluster_embeddings(embeddings) -> np.ndarray:
    if MIN_CLUSTER_SIZE < 2:
        raise ValueError("MIN_CLUSTER_SIZE must be >= 2 for HDBSCAN.")
    if HDBSCAN_MIN_SAMPLES < 1:
        raise ValueError("HDBSCAN_MIN_SAMPLES must be >= 1.")

    embedding_matrix = np.asarray(embeddings, dtype=np.float32)
    if embedding_matrix.ndim != 2:
        raise ValueError("embeddings must be a 2D array-like object.")

    if embedding_matrix.shape[0] == 0:
        return np.array([], dtype=np.int32)

    if embedding_matrix.shape[0] < MIN_CLUSTER_SIZE:
        return np.full(embedding_matrix.shape[0], -1, dtype=np.int32)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(embedding_matrix)

    # HDBSCAN can mark all points as noise on tiny datasets; use a deterministic fallback.
    if np.all(labels == -1):
        target_clusters = max(2, embedding_matrix.shape[0] // MIN_CLUSTER_SIZE)
        try:
            fallback = AgglomerativeClustering(
                n_clusters=target_clusters,
                metric="cosine",
                linkage="average",
            )
        except TypeError:
            fallback = AgglomerativeClustering(
                n_clusters=target_clusters,
                affinity="cosine",
                linkage="average",
            )
        return fallback.fit_predict(embedding_matrix).astype(np.int32)

    return labels