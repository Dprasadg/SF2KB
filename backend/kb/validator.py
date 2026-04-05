from backend.config import KB_DUPLICATE_THRESHOLD
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def is_duplicate_kb(cluster_vec: np.ndarray, kb_list: list, threshold: float = KB_DUPLICATE_THRESHOLD) -> bool:
    """
    Check if cluster vector is duplicate against all existing KB cluster vectors.
    
    Uses cluster_vec (problem-only) instead of kb_vec for consistency.
    
    Args:
        cluster_vec: Issue-only embedding vector for this cluster
        kb_list: List of KB dicts (each contains '_cluster_vec' field)
        threshold: Similarity threshold (0-1) above which considered duplicate
        
    Returns:
        True if similar KB exists, False otherwise
    """
    if not kb_list:
        return False
    
    cluster_vecs = [kb.get("_cluster_vec") for kb in kb_list if "_cluster_vec" in kb]
    
    if not cluster_vecs:
        return False
    
    # Convert db vectors back to numpy if needed
    cluster_vecs = [np.array(vec) if isinstance(vec, list) else vec for vec in cluster_vecs]
    
    # Stack all cluster vecs and compute similarity
    cluster_vecs_array = np.array(cluster_vecs)
    similarities = cosine_similarity([cluster_vec], cluster_vecs_array)[0]
    
    return np.max(similarities) > threshold