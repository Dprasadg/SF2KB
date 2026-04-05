import numpy as np


# ------------------------------
# KEYWORD SCORE
# ------------------------------
def keyword_score(query, text):
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())

    overlap = query_words.intersection(text_words)

    return len(overlap) / (len(query_words) + 1)


# ------------------------------
# BUILD SEARCH TEXT FROM KB
# ------------------------------
def build_kb_text(kb):
    retrieval_text = kb.get("_retrieval_text", "")
    if isinstance(retrieval_text, str) and retrieval_text.strip():
        return retrieval_text

    symptoms = kb.get("symptoms", [])
    keywords = kb.get("keywords", [])
    resolution = kb.get("resolution", [])

    if isinstance(symptoms, str):
        symptoms = [symptoms]
    if isinstance(keywords, str):
        keywords = [keywords]
    if isinstance(resolution, str):
        resolution = [resolution]

    return " ".join([
        kb.get("title", ""),
        kb.get("summary", ""),
        kb.get("cause", ""),
        kb.get("additional_info", ""),
        " ".join(symptoms),
        " ".join(resolution),
        " ".join(keywords),
    ])


# ------------------------------
# MAIN RAG SEARCH FUNCTION
# ------------------------------
def search_kb(query, embedder, vector_store, top_k=10):
    if not query or not str(query).strip():
        return []

    query = str(query).strip()
    query_enhanced = query + " issue problem error"

    try:
        q_vec = embedder.encode([query_enhanced])[0]
    except Exception:
        return []

    try:
        results = vector_store.search(q_vec, k=top_k)
    except Exception:
        return []

    if not results:
        return []

    scored_results = []

    for r in results:
        kb = r.get("metadata", {})
        if not isinstance(kb, dict):
            continue
        text = build_kb_text(kb)

        semantic_score = float(r.get("score", 0.0))
        keyword_sim = keyword_score(query, text)

        final_score = (0.7 * semantic_score) + (0.3 * keyword_sim)

        scored_results.append((final_score, kb))

    if not scored_results:
        return []

    # ------------------------------
    # SORT RESULTS
    # ------------------------------
    scored_results.sort(reverse=True, key=lambda x: x[0])

    max_score = scored_results[0][0]

    # ------------------------------
    # DYNAMIC FILTERING
    # ------------------------------
    filtered_results = [
        (score, kb)
        for score, kb in scored_results
        if score >= max_score * 0.7
    ]

    # ------------------------------
    # REMOVE DUPLICATES
    # ------------------------------
    seen_titles = set()
    final_results = []

    for score, kb in filtered_results:
        title = kb.get("title")

        if title not in seen_titles:
            final_results.append(kb)
            seen_titles.add(title)

        if len(final_results) >= 5:
            break

    return final_results