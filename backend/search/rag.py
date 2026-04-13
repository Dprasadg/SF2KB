import numpy as np
import re


def keyword_score(query, text):
    query_words = _tokenize(query)
    text_words = _tokenize(text)
    overlap = query_words.intersection(text_words)
    return len(overlap) / (len(query_words) + 1)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def build_kb_text(kb):
    retrieval_text = kb.get("_retrieval_text", "")
    if isinstance(retrieval_text, str) and retrieval_text.strip():
        return retrieval_text

    symptoms = _as_list(kb.get("symptoms"))
    keyword_variations = _as_list(kb.get("keyword_variations", kb.get("keywords", [])))
    resolution = _as_list(kb.get("resolution"))
    steps = _as_list(kb.get("steps"))
    applies_to = _as_list(kb.get("applies_to"))

    return " ".join([
        kb.get("template_type", "solution"),
        kb.get("title", ""),
        kb.get("summary", ""),
        kb.get("objective", ""),
        kb.get("answer", ""),
        kb.get("cause", ""),
        kb.get("additional_info", ""),
        " ".join(symptoms),
        " ".join(applies_to),
        " ".join(resolution),
        " ".join(steps),
        " ".join(keyword_variations),
    ])


def search_kb(query, embedder, vector_store, top_k=10, include_scores=False):
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
    for result in results:
        kb = result.get("metadata", {})
        if not isinstance(kb, dict):
            continue
        text = build_kb_text(kb)
        semantic_score = float(result.get("score", 0.0))
        keyword_sim = keyword_score(query, text)
        final_score = (0.7 * semantic_score) + (0.3 * keyword_sim)
        matched_terms = sorted(_tokenize(query).intersection(_tokenize(text)))
        scored_results.append((final_score, semantic_score, keyword_sim, matched_terms, kb))

    if not scored_results:
        return []

    scored_results.sort(reverse=True, key=lambda item: item[0])
    max_score = scored_results[0][0]
    filtered_results = [item for item in scored_results if item[0] >= max_score * 0.7]

    seen_titles = set()
    final_results = []
    for final_score, semantic_score, keyword_sim, matched_terms, kb in filtered_results:
        title = kb.get("title")
        if title not in seen_titles:
            if include_scores:
                final_results.append({
                    "kb": kb,
                    "score": round(final_score, 4),
                    "semantic_score": round(semantic_score, 4),
                    "keyword_score": round(keyword_sim, 4),
                    "matched_terms": matched_terms[:8],
                })
            else:
                final_results.append(kb)
            seen_titles.add(title)
        if len(final_results) >= 5:
            break

    return final_results
