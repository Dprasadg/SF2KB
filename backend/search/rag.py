import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - optional fallback if dependency is missing
    BM25Okapi = None

# Priority score boosts added to final relevance score
_PRIORITY_BOOST = {
    "critical": 0.08,
    "high": 0.05,
    "medium": 0.02,
    "low": 0.0,
}

_FIELD_WEIGHTS = {
    "title": 4.0,
    "keyword_variations": 3.0,
    "applies_to": 2.0,
    "symptoms": 2.0,
    "objective": 2.0,
    "answer": 2.0,
    "summary": 2.0,
    "resolution": 1.0,
    "steps": 1.0,
    "cause": 1.0,
    "additional_info": 1.0,
}

_INTENT_EXPANSION = {
    "solution": ["error", "fail", "unable"],
    "how_to": ["configure", "steps", "setup"],
    "qa": ["what", "why", "answer"],
}

# Stemming suffix rules (longest match first — order matters)
_STEM_RULES = [
    ("nesses", ""), ("ational", "ate"), ("tional", "tion"),
    ("ingness", ""), ("ations", "ate"), ("ation", "ate"),
    ("ating", "ate"), ("ated", "ate"), ("ates", "ate"),
    ("izing", "ize"), ("ized", "ize"), ("izes", "ize"),
    ("ising", "ise"), ("ised", "ise"), ("ises", "ise"),
    ("ings", ""), ("ing", ""), ("edly", ""), ("edly", ""),
    ("ness", ""), ("ful", ""), ("ment", ""), ("ments", ""),
    ("less", ""), ("ers", ""), ("er", ""), ("est", ""),
    ("ies", "y"), ("ied", "y"), ("ying", "y"),
    ("sses", "ss"), ("ses", "s"), ("ves", "f"),
    ("ally", ""), ("ically", "ic"), ("ically", "ic"),
    ("ly", ""), ("tion", ""), ("tions", ""),
    ("ed", ""), ("es", ""), ("s", ""),
]


def _stem(word: str) -> str:
    """Apply simple suffix-stripping stemming (Porter-lite). Keeps root >= 3 chars."""
    if len(word) <= 3:
        return word
    for suffix, replacement in _STEM_RULES:
        if word.endswith(suffix):
            root = word[: len(word) - len(suffix)] + replacement
            if len(root) >= 3:
                return root
    return word


def _tokenize(text: str) -> Set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    return tokens | {_stem(t) for t in tokens}


def _tokenize_for_bm25(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if token]


def _detect_query_intent(query: str) -> str:
    q = str(query or "").strip().lower()
    if not q:
        return "solution"

    if (
        q.startswith("how to")
        or "configure" in q
        or "set up" in q
        or "setup" in q
        or "steps" in q
    ):
        return "how_to"

    if (
        "?" in q
        or q.startswith("what")
        or q.startswith("why")
        or q.startswith("when")
        or q.startswith("where")
        or q.startswith("who")
        or q.startswith("which")
    ):
        return "qa"

    return "solution"


def _expand_query(query: str) -> str:
    intent = _detect_query_intent(query)
    additions = _INTENT_EXPANSION.get(intent, _INTENT_EXPANSION["solution"])
    raw_tokens = _tokenize_for_bm25(query)
    expanded = raw_tokens + [token for token in additions if token not in raw_tokens]
    return " ".join(expanded)


def keyword_score(query: str, text: str) -> float:
    query_words = _tokenize(query)
    text_words = _tokenize(text)
    overlap = query_words.intersection(text_words)
    return len(overlap) / (len(query_words) + 1)


def _title_keyword_score(query: str, title: str) -> float:
    """Title-weighted keyword score: 2× boost over body text."""
    query_words = _tokenize(query)
    title_words = _tokenize(title)
    overlap = query_words.intersection(title_words)
    return (2 * len(overlap)) / (len(query_words) + 1)


def _kb_key(kb: dict) -> str:
    kb_id = str(kb.get("kb_id", "")).strip()
    if kb_id:
        return f"id:{kb_id}"
    title = str(kb.get("title", "")).strip().lower()
    return f"title:{title}"


def _field_weighted_keyword_score(query: str, kb: dict) -> Tuple[float, List[str]]:
    query_words = _tokenize(query)
    if not query_words:
        return 0.0, []

    weighted_sum = 0.0
    total_weight = 0.0
    matched_fields: List[str] = []

    for field, weight in _FIELD_WEIGHTS.items():
        value = kb.get(field)
        values = _as_list(value) if isinstance(value, list) else [str(value or "")] 
        tokens: Set[str] = set()
        for item in values:
            tokens |= _tokenize(item)

        if not tokens:
            continue

        overlap = query_words.intersection(tokens)
        field_score = len(overlap) / (len(query_words) + 1)
        weighted_sum += field_score * weight
        total_weight += weight
        if overlap:
            matched_fields.append(field)

    if total_weight == 0:
        return 0.0, []

    return weighted_sum / total_weight, matched_fields


def _bm25_candidates(query: str, metadata_list: List[dict], candidate_k: int) -> Dict[str, Tuple[float, dict]]:
    if not metadata_list or BM25Okapi is None:
        return {}

    docs: List[List[str]] = []
    valid_kbs: List[dict] = []
    for kb in metadata_list:
        if not isinstance(kb, dict):
            continue
        tokens = _tokenize_for_bm25(build_kb_text(kb))
        docs.append(tokens)
        valid_kbs.append(kb)

    if not docs:
        return {}

    bm25 = BM25Okapi(docs)
    query_tokens = _tokenize_for_bm25(query)
    if not query_tokens:
        return {}

    scores = bm25.get_scores(query_tokens)
    if len(scores) == 0:
        return {}

    max_score = max(float(s) for s in scores) if len(scores) else 0.0
    if max_score <= 0:
        return {}

    ranked = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:candidate_k]

    out: Dict[str, Tuple[float, dict]] = {}
    for idx, score in ranked:
        kb = valid_kbs[idx]
        key = _kb_key(kb)
        out[key] = (float(score) / max_score, kb)
    return out


def _as_list(value) -> List[str]:
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


def _log_null_query(query: str, log_dir: Optional[str] = None) -> None:
    """Append zero-result query to null_queries.log for coverage gap analysis."""
    try:
        log_path = Path(log_dir) / "null_queries.log" if log_dir else Path(__file__).parent.parent / "data" / "logs" / "null_queries.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{timestamp}\t{query}\n")
    except Exception:
        pass


def search_kb(query, embedder, vector_store, top_k=10, candidate_k=40, include_scores=False, log_dir=None):
    if not query or not str(query).strip():
        return []

    query = str(query).strip()
    query_enhanced = _expand_query(query)
    top_k = max(1, int(top_k or 10))
    candidate_k = max(top_k, int(candidate_k or 40), 30)

    try:
        q_vec = embedder.encode([query_enhanced])[0]
    except Exception:
        return []

    try:
        semantic_results = vector_store.search(q_vec, k=candidate_k)
    except Exception:
        return []

    bm25_results = _bm25_candidates(query_enhanced, getattr(vector_store, "metadata", []) or [], candidate_k)

    if not semantic_results and not bm25_results:
        _log_null_query(query, log_dir)
        return []

    semantic_candidates: Dict[str, Tuple[float, dict]] = {}
    max_semantic = max((float(item.get("score", 0.0)) for item in semantic_results), default=0.0)
    sem_norm = max_semantic if max_semantic > 0 else 1.0
    for item in semantic_results:
        kb = item.get("metadata", {})
        if not isinstance(kb, dict):
            continue
        key = _kb_key(kb)
        semantic_candidates[key] = (float(item.get("score", 0.0)) / sem_norm, kb)

    all_keys = set(semantic_candidates.keys()) | set(bm25_results.keys())
    scored_results = []
    for key in all_keys:
        semantic_score, kb_sem = semantic_candidates.get(key, (0.0, None))
        bm25_score, kb_bm25 = bm25_results.get(key, (0.0, None))
        kb = kb_sem or kb_bm25
        if not isinstance(kb, dict):
            continue

        text = build_kb_text(kb)
        title = kb.get("title", "")

        # Blend title overlap with weighted field matching.
        title_kw = _title_keyword_score(query, title)
        field_kw, matched_fields = _field_weighted_keyword_score(query, kb)
        keyword_sim = max(title_kw, field_kw)

        # Priority boost
        priority = str(kb.get("priority", "")).strip().lower()
        priority_boost = _PRIORITY_BOOST.get(priority, 0.0)

        # confidence score used as tie-breaker (small additive factor)
        confidence = float(kb.get("confidence_score", 0.5))
        confidence_factor = confidence * 0.01  # max 0.01 impact

        final_score = (
            (0.5 * semantic_score)
            + (0.25 * keyword_sim)
            + (0.2 * bm25_score)
            + priority_boost
            + confidence_factor
        )

        matched_tokens = _tokenize(query).intersection(_tokenize(text))
        # Only return original non-stemmed tokens that appear in the query
        query_raw = set(re.findall(r"[a-z0-9]+", query.lower()))
        text_raw = set(re.findall(r"[a-z0-9]+", text.lower()))
        matched_terms = sorted(query_raw.intersection(text_raw))

        scored_results.append((
            final_score,
            semantic_score,
            keyword_sim,
            bm25_score,
            matched_terms,
            matched_fields,
            confidence,
            kb,
        ))

    if not scored_results:
        _log_null_query(query, log_dir)
        return []

    # Sort by final score, then confidence as tie-breaker
    scored_results.sort(reverse=True, key=lambda item: (item[0], item[6]))
    max_score = scored_results[0][0]
    filtered_results = [item for item in scored_results if item[0] >= max_score * 0.7]

    seen_keys: Set[str] = set()
    final_results = []
    for final_score, semantic_score, keyword_sim, bm25_score, matched_terms, matched_fields, confidence, kb in filtered_results:
        dedupe_key = _kb_key(kb)
        if dedupe_key not in seen_keys:
            if include_scores:
                final_results.append({
                    "kb": kb,
                    "score": round(final_score, 4),
                    "semantic_score": round(semantic_score, 4),
                    "keyword_score": round(keyword_sim, 4),
                    "bm25_score": round(bm25_score, 4),
                    "matched_fields": matched_fields,
                    "matched_terms": matched_terms[:8],
                })
            else:
                final_results.append(kb)
            seen_keys.add(dedupe_key)
        if len(final_results) >= top_k:
            break

    if not final_results:
        _log_null_query(query, log_dir)

    return final_results
