import re
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import KB_DUPLICATE_THRESHOLD


GENERIC_APPLIES_TO = {
    "product",
    "module",
    "service",
    "environment",
    "application",
    "system",
}
IMPERATIVE_VERBS = {
    "add",
    "assign",
    "check",
    "clear",
    "click",
    "collect",
    "confirm",
    "contact",
    "create",
    "disable",
    "enable",
    "escalate",
    "gather",
    "increase",
    "log",
    "navigate",
    "open",
    "press",
    "reindex",
    "remove",
    "restart",
    "resume",
    "retry",
    "review",
    "run",
    "select",
    "set",
    "update",
    "verify",
}
CUSTOMER_NOISE_PATTERNS = (
    re.compile(r"\bcustomer reports?\b", re.IGNORECASE),
    re.compile(r"\bthe customer\b", re.IGNORECASE),
    re.compile(r"\buser said\b", re.IGNORECASE),
)
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _iter_public_text_fields(kb: dict):
    for key in ("title", "summary", "cause", "additional_info", "objective", "answer"):
        value = kb.get(key)
        if isinstance(value, str) and value.strip():
            yield key, value

    for key in ("symptoms", "applies_to", "resolution", "keyword_variations", "steps"):
        for item in _as_list(kb.get(key)):
            yield key, item


def _contains_pii(text: str) -> bool:
    return bool(
        EMAIL_PATTERN.search(text)
        or PHONE_PATTERN.search(text)
        or IPV4_PATTERN.search(text)
    )


def _coerce_matching_vector(vec, shape):
    try:
        arr = np.asarray(vec, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.shape != shape:
        return None
    return arr


def _normalize_for_match(text: str) -> str:
    return NON_WORD_RE.sub(" ", str(text).lower()).strip()


def _matches_known_fix(candidate: str, generated_steps: list[str]) -> bool:
    normalized_candidate = _normalize_for_match(candidate)
    if not normalized_candidate:
        return True

    for step in generated_steps:
        normalized_step = _normalize_for_match(step)
        if normalized_candidate in normalized_step or normalized_step in normalized_candidate:
            return True

        candidate_words = set(normalized_candidate.split())
        step_words = set(normalized_step.split())
        if candidate_words and len(candidate_words.intersection(step_words)) >= max(2, min(4, len(candidate_words))):
            return True

    return False


def is_duplicate_kb(cluster_vec: np.ndarray, kb_list: list, threshold: float = KB_DUPLICATE_THRESHOLD) -> bool:
    if not kb_list:
        return False

    cluster_vecs = [
        kb.get("_cluster_vec")
        for kb in kb_list
        if isinstance(kb, dict) and "_cluster_vec" in kb
    ]

    if not cluster_vecs:
        return False

    cluster_vec = np.asarray(cluster_vec, dtype=np.float32)
    cluster_vecs = [
        arr
        for arr in (_coerce_matching_vector(vec, cluster_vec.shape) for vec in cluster_vecs)
        if arr is not None
    ]

    if not cluster_vecs:
        return False

    similarities = cosine_similarity([cluster_vec], np.array(cluster_vecs))[0]
    return np.max(similarities) > threshold


def validate_kb_template(
    kb: dict,
    template_type: Optional[str] = None,
    known_fix_steps: Optional[list[str]] = None,
) -> tuple[bool, list[str]]:
    issues: list[str] = []

    if not isinstance(kb, dict):
        return False, ["KB payload is not a dictionary"]

    template = str(template_type or kb.get("template_type") or "solution").strip().lower()
    known_fix_steps = [step for step in (known_fix_steps or []) if str(step).strip()]

    required_present = ["title", "keyword_variations", "visibility", "validation_state", "internal_to_smarsh"]
    required_non_empty_by_template = {
        "solution": ["summary", "symptoms", "applies_to", "resolution"],
        "how_to": ["objective", "applies_to", "steps"],
        "qa": ["answer"],
    }

    for key in required_present:
        if key not in kb:
            issues.append(f"Missing field: {key}")

    for key in required_non_empty_by_template.get(template, required_non_empty_by_template["solution"]):
        value = kb.get(key)
        if value is None:
            issues.append(f"Missing required field: {key}")
            continue
        if isinstance(value, str) and not value.strip():
            issues.append(f"Required field is empty: {key}")
        if isinstance(value, list) and len([item for item in value if str(item).strip()]) == 0:
            issues.append(f"Required field is empty: {key}")

    title = str(kb.get("title", "")).strip()
    if len(title) > 200:
        issues.append("Title exceeds 200 character limit")

    allowed_visibility = {
        "Visible in Internal App",
        "Visible in Public KB",
        "Visible to Customer",
    }
    if kb.get("visibility") not in allowed_visibility:
        issues.append("Invalid visibility value")

    allowed_validation_states = {"Not Validated", "Validated"}
    if kb.get("validation_state") not in allowed_validation_states:
        issues.append("Invalid validation_state value")

    applies_to = _as_list(kb.get("applies_to"))
    if template in {"solution", "how_to"}:
        if not applies_to:
            issues.append("Applies To must contain at least one specific value")
        if any(item.lower() in GENERIC_APPLIES_TO for item in applies_to):
            issues.append("Applies To contains generic placeholder values")

    for symptom in _as_list(kb.get("symptoms")):
        lower = symptom.lower()
        if lower.startswith("error") and not symptom.startswith("Error:"):
            issues.append("Error symptom must start with 'Error:'")
        if lower.startswith("warning") and not symptom.startswith("Warning:"):
            issues.append("Warning symptom must start with 'Warning:'")

    procedural_steps = _as_list(kb.get("resolution" if template == "solution" else "steps"))
    if template in {"solution", "how_to"}:
        for step in procedural_steps:
            first_word = step.split(" ", 1)[0].strip(":.").lower()
            if not (first_word in IMPERATIVE_VERBS or first_word == "if"):
                issues.append(f"Step should start with an action verb: {step[:80]}")

    if template == "solution" and known_fix_steps:
        missing = [step for step in known_fix_steps if not _matches_known_fix(step, procedural_steps)]
        if missing:
            issues.append("Known fixes missing from resolution: " + "; ".join(missing[:3]))

    for field, text in _iter_public_text_fields(kb):
        if _contains_pii(text):
            issues.append(f"Potential PII detected in field: {field}")
        if any(pattern.search(text) for pattern in CUSTOMER_NOISE_PATTERNS):
            issues.append(f"Customer-specific/noise phrasing detected in field: {field}")

    if template == "qa" and procedural_steps:
        issues.append("Q&A articles should not include procedural steps")

    return len(issues) == 0, issues


def validate_smarsh_solution_template(kb: dict) -> tuple[bool, list[str]]:
    return validate_kb_template(kb, template_type="solution")
