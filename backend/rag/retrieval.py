from typing import Optional


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def build_kb_retrieval(
    kb: dict,
    cluster_texts: Optional[list[str]] = None,
    resolution_hints: Optional[list[str]] = None,
    next_step_hints: Optional[list[str]] = None,
) -> str:
    cluster_texts = cluster_texts or []
    resolution_hints = resolution_hints or []
    next_step_hints = next_step_hints or []

    symptoms = _as_list(kb.get("symptoms"))
    resolution_steps = _as_list(kb.get("resolution"))
    how_to_steps = _as_list(kb.get("steps"))
    keyword_variations = _as_list(kb.get("keyword_variations", kb.get("keywords", [])))
    applies_to = _as_list(kb.get("applies_to"))

    sections = [
        kb.get("template_type", "solution"),
        kb.get("title", ""),
        kb.get("summary", ""),
        kb.get("objective", ""),
        kb.get("answer", ""),
        kb.get("cause", ""),
        kb.get("additional_info", ""),
        " ".join(symptoms),
        " ".join(applies_to),
        " ".join(resolution_steps),
        " ".join(how_to_steps),
        " ".join(keyword_variations),
        " ".join(cluster_texts),
        " ".join(resolution_hints),
        " ".join(next_step_hints),
    ]

    return " ".join(section.strip() for section in sections if section and str(section).strip())
