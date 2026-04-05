from typing import Optional


def build_kb_retrieval(
    kb: dict,
    cluster_texts: Optional[list[str]] = None,
    resolution_hints: Optional[list[str]] = None,
    next_step_hints: Optional[list[str]] = None,
) -> str:
    """
    Build rich retrieval text for KB vectorization.

    Supports both:
    - Pipeline path: kb + cluster context + hints
    - Rebuild path: kb only
    """
    cluster_texts = cluster_texts or []
    resolution_hints = resolution_hints or []
    next_step_hints = next_step_hints or []

    symptoms = kb.get("symptoms", [])
    if isinstance(symptoms, str):
        symptoms = [symptoms]

    resolution_steps = kb.get("resolution", [])
    if isinstance(resolution_steps, str):
        resolution_steps = [resolution_steps]

    keywords = kb.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]

    sections = [
        kb.get("title", ""),
        kb.get("summary", ""),
        kb.get("cause", ""),
        " ".join(symptoms),
        " ".join(resolution_steps),
        " ".join(keywords),
        " ".join(cluster_texts),
        " ".join(resolution_hints),
        " ".join(next_step_hints),
    ]

    return " ".join(section.strip() for section in sections if section and str(section).strip())
