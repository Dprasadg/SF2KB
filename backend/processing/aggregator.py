from collections import Counter


def combine_issue_texts(cluster_texts: list[str]) -> str:
    return "\n\n".join([f"CASE:\n{text}" for text in cluster_texts if text])


def aggregate_resolution_steps(resolution_hints: list[str]) -> dict:
    """
    Aggregate and rank resolution steps by frequency.

    Parses all resolution hints, counts frequency, and categorizes into:
    - primary: frequency >= 2 (appears in multiple cases)
    - secondary: frequency == 1 (appears in one case)
    """
    if not resolution_hints:
        return {"all_steps": [], "primary": [], "secondary": [], "frequencies": {}}

    all_steps_raw = []
    for hint in resolution_hints:
        if isinstance(hint, str) and hint.strip():
            steps = [s.strip() for s in hint.replace(";", ",").split(",")]
            all_steps_raw.extend([s for s in steps if s])

    normalized_steps = {}
    for step in all_steps_raw:
        normalized = step.lower()
        if normalized not in normalized_steps:
            normalized_steps[normalized] = step

    step_counts = Counter()
    for hint in resolution_hints:
        if isinstance(hint, str) and hint.strip():
            steps = [s.strip() for s in hint.replace(";", ",").split(",")]
            for step in steps:
                if step:
                    step_counts[step.lower()] += 1

    sorted_steps = sorted(step_counts.items(), key=lambda x: x[1], reverse=True)
    primary = [normalized_steps[step] for step, count in sorted_steps if count >= 2]
    secondary = [normalized_steps[step] for step, count in sorted_steps if count == 1]

    return {
        "all_steps": [normalized_steps[step] for step, _ in sorted_steps],
        "primary": primary,
        "secondary": secondary,
        "frequencies": dict(sorted_steps),
    }
