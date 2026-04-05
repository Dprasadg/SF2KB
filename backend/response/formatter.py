def build_structured_resolution(
    primary_steps: list[str],
    secondary_steps: list[str],
    next_step_hints: list[str],
) -> str:
    """
    Build structured resolution context for LLM consumption.

    Creates a categorized resolution strategy with:
    - Primary steps (high confidence, tested by multiple cases)
    - Secondary steps (single case, but valuable)
    - Fallback/edge cases (next steps that might be needed)
    """
    context_parts = []

    if primary_steps:
        context_parts.append(
            "PRIMARY RESOLUTION STEPS (high confidence):\n"
            + "\n".join(f"- {step}" for step in primary_steps)
        )

    if secondary_steps:
        context_parts.append(
            "SECONDARY STEPS (context from additional case variations):\n"
            + "\n".join(f"- {step}" for step in secondary_steps)
        )

    if next_step_hints:
        unique_next = {s.lower(): s for s in next_step_hints if s}
        context_parts.append(
            "EDGE CASES / FALLBACK ACTIONS:\n"
            + "\n".join(f"- {step}" for step in unique_next.values())
        )

    return "\n\n".join(context_parts)
