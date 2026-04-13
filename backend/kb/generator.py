import json
import re
import time
from typing import Optional

import ollama


GENERIC_APPLIES_TO = {
    "product",
    "module",
    "service",
    "environment",
    "application",
    "system",
}

PAST_TO_IMPERATIVE = {
    "resumed": "Resume",
    "assigned": "Assign",
    "restarted": "Restart",
    "cleared": "Clear",
    "checked": "Check",
    "asked": "Ask",
    "verified": "Verify",
    "increased": "Increase",
    "updated": "Update",
    "reindexed": "Reindex",
    "reset": "Reset",
    "retried": "Retry",
    "confirmed": "Confirm",
}

SOLUTION_MARKERS = (
    "error",
    "warning",
    "fail",
    "fails",
    "failed",
    "failing",
    "unable",
    "cannot",
    "can't",
    "not loading",
    "not working",
    "stuck",
    "missing",
    "crash",
    "slow",
    "delayed",
)
HOW_TO_MARKERS = (
    "how to",
    "how do i",
    "how do we",
    "steps to",
    "procedure to",
    "configure",
    "set up",
    "setup",
    "create",
    "update",
    "change",
)
QA_MARKERS = (
    "what is",
    "why is",
    "why does",
    "when does",
    "where is",
    "can i",
    "is it possible",
    "question:",
)


def clean_json_text(text):
    text = text.replace('"""', '"')
    text = re.sub(r',\s*}', '}', text)
    return text


def extract_first_json(text):
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end = None

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        return None

    try:
        return json.loads(text[start:end + 1].strip())
    except Exception:
        return None


def classify_kb_template(
    cluster_texts: list[str],
    resolution_hints: Optional[list[str]] = None,
    next_step_hints: Optional[list[str]] = None,
) -> str:
    resolution_hints = resolution_hints or []
    next_step_hints = next_step_hints or []
    combined = " ".join(cluster_texts).lower()

    has_solution_markers = any(marker in combined for marker in SOLUTION_MARKERS)
    has_how_to_markers = any(marker in combined for marker in HOW_TO_MARKERS)
    has_qa_markers = any(marker in combined for marker in QA_MARKERS) or "?" in combined
    has_procedural_content = any(str(item).strip() for item in resolution_hints + next_step_hints)

    if has_how_to_markers and not has_solution_markers:
        return "how_to"

    if has_qa_markers and not has_solution_markers and not has_procedural_content:
        return "qa"

    if has_solution_markers or has_procedural_content:
        return "solution"

    if has_qa_markers:
        return "qa"

    return "solution"


def _derive_applies_to(cluster_texts: list[str]) -> list[str]:
    values = []
    seen = set()

    for text in cluster_texts:
        if not isinstance(text, str):
            continue
        chunks = [chunk.strip() for chunk in text.split("|") if chunk.strip()]
        for chunk in chunks:
            lower = chunk.lower()
            if lower.startswith("product:"):
                candidate = chunk.split(":", 1)[1].strip()
            elif lower.startswith("environment:"):
                candidate = chunk.split(":", 1)[1].strip()
            elif lower.startswith("topic:"):
                candidate = chunk.split(":", 1)[1].strip()
            elif lower.startswith("subtopic:"):
                candidate = chunk.split(":", 1)[1].strip()
            else:
                continue

            if not candidate or candidate.lower() in {"na", "n/a", "none", "unknown"}:
                continue

            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                values.append(candidate)

    return values[:5]


def _imperative_step(step: str) -> str:
    value = str(step).strip()
    if not value:
        return value

    if value.lower().startswith("if "):
        return value[0].upper() + value[1:]

    first, *rest = value.split(" ")
    base = PAST_TO_IMPERATIVE.get(first.lower())
    if base:
        return " ".join([base] + rest).strip()

    return value[0].upper() + value[1:]


def _has_generic_applies_to(applies_to: list[str]) -> bool:
    normalized = [str(item).strip().lower() for item in applies_to if str(item).strip()]
    return not normalized or all(item in GENERIC_APPLIES_TO for item in normalized)


def _coerce_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_output(parsed, cluster_texts, template_type: str = "solution"):
    if "keywords" in parsed and "keyword_variations" not in parsed:
        parsed["keyword_variations"] = parsed.get("keywords")

    parsed.setdefault("template_type", template_type)
    parsed.setdefault("title", "Untitled Article")
    parsed.setdefault("applies_to", [])
    parsed.setdefault("additional_info", "")
    parsed.setdefault("internal_to_smarsh", "")
    parsed.setdefault("keyword_variations", [])
    parsed.setdefault("visibility", "Visible in Internal App")
    parsed.setdefault("validation_state", "Not Validated")

    if template_type == "solution":
        parsed.setdefault("summary", "")
        parsed.setdefault("symptoms", [])
        parsed.setdefault("resolution", [])
        parsed.setdefault("cause", "")
    elif template_type == "how_to":
        parsed.setdefault("objective", "")
        parsed.setdefault("steps", [])
        parsed.setdefault("summary", parsed.get("objective", ""))
        parsed.setdefault("resolution", parsed.get("steps", []))
        parsed.setdefault("symptoms", [])
        parsed.setdefault("cause", "")
    else:
        parsed.setdefault("answer", "")
        parsed.setdefault("summary", parsed.get("answer", ""))
        parsed.setdefault("resolution", [])
        parsed.setdefault("symptoms", [])
        parsed.setdefault("cause", "")

    parsed["title"] = str(parsed.get("title", "Untitled Article")).strip()[:200] or "Untitled Article"
    parsed["applies_to"] = _coerce_list(parsed.get("applies_to"))
    parsed["keyword_variations"] = _coerce_list(parsed.get("keyword_variations"))

    if template_type == "solution":
        parsed["symptoms"] = _coerce_list(parsed.get("symptoms"))
        parsed["resolution"] = [_imperative_step(item) for item in _coerce_list(parsed.get("resolution"))]
    elif template_type == "how_to":
        parsed["steps"] = [_imperative_step(item) for item in _coerce_list(parsed.get("steps"))]
        parsed["resolution"] = list(parsed["steps"])
        parsed["summary"] = str(parsed.get("summary") or parsed.get("objective") or "").strip()
    else:
        parsed["answer"] = str(parsed.get("answer", "")).strip()
        parsed["summary"] = str(parsed.get("summary") or parsed.get("answer") or "").strip()
        parsed["resolution"] = []

    if _has_generic_applies_to(parsed.get("applies_to", [])):
        derived = _derive_applies_to(cluster_texts)
        if derived:
            parsed["applies_to"] = derived

    normalized_symptoms = []
    for symptom in _coerce_list(parsed.get("symptoms")):
        item = symptom
        if item.lower().startswith("error") and not item.startswith("Error:"):
            item = "Error: " + item.split(":", 1)[-1].strip(' "')
        if item.lower().startswith("warning") and not item.startswith("Warning:"):
            item = "Warning: " + item.split(":", 1)[-1].strip(' "')
        normalized_symptoms.append(item)
    parsed["symptoms"] = normalized_symptoms

    if template_type == "solution" and not parsed["resolution"]:
        parsed["resolution"] = [
            "If the issue persists after standard checks, gather logs and escalate to Smarsh support with symptoms and applies-to details."
        ]

    parsed["keywords"] = list(parsed.get("keyword_variations", []))
    return parsed


def _build_prompt(template_type: str, structured_input: str) -> str:
    common_header = f"""
You are a professional technical writer producing Knowledge Base articles following Smarsh KCS standards.

Your input contains MULTIPLE SUPPORT CASES with sanitized issue descriptions, troubleshooting context, root-cause observations, and known fixes.

You are writing one publication-ready KB article.

GLOBAL RULES:
- Synthesize and rephrase. Do not copy case text verbatim.
- Remove noise phrases such as "customer reported" or "user said".
- Do not invent products, environments, causes, or fixes that are not supported by the cases.
- Keep `internal_to_smarsh` empty unless the input explicitly contains internal-only steps or escalation details.
- Preserve all high-confidence known fixes in the final instructional content.
- Return ONLY one valid JSON object.
- The response must start with {{ and end with }} with no prose before or after.
"""

    if template_type == "how_to":
        return common_header + f"""
TEMPLATE TYPE: HOW_TO

Write a How To article when the input is about accomplishing a task rather than fixing a broken behavior.

FIELD RULES:
- title: concise user-facing task title
- objective: what the user wants to accomplish, written clearly in 1-2 sentences
- applies_to: specific product/component/environment names only
- steps: imperative action steps in execution order
- additional_info: caveats, prerequisites, or follow-up notes
- internal_to_smarsh: internal-only notes or blank
- keyword_variations: search synonyms
- visibility: "Visible in Internal App"
- validation_state: "Not Validated"

RETURN FORMAT:
{{
  "template_type": "how_to",
  "title": "How to ...",
  "objective": "Clear objective statement.",
  "applies_to": ["Specific Product", "Specific Environment"],
  "steps": ["Imperative step 1", "Imperative step 2"],
  "additional_info": "Helpful notes.",
  "internal_to_smarsh": "",
  "keyword_variations": ["keyword 1", "keyword 2"],
  "visibility": "Visible in Internal App",
  "validation_state": "Not Validated"
}}

CASES AND RESOLUTION CONTEXT:
{structured_input}
"""

    if template_type == "qa":
        return common_header + f"""
TEMPLATE TYPE: QA

Write a Q&A article when the input is primarily a question that needs an explanation rather than a troubleshooting flow.

FIELD RULES:
- title: concise question/topic title
- applies_to: optional but specific product/component/environment names only
- answer: direct answer in professional language
- additional_info: supporting context or caveats
- internal_to_smarsh: internal-only notes or blank
- keyword_variations: search synonyms
- visibility: "Visible in Internal App"
- validation_state: "Not Validated"

RETURN FORMAT:
{{
  "template_type": "qa",
  "title": "Question topic title",
  "applies_to": ["Specific Product"],
  "answer": "Direct answer.",
  "additional_info": "Helpful supporting context.",
  "internal_to_smarsh": "",
  "keyword_variations": ["keyword 1", "keyword 2"],
  "visibility": "Visible in Internal App",
  "validation_state": "Not Validated"
}}

CASES AND RESOLUTION CONTEXT:
{structured_input}
"""

    return common_header + f"""
TEMPLATE TYPE: SOLUTION

Write a Solution article for a broken behavior or issue.

FIELD RULES:
- title: combine affected product/component and observable issue
- summary: 1-2 professional sentences describing the issue and impact
- symptoms: single-thought observable facts, ordered from general to specific
- applies_to: specific product/component/environment names only
- resolution: imperative action steps; every high-confidence known fix must appear in these steps
- cause: concise explanation of why the issue occurred
- additional_info: caveats, edge cases, or follow-up notes
- internal_to_smarsh: internal-only notes or blank
- keyword_variations: search synonyms
- visibility: "Visible in Internal App"
- validation_state: "Not Validated"

RETURN FORMAT:
{{
  "template_type": "solution",
  "title": "User-facing issue title",
  "summary": "Professional issue summary.",
  "symptoms": ["Observable symptom 1", "Observable symptom 2"],
  "applies_to": ["Specific Product", "Specific Environment"],
  "resolution": ["Imperative step 1", "Imperative step 2", "If unresolved, follow-up step"],
  "cause": "Root cause explanation.",
  "additional_info": "Helpful notes or caveats.",
  "internal_to_smarsh": "",
  "keyword_variations": ["keyword 1", "keyword 2"],
  "visibility": "Visible in Internal App",
  "validation_state": "Not Validated"
}}

CASES AND RESOLUTION CONTEXT:
{structured_input}
"""


def generate_kb(cluster_texts, template_type: str = "solution"):
    structured_input = "\n\n".join([f"CASE:\n{text}" for text in cluster_texts])
    prompt = _build_prompt(template_type, structured_input)
    retry_feedback = ""

    for attempt in range(3):
        print(f"[LLM] Attempt {attempt + 1}")
        try:
            response = ollama.chat(
                model="llama3",
                messages=[
                    {
                        "role": "user",
                        "content": prompt + (f"\n\nRETRY FEEDBACK:\n{retry_feedback}" if retry_feedback else ""),
                    }
                ],
                options={"temperature": 0.3},
            )

            raw_output = response["message"]["content"]
            print(f"\n[LLM RAW OUTPUT]\n{raw_output}\n")

            parsed = extract_first_json(clean_json_text(raw_output))
            if not parsed:
                print("[ERROR] Could not extract valid JSON")
                continue

            parsed = normalize_output(parsed, cluster_texts, template_type=template_type)

            if parsed.get("template_type") != template_type:
                parsed["template_type"] = template_type

            if parsed.get("template_type") in {"solution", "how_to"} and _has_generic_applies_to(parsed.get("applies_to", [])):
                print("[ERROR] Generic applies_to detected; retrying generation")
                retry_feedback = (
                    "Previous draft used generic Applies To values. Return specific product, component, or environment names from the case text."
                )
                continue

            return parsed
        except Exception as exc:
            print(f"[ERROR] LLM failure: {exc}")
            time.sleep(1)

    print("[ERROR] All attempts failed")
    return None
