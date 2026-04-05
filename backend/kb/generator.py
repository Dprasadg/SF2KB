import json
import time
import re
import ollama


def clean_json_text(text):
    """
    Fix common LLM formatting issues
    """
    text = text.replace('"""', '"')
    text = re.sub(r',\s*}', '}', text)  # trailing comma fix
    return text


def extract_first_json(text):
    """
    Extract first valid JSON object and fix missing braces
    """
    matches = re.findall(r'\{.*', text, re.DOTALL)

    for match in matches:
        candidate = match.strip()

        # Fix missing closing brace
        if not candidate.endswith("}"):
            candidate += "}"

        try:
            return json.loads(candidate)
        except:
            continue

    return None


def normalize_output(parsed, cluster_texts):
    """
    Ensure output follows Smarsh Solution Template
    """

    # Required fields
    parsed.setdefault("title", "Untitled Issue")
    parsed.setdefault("summary", "")
    parsed.setdefault("symptoms", [])
    parsed.setdefault("applies_to", [])
    parsed.setdefault("resolution", [])
    parsed.setdefault("cause", "")
    parsed.setdefault("additional_info", "")
    parsed.setdefault("keywords", [])

    # Ensure lists
    if isinstance(parsed["symptoms"], str):
        parsed["symptoms"] = [parsed["symptoms"]]

    if isinstance(parsed["applies_to"], str):
        parsed["applies_to"] = [parsed["applies_to"]]

    if isinstance(parsed["resolution"], str):
        parsed["resolution"] = [parsed["resolution"]]

    if isinstance(parsed["keywords"], str):
        parsed["keywords"] = [parsed["keywords"]]

    # Fallback for empty resolution
    if not parsed["resolution"]:
        print("[WARNING] Empty resolution — applying fallback")
        parsed["resolution"] = [cluster_texts[0][:200]]

    return parsed


def generate_kb(cluster_texts):
    """
    Generate KB article following Smarsh KCS Solution Template
    """

    # Structure input better for LLM
    structured_input = "\n\n".join([
        f"CASE:\n{text}" for text in cluster_texts
    ])

    prompt = f"""
You are a Knowledge Base Article generator following KCS standards.

CRITICAL INSTRUCTION: KB Synthesis from Multiple Cases
=======================================================

Your input contains MULTIPLE CASES with MULTIPLE RESOLUTION ATTEMPTS.

Your job is to synthesize them into ONE comprehensive KB article that captures:
1. The SHARED ISSUE (core problem common to all)
2. The AGGREGATED RESOLUTION (all tested steps)
3. Confidence levels (primary vs secondary vs edge cases)

STRUCTURED RESOLUTION STRATEGY:
==============================

You will receive resolution steps organized as:

PRIMARY RESOLUTION STEPS (high confidence):
- These steps work across multiple cases
- These should be the PRIMARY sequence in your KB

SECONDARY STEPS (context from case variations):
- Additional steps from individual cases
- Include these as alternatives or follow-ups

EDGE CASES / FALLBACK ACTIONS:
- Rare or conditional steps
- Include with guidance (e.g., "if above doesn't work, try...")

YOUR TASK:
==========
1. Extract the SHARED ISSUE from all cases
2. Create PRIMARY resolution from high-confidence steps
3. ADD SECONDARY steps as alternatives or conditional follow-ups
4. Include EDGE CASES with appropriate context
5. Do NOT truncate any steps — be comprehensive

STRICT RULES:
- Write about ONE issue only
- NEVER leave resolution empty
- Do NOT include customer-specific data
- Use active voice (Do this, Click this)
- Return ONLY ONE JSON object (complete and valid)
- JSON MUST start with {{ and end with }}
- Do NOT truncate

FORMAT:
{{
  "title": "short issue title",
  "summary": "1-2 sentence summary",
  "symptoms": ["symptom 1", "symptom 2"],
  "applies_to": ["product", "module"],
  "resolution": ["step 1", "step 2", "step 3...all unique steps from primary + secondary + edge cases"],
  "cause": "root cause if known",
  "additional_info": "notes about conditional sequences or fallbacks",
  "keywords": ["keyword1", "keyword2"]
}}

RESOLUTION ORDERING:
====================
1. Primary steps first (most common)
2. Secondary/alternative steps next
3. Edge case steps last (with context)

Example (your scenario):
Input:  Primary: [B, A], Secondary: [C], Edge Cases: [D, E]
Output: resolution: ["B", "A", "C", "If unresolved try D or E"]

CASES AND RESOLUTION CONTEXT:
{structured_input}
"""

    for attempt in range(3):
        print(f"[LLM] Attempt {attempt + 1}")

        try:
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2}
            )

            raw_output = response["message"]["content"]

            print(f"\n[LLM RAW OUTPUT]\n{raw_output}\n")

            # Clean bad formatting
            cleaned = clean_json_text(raw_output)

            # Extract JSON
            parsed = extract_first_json(cleaned)

            if not parsed:
                print("[ERROR] Could not extract valid JSON")
                continue

            # Validate minimum required fields
            required_fields = ["title", "symptoms", "resolution"]
            if not all(field in parsed for field in required_fields):
                print("[ERROR] Missing required fields")
                continue

            # Normalize structure
            parsed = normalize_output(parsed, cluster_texts)

            return parsed

        except Exception as e:
            print(f"[ERROR] LLM failure: {e}")
            time.sleep(1)

    print("[ERROR] All attempts failed")
    return None