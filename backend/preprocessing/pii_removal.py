import re
from typing import Optional


EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
ACCOUNT_PATTERN = re.compile(r"\baccount\s*id\s*[:#-]?\s*[a-zA-Z0-9_-]+\b", flags=re.IGNORECASE)


def remove_pii(text: Optional[str]) -> str:
    if text is None:
        return ""

    sanitized = str(text)
    sanitized = EMAIL_PATTERN.sub("[EMAIL]", sanitized)
    sanitized = PHONE_PATTERN.sub("[PHONE]", sanitized)
    sanitized = ACCOUNT_PATTERN.sub("[ACCOUNT]", sanitized)
    return sanitized