import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from backend.config import KB_RUNTIME_METADATA_PATH, KB_STORE_PATH


def _load_json_list(path: Path) -> list:
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, list) else []


def _atomic_write_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def _split_kb_payload(kb: dict) -> tuple[dict, dict]:
    kb_id = str(kb.get("kb_id") or uuid4().hex)

    article = {"kb_id": kb_id}
    runtime = {"kb_id": kb_id}

    for key, value in kb.items():
        if key == "kb_id":
            continue
        if str(key).startswith("_"):
            runtime[key] = value
        else:
            article[key] = value

    return article, runtime


def load_kbs(include_runtime: bool = True) -> list:
    """
    Load KB articles.

    include_runtime=True merges runtime metadata fields (keys starting with "_")
    from the runtime metadata store for internal pipeline/search use.
    """
    articles = _load_json_list(Path(KB_STORE_PATH))
    if not include_runtime:
        return [kb for kb in articles if isinstance(kb, dict)]

    runtime_entries = _load_json_list(Path(KB_RUNTIME_METADATA_PATH))
    runtime_by_id = {
        str(entry.get("kb_id")): entry
        for entry in runtime_entries
        if isinstance(entry, dict) and entry.get("kb_id")
    }

    merged = []
    for kb in articles:
        if not isinstance(kb, dict):
            continue

        merged_kb = dict(kb)
        kb_id = str(merged_kb.get("kb_id", "")).strip()

        # Backward compatibility for old files where runtime fields were inline.
        if kb_id and kb_id in runtime_by_id:
            runtime = runtime_by_id[kb_id]
            for key, value in runtime.items():
                if key != "kb_id" and str(key).startswith("_"):
                    merged_kb[key] = value

        merged.append(merged_kb)

    return merged


def save_kb(kb: dict) -> None:
    articles = _load_json_list(Path(KB_STORE_PATH))
    runtime_entries = _load_json_list(Path(KB_RUNTIME_METADATA_PATH))

    article, runtime = _split_kb_payload(kb)

    articles.append(article)

    runtime_entries = [
        entry
        for entry in runtime_entries
        if not (isinstance(entry, dict) and str(entry.get("kb_id")) == article["kb_id"])
    ]

    # Store runtime entry only when internal fields exist.
    if any(str(key).startswith("_") for key in runtime.keys()):
        runtime_entries.append(runtime)

    _atomic_write_json(Path(KB_STORE_PATH), articles)
    _atomic_write_json(Path(KB_RUNTIME_METADATA_PATH), runtime_entries)


def update_kb_fields(kb_id: str, updates: dict) -> Optional[dict]:
    if not kb_id:
        return None

    kb_id = str(kb_id).strip()
    if not kb_id:
        return None

    articles = _load_json_list(Path(KB_STORE_PATH))
    updated_article = None

    for index, article in enumerate(articles):
        if not isinstance(article, dict):
            continue
        if str(article.get("kb_id", "")).strip() != kb_id:
            continue

        merged = dict(article)
        merged.update(updates or {})
        articles[index] = merged
        updated_article = merged
        break

    if updated_article is None:
        return None

    _atomic_write_json(Path(KB_STORE_PATH), articles)
    return updated_article
