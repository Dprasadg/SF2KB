import json
import os
from backend.config import KB_STORE_PATH

def load_kbs():
    if not os.path.exists(KB_STORE_PATH):
        return []
    with open(KB_STORE_PATH) as f:
        return json.load(f)

def save_kb(kb):
    data = load_kbs()
    data.append(kb)

    with open(KB_STORE_PATH, "w") as f:
        json.dump(data, f, indent=2)