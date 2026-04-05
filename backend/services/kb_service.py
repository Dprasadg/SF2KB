import json
from backend.config import KB_STORE_PATH

def get_all_kb():
    with open(KB_STORE_PATH) as f:
        return json.load(f)