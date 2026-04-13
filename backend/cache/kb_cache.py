"""In-memory cache for KB metadata to optimize search performance."""

import json
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class KBMetadataCache:
    """In-memory cache for KB articles with thread-safe updates."""

    def __init__(self, kb_store_path: str):
        self.kb_store_path = Path(kb_store_path)
        self._cache: Dict[str, dict] = {}  # kb_id -> kb dict
        self._title_index: Dict[str, dict] = {}  # title -> kb dict
        self._lock = Lock()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load KB metadata from disk into cache."""
        try:
            if self.kb_store_path.exists():
                with self.kb_store_path.open(encoding="utf-8") as f:
                    kbs = json.load(f)
                    if isinstance(kbs, list):
                        for kb in kbs:
                            if isinstance(kb, dict):
                                kb_id = str(kb.get("kb_id", "")).strip()
                                title = str(kb.get("title", "")).strip()
                                
                                if kb_id:
                                    self._cache[kb_id] = kb
                                if title:
                                    self._title_index[title] = kb
                        
                        logger.debug(f"Loaded {len(self._cache)} KBs into cache")
        except Exception as exc:
            logger.error(f"Failed to load KB metadata cache: {exc}", exc_info=True)

    def reload(self) -> None:
        """Reload cache from disk (call after KB updates)."""
        with self._lock:
            self._cache.clear()
            self._title_index.clear()
            self._load_from_disk()
            logger.info("KB metadata cache reloaded from disk")

    def get_by_id(self, kb_id: str) -> Optional[dict]:
        """Get KB by ID from cache."""
        with self._lock:
            return self._cache.get(str(kb_id).strip())

    def get_by_title(self, title: str) -> Optional[dict]:
        """Get KB by title from cache."""
        with self._lock:
            return self._title_index.get(str(title).strip())

    def get_all(self) -> List[dict]:
        """Get all KBs from cache."""
        with self._lock:
            return list(self._cache.values())

    def update_kb(self, kb: dict) -> None:
        """Update a single KB in cache."""
        with self._lock:
            kb_id = str(kb.get("kb_id", "")).strip()
            title = str(kb.get("title", "")).strip()
            
            if kb_id:
                self._cache[kb_id] = kb
            if title:
                self._title_index[title] = kb
            
            logger.debug(f"Updated KB in cache: {title or kb_id}")

    def get_by_id_or_title(self, kb_id: Optional[str] = None, title: Optional[str] = None) -> Optional[dict]:
        """Get KB by ID (primary) or title (fallback)."""
        with self._lock:
            if kb_id:
                kb_id_str = str(kb_id).strip()
                if kb_id_str in self._cache:
                    return self._cache[kb_id_str]
            
            if title:
                title_str = str(title).strip()
                if title_str in self._title_index:
                    return self._title_index[title_str]
            
            return None
