"""Disk caching utilities for LLM calls and expensive operations."""

import atexit
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DiskCache:
    """Thread-safe JSON disk cache for caching LLM responses."""

    def __init__(self, cache_file: Path, auto_save_interval: int = 1):
        self.cache_file = Path(cache_file)
        self.auto_save_interval = auto_save_interval
        self._cache: Dict[str, Any] = {}
        self._dirty_count = 0
        self.load()
        atexit.register(self.save)

    def load(self) -> None:
        """Load cache from disk if the file exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded {len(self._cache)} items from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to read cache file {self.cache_file}: {e}. Starting fresh.")
                self._cache = {}
        else:
            self._cache = {}

    def save(self) -> None:
        """Persist cache contents atomically to disk."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.cache_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.cache_file)
            self._dirty_count = 0
            logger.debug(f"Saved {len(self._cache)} items to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {self.cache_file}: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached entry by key."""
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store a key-value entry, auto-flushing to disk periodically."""
        self._cache[key] = value
        self._dirty_count += 1
        if self._dirty_count >= self.auto_save_interval:
            self.save()

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    @staticmethod
    def make_key(*args: Any, **kwargs: Any) -> str:
        """Generate a deterministic MD5 hash key from arbitrary arguments."""
        raw_repr = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(raw_repr.encode("utf-8")).hexdigest()
