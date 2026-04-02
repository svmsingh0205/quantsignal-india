"""
TTL Cache — In-process dictionary cache with Redis-compatible interface.

Designed so swapping to Redis later requires only changing the backend:
    from backend.engines.ttl_cache import cache
    cache.set("key", value, ttl=5)
    cache.get("key")
    cache.delete("key")
    cache.clear()

When Redis is available, set REDIS_URL env var and this module auto-switches.
"""
from __future__ import annotations

import os
import time
import threading
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REDIS_URL = os.environ.get("REDIS_URL", "")


class _InProcessCache:
    """Thread-safe TTL dictionary cache."""

    def __init__(self):
        self._store: dict[str, tuple[float, Any]] = {}  # key → (expires_at, value)
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, ttl: float = 5.0) -> None:
        expires_at = time.time() + ttl
        with self._lock:
            self._store[key] = (expires_at, value)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            with self._lock:
                self._store.pop(key, None)
            return None
        return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self, prefix: str = "") -> None:
        with self._lock:
            if prefix:
                keys = [k for k in self._store if k.startswith(prefix)]
                for k in keys:
                    del self._store[k]
            else:
                self._store.clear()

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def keys(self, prefix: str = "") -> list[str]:
        now = time.time()
        with self._lock:
            return [k for k, (exp, _) in self._store.items()
                    if (not prefix or k.startswith(prefix)) and exp > now]

    @property
    def backend(self) -> str:
        return "in-process"


class _RedisCache:
    """Redis-backed cache with same interface as _InProcessCache."""

    def __init__(self, url: str):
        import redis
        self._r = redis.from_url(url, decode_responses=False)
        import pickle
        self._pickle = pickle

    def set(self, key: str, value: Any, ttl: float = 5.0) -> None:
        self._r.setex(key, int(ttl), self._pickle.dumps(value))

    def get(self, key: str) -> Optional[Any]:
        raw = self._r.get(key)
        if raw is None:
            return None
        try:
            return self._pickle.loads(raw)
        except Exception:
            return None

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def clear(self, prefix: str = "") -> None:
        if prefix:
            keys = self._r.keys(f"{prefix}*")
            if keys:
                self._r.delete(*keys)
        else:
            self._r.flushdb()

    def exists(self, key: str) -> bool:
        return bool(self._r.exists(key))

    def keys(self, prefix: str = "") -> list[str]:
        pattern = f"{prefix}*" if prefix else "*"
        return [k.decode() if isinstance(k, bytes) else k for k in self._r.keys(pattern)]

    @property
    def backend(self) -> str:
        return "redis"


def _build_cache():
    if _REDIS_URL:
        try:
            c = _RedisCache(_REDIS_URL)
            c.set("__ping__", 1, ttl=1)
            logger.info("TTL cache: Redis connected at %s", _REDIS_URL)
            return c
        except Exception as e:
            logger.warning("Redis unavailable (%s) — falling back to in-process cache", e)
    return _InProcessCache()


# Singleton — import and use directly
cache = _build_cache()

# Convenience TTL constants (seconds)
TTL_TICK      = 3      # live price tick
TTL_OHLCV_5M  = 60     # 5-minute bars
TTL_OHLCV_1D  = 300    # daily bars (5 min)
TTL_OHLCV_1W  = 900    # weekly bars (15 min)
TTL_INDICATOR = 30     # computed indicators
TTL_SIGNAL    = 10     # generated signals
TTL_BUNDLE    = 600    # full analysis bundle (10 min)
TTL_PEER      = 300    # peer comparison data
TTL_NEWS      = 3600   # news sentiment (1 hour)
TTL_MACRO     = 900    # macro context (15 min)
