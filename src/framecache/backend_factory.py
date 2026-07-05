"""Factory for constructing :class:`CacheBackend` instances from configuration.

Clients should prefer this module (or :meth:`FrameCache.from_config` /
:meth:`FrameCache.from_yaml`) over importing concrete backend classes such as
``RedisBackend``, ``SQLiteBackend``, or ``DuckDBBackend`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framecache.backends import CacheBackend
    from framecache.cache_config import CacheConfig


class BackendFactory:
    """Instantiate :class:`CacheBackend` implementations from :class:`CacheConfig`.

    Example::

        from framecache import BackendFactory, CacheConfig, FrameCache

        config = CacheConfig.from_yaml("cache.yaml")
        fc = FrameCache(BackendFactory.create(config))
        # or simply:
        fc = FrameCache.from_config(config)
    """

    _SUPPORTED = frozenset({"redis", "sqlite", "duckdb"})

    @classmethod
    def supported_backends(cls) -> frozenset[str]:
        """Return the set of ``backend_type`` values accepted by :meth:`create`."""
        return cls._SUPPORTED

    @classmethod
    def create(cls, config: "CacheConfig") -> "CacheBackend":
        """Build the :class:`CacheBackend` described by ``config``.

        Args:
            config: A populated :class:`~framecache.cache_config.CacheConfig`.

        Returns:
            A backend instance ready to pass to :class:`FrameCache`.

        Raises:
            ValueError: if ``config.backend_type`` is not supported.
        """
        backend_type = config.backend_type.lower()
        if backend_type not in cls._SUPPORTED:
            raise ValueError(
                f"backend_type must be one of {sorted(cls._SUPPORTED)}, "
                f"got {backend_type!r}"
            )

        if backend_type == "redis":
            import redis
            from framecache.backends import RedisBackend

            client = redis.Redis(
                host=config.host,
                port=config.port,
                db=config.db,
                password=config.password,
            )
            return RedisBackend(client)

        if backend_type == "sqlite":
            from framecache.backends import SQLiteBackend

            return SQLiteBackend(config.db_path)

        if backend_type == "duckdb":
            from framecache.backends import DuckDBBackend

            return DuckDBBackend(config.db_path)

        raise ValueError(f"Unknown backend_type: {backend_type!r}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CacheBackend":
        """Load a YAML config file and return the configured backend."""
        from framecache.cache_config import CacheConfig

        return cls.create(CacheConfig.from_yaml(path))

    @classmethod
    def create_framecache(cls, config: "CacheConfig"):
        """Build a :class:`FrameCache` from ``config`` (convenience wrapper)."""
        from framecache.framecache import FrameCache

        return FrameCache.from_config(config)
