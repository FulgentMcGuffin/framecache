"""YAML-driven configuration and backend factory for FrameCache.

Typical usage::

    config = CacheConfig.from_yaml("cache.yaml")
    fc = FrameCache.from_config(config)

YAML format
-----------
Redis backend::

    backend_type: redis
    framecache_key: MyCache      # optional, default "FrameCache"
    use_hash_keys: false         # optional
    default_ttl_hours: 1.0       # optional

    host: localhost
    port: 6379
    db: 0
    password: null               # optional

SQLite backend::

    backend_type: sqlite
    framecache_key: MyCache
    use_hash_keys: false
    default_ttl_hours: 24.0      # null or omit for no expiry

    db_path: ./cache/framecache.db   # ":memory:" for an in-memory db
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class CacheConfig:
    """Unified configuration for a FrameCache storage backend.

    Attributes:
        backend_type:       ``"redis"`` or ``"sqlite"``.
        framecache_key:     Prefix/namespace used for all cache keys.
        use_hash_keys:      Whether to SHA-256 hash the argument portion of
                            cache_instance_ids (keeps Redis key length bounded).
        default_ttl_hours:  Lifetime of cached entries in hours.  ``None``
                            means no expiry (SQLite only; Redis always requires
                            a TTL or the entry persists until eviction).

    Redis-specific:
        host, port, db, password

    SQLite-specific:
        db_path:  Path to the ``.db`` file, or ``":memory:"``.
    """

    backend_type: str = "redis"
    framecache_key: str = "FrameCache"
    use_hash_keys: bool = False
    default_ttl_hours: float | None = 1.0

    # Redis
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None

    # SQLite
    db_path: str = "./framecache.db"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CacheConfig":
        """Load a :class:`CacheConfig` from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully populated :class:`CacheConfig` instance.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            ValueError: if ``backend_type`` is not ``"redis"`` or ``"sqlite"``.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for from_yaml(). "
                "Install it with: pip install pyyaml"
            ) from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cache config file not found: {path}")

        with path.open() as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheConfig":
        """Build a :class:`CacheConfig` from a plain dictionary.

        Unknown keys are silently ignored so users can annotate YAML files
        with extra documentation fields.
        """
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}

        backend_type = str(filtered.get("backend_type", "redis")).lower()
        if backend_type not in ("redis", "sqlite"):
            raise ValueError(
                f"backend_type must be 'redis' or 'sqlite', got {backend_type!r}"
            )
        filtered["backend_type"] = backend_type

        # Resolve ~ and relative paths for db_path
        if "db_path" in filtered and filtered["db_path"] != ":memory:":
            filtered["db_path"] = str(Path(filtered["db_path"]).expanduser())

        return cls(**filtered)

    # ------------------------------------------------------------------
    # Backend factory
    # ------------------------------------------------------------------

    @property
    def default_ttl(self) -> timedelta | None:
        """The default TTL as a :class:`timedelta`, or ``None`` for no expiry."""
        if self.default_ttl_hours is None:
            return None
        return timedelta(hours=self.default_ttl_hours)

    def build_backend(self):
        """Construct and return the appropriate :class:`CacheBackend`.

        Returns:
            A :class:`~framecache.backends.RedisBackend` or
            :class:`~framecache.backends.SQLiteBackend` instance.
        """
        from framecache.backends import RedisBackend, SQLiteBackend  # noqa: E402

        if self.backend_type == "redis":
            import redis
            client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
            )
            return RedisBackend(client)

        if self.backend_type == "sqlite":
            return SQLiteBackend(self.db_path)

        raise ValueError(f"Unknown backend_type: {self.backend_type!r}")

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (round-trips with :meth:`from_dict`)."""
        import dataclasses
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file.

        Args:
            path: Destination path.  Parent directories are created if needed.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for to_yaml(). "
                "Install it with: pip install pyyaml"
            ) from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.safe_dump(self.to_dict(), fh, default_flow_style=False, sort_keys=False)
