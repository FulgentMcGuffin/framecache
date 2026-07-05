"""Cache storage backends for FrameCache.

The :class:`CacheBackend` Protocol defines the minimal key/value-store interface
that :class:`FrameCache` requires.  Concrete implementations are shipped for
Redis, SQLite, and DuckDB.  Use :class:`~framecache.backend_factory.BackendFactory`
to construct backends from configuration rather than importing these classes
directly.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable


def _glob_to_sql_like(pattern: str) -> str:
    """Translate a Redis/glob wildcard pattern to an SQL LIKE pattern.

    ``*``  → ``%``       (any sequence)
    ``?``  → ``_``       (any single character)
    ``%``  → ``\\%``     (escaped SQL metachar, literal percent)
    ``_``  → ``\\_``     (escaped SQL metachar, literal underscore)
    ``\\x``→ ``x``       (Redis-glob-escaped literal)
    """
    result: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            if nxt == "%":
                result.append("\\%")
            elif nxt == "_":
                result.append("\\_")
            elif nxt == "\\":
                result.append("\\\\")
            else:
                result.append(nxt)
            i += 2
        elif ch == "*":
            result.append("%")
            i += 1
        elif ch == "?":
            result.append("_")
            i += 1
        elif ch == "%":
            result.append("\\%")
            i += 1
        elif ch == "_":
            result.append("\\_")
            i += 1
        else:
            result.append(ch)
            i += 1
    return "".join(result)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _expiry_iso(ttl: timedelta) -> str:
    return (datetime.now() + ttl).isoformat()


# Shared schema for file/SQL backends (SQLite and DuckDB).
_SQL_BACKEND_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS cache_entries (
        cache_id   TEXT PRIMARY KEY,
        value      BLOB NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries (expires_at)",
    """
    CREATE TABLE IF NOT EXISTS cache_hashes (
        hash_name TEXT NOT NULL,
        field     TEXT NOT NULL,
        value     TEXT NOT NULL,
        PRIMARY KEY (hash_name, field)
    )
    """,
]

@runtime_checkable
class CacheBackend(Protocol):
    """Minimal key/value-store + hash-map interface consumed by FrameCache.

    All keys and field names are plain strings.  Values are raw bytes.
    Hash values (hset/hgetall) are strings — they store small metadata such
    as serialization format names.
    """

    def exists(self, key: str) -> bool:
        """Return ``True`` iff ``key`` exists (and has not expired)."""
        ...

    def get(self, key: str) -> bytes | None:
        """Return the raw bytes for ``key``, or ``None`` if absent/expired."""
        ...

    def set(self, key: str, value: bytes, ttl: timedelta | None = None) -> None:
        """Store ``value`` under ``key``, optionally expiring after ``ttl``."""
        ...

    def delete(self, *keys: str) -> None:
        """Remove one or more keys (silently ignores missing ones)."""
        ...

    def scan(self, pattern: str) -> Iterator[str]:
        """Yield keys whose names match the glob ``pattern``.

        Glob metacharacters: ``*`` (any sequence), ``?`` (any single char).
        Characters escaped with a leading ``\\`` (e.g. ``\\*``) are treated
        as literals.
        """
        ...

    def hgetall(self, name: str) -> dict[str, str]:
        """Return all ``{field: value}`` pairs stored in the hash ``name``."""
        ...

    def hset(self, name: str, field: str, value: str) -> None:
        """Set ``field`` to ``value`` in the hash ``name``."""
        ...

    def hdel(self, name: str, *fields: str) -> None:
        """Remove one or more ``fields`` from the hash ``name``."""
        ...

    def close(self) -> None:
        """Release any resources held by the backend."""
        ...


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

class RedisBackend:
    """CacheBackend backed by a Redis server.

    Wraps a ``redis.Redis`` (or compatible) client and translates the
    :class:`CacheBackend` interface to native Redis commands.

    Args:
        redis_client: A live ``redis.Redis`` instance (or subclass, e.g.
            ``fakeredis.FakeRedis``).
    """

    def __init__(self, redis_client) -> None:
        try:
            import redis as _redis
            if not isinstance(redis_client, _redis.client.Redis):
                raise TypeError(
                    f"Expected a redis.Redis instance, got {type(redis_client).__name__}"
                )
        except ImportError:
            pass  # allow duck-typed fakes in environments without redis
        self._r = redis_client

    # -- key/value -----------------------------------------------------------

    def exists(self, key: str) -> bool:
        return self._r.exists(key) == 1

    def get(self, key: str) -> bytes | None:
        return self._r.get(key)

    def set(self, key: str, value: bytes, ttl: timedelta | None = None) -> None:
        self._r.set(key, value, ex=ttl)

    def delete(self, *keys: str) -> None:
        if keys:
            self._r.delete(*keys)

    def scan(self, pattern: str) -> Iterator[str]:
        for k in self._r.scan_iter(match=pattern):
            yield k.decode() if isinstance(k, bytes) else k

    # -- hash operations -----------------------------------------------------

    def hgetall(self, name: str) -> dict[str, str]:
        raw = self._r.hgetall(name) or {}
        return {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in raw.items()
        }

    def hset(self, name: str, field: str, value: str) -> None:
        self._r.hset(name, field, value)

    def hdel(self, name: str, *fields: str) -> None:
        if fields:
            self._r.hdel(name, *fields)

    def close(self) -> None:
        try:
            self._r.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

class SQLiteBackend:
    """CacheBackend backed by a local SQLite database.

    Data is stored in two tables:

    ``cache_entries``
        Holds the serialized cache blobs, one row per cache_instance_id.

        ==================  ============================================================
        Column              Description
        ==================  ============================================================
        ``cache_id``        TEXT PRIMARY KEY — the full cache_instance_id string.
        ``value``           BLOB — the serialized payload.
        ``created_at``      TEXT — ISO-8601 timestamp of when the entry was written.
        ``expires_at``      TEXT or NULL — ISO-8601 expiry timestamp; NULL = no expiry.
        ==================  ============================================================

    ``cache_hashes``
        Generic hash-map store (mirrors Redis hash semantics) used by
        FrameCache to persist the serialization-format metadata.

        ==================  ============================================================
        Column              Description
        ==================  ============================================================
        ``hash_name``       TEXT — name of the hash (e.g. ``FrameCache::__formats__``).
        ``field``           TEXT — field key within the hash.
        ``value``           TEXT — field value.
        ==================  ============================================================

    TTL is enforced lazily: expired entries are skipped on reads and pruned
    during ``scan`` calls and on ``close``.  WAL journal mode is enabled for
    safe concurrent reads.

    Args:
        db_path: Path to the SQLite file or ``":memory:"`` for an in-memory
            database (useful for testing).
    """

    _SCHEMA = _SQL_BACKEND_SCHEMA

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection: sqlite3.Connection | None = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        for stmt in self._SCHEMA:
            self._connection.execute(stmt)
        self._connection.commit()

    def _purge_expired(self) -> None:
        self._connection.execute(
            "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
            (_now_iso(),),
        )
        self._connection.commit()

    # -- CacheBackend interface ----------------------------------------------

    def exists(self, key: str) -> bool:
        now = _now_iso()
        row = self._connection.execute(
            "SELECT expires_at FROM cache_entries WHERE cache_id = ?", (key,)
        ).fetchone()
        if row is None:
            return False
        expires_at = row[0]
        if expires_at is not None and expires_at < now:
            self._connection.execute(
                "DELETE FROM cache_entries WHERE cache_id = ?", (key,)
            )
            self._connection.commit()
            return False
        return True

    def get(self, key: str) -> bytes | None:
        if not self.exists(key):
            return None
        row = self._connection.execute(
            "SELECT value FROM cache_entries WHERE cache_id = ?", (key,)
        ).fetchone()
        return bytes(row[0]) if row else None

    def set(self, key: str, value: bytes, ttl: timedelta | None = None) -> None:
        now = _now_iso()
        expires_at = _expiry_iso(ttl) if ttl is not None else None
        self._connection.execute(
            """
            INSERT INTO cache_entries (cache_id, value, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_id) DO UPDATE SET
                value      = excluded.value,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at
            """,
            (key, value, now, expires_at),
        )
        self._connection.commit()

    def delete(self, *keys: str) -> None:
        if not keys:
            return
        placeholders = ",".join("?" * len(keys))
        self._connection.execute(
            f"DELETE FROM cache_entries WHERE cache_id IN ({placeholders})", keys
        )
        self._connection.commit()

    def scan(self, pattern: str) -> Iterator[str]:
        self._purge_expired()
        like_pattern = _glob_to_sql_like(pattern)
        now = _now_iso()
        rows = self._connection.execute(
            """
            SELECT cache_id FROM cache_entries
            WHERE  cache_id LIKE ? ESCAPE '\\'
              AND  (expires_at IS NULL OR expires_at >= ?)
            ORDER BY cache_id
            """,
            (like_pattern, now),
        ).fetchall()
        for row in rows:
            yield row[0]

    def hgetall(self, name: str) -> dict[str, str]:
        rows = self._connection.execute(
            "SELECT field, value FROM cache_hashes WHERE hash_name = ?", (name,)
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def hset(self, name: str, field: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO cache_hashes (hash_name, field, value) VALUES (?, ?, ?)
            ON CONFLICT(hash_name, field) DO UPDATE SET value = excluded.value
            """,
            (name, field, value),
        )
        self._connection.commit()

    def hdel(self, name: str, *fields: str) -> None:
        if not fields:
            return
        placeholders = ",".join("?" * len(fields))
        self._connection.execute(
            f"DELETE FROM cache_hashes WHERE hash_name = ? AND field IN ({placeholders})",
            (name, *fields),
        )
        self._connection.commit()

    def close(self) -> None:
        if self._connection is not None:
            self._purge_expired()
            self._connection.close()
            self._connection = None

    # -- SQLite-only extras --------------------------------------------------

    def metadata_df(self):
        """Return a ``polars.DataFrame`` with all live cache entries.

        Columns: ``cache_id``, ``created_at``, ``expires_at``.
        Expired rows are excluded.  Useful for inspection and debugging.
        """
        import polars as pl

        self._purge_expired()
        now = _now_iso()
        rows = self._connection.execute(
            """
            SELECT cache_id, created_at, expires_at
            FROM   cache_entries
            WHERE  expires_at IS NULL OR expires_at >= ?
            ORDER  BY created_at
            """,
            (now,),
        ).fetchall()
        return pl.DataFrame(
            rows,
            schema=["cache_id", "created_at", "expires_at"],
            orient="row",
        )

    def __enter__(self) -> "SQLiteBackend":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# DuckDB backend
# ---------------------------------------------------------------------------

class DuckDBBackend:
    """CacheBackend backed by a local DuckDB database.

    Uses the same table layout and TTL semantics as :class:`SQLiteBackend`.
    DuckDB is well suited to analytical workloads and single-file persistence.

    Args:
        db_path: Path to the DuckDB file or ``":memory:"`` for an in-memory
            database (useful for testing).
    """

    _SCHEMA = _SQL_BACKEND_SCHEMA

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        import duckdb

        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(self._db_path)
        for stmt in self._SCHEMA:
            self._connection.execute(stmt)

    def _purge_expired(self) -> None:
        self._connection.execute(
            "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
            [_now_iso()],
        )

    def exists(self, key: str) -> bool:
        now = _now_iso()
        row = self._connection.execute(
            "SELECT expires_at FROM cache_entries WHERE cache_id = ?", [key]
        ).fetchone()
        if row is None:
            return False
        expires_at = row[0]
        if expires_at is not None and expires_at < now:
            self._connection.execute(
                "DELETE FROM cache_entries WHERE cache_id = ?", [key]
            )
            return False
        return True

    def get(self, key: str) -> bytes | None:
        if not self.exists(key):
            return None
        row = self._connection.execute(
            "SELECT value FROM cache_entries WHERE cache_id = ?", [key]
        ).fetchone()
        return bytes(row[0]) if row else None

    def set(self, key: str, value: bytes, ttl: timedelta | None = None) -> None:
        now = _now_iso()
        expires_at = _expiry_iso(ttl) if ttl is not None else None
        self._connection.execute(
            """
            INSERT INTO cache_entries (cache_id, value, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_id) DO UPDATE SET
                value      = excluded.value,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at
            """,
            [key, value, now, expires_at],
        )

    def delete(self, *keys: str) -> None:
        if not keys:
            return
        placeholders = ",".join("?" * len(keys))
        self._connection.execute(
            f"DELETE FROM cache_entries WHERE cache_id IN ({placeholders})", list(keys)
        )

    def scan(self, pattern: str) -> Iterator[str]:
        self._purge_expired()
        like_pattern = _glob_to_sql_like(pattern)
        now = _now_iso()
        rows = self._connection.execute(
            """
            SELECT cache_id FROM cache_entries
            WHERE  cache_id LIKE ? ESCAPE '\\'
              AND  (expires_at IS NULL OR expires_at >= ?)
            ORDER BY cache_id
            """,
            [like_pattern, now],
        ).fetchall()
        for row in rows:
            yield row[0]

    def hgetall(self, name: str) -> dict[str, str]:
        rows = self._connection.execute(
            "SELECT field, value FROM cache_hashes WHERE hash_name = ?", [name]
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def hset(self, name: str, field: str, value: str) -> None:
        self._connection.execute(
            """
            INSERT INTO cache_hashes (hash_name, field, value) VALUES (?, ?, ?)
            ON CONFLICT(hash_name, field) DO UPDATE SET value = excluded.value
            """,
            [name, field, value],
        )

    def hdel(self, name: str, *fields: str) -> None:
        if not fields:
            return
        placeholders = ",".join("?" * len(fields))
        self._connection.execute(
            f"DELETE FROM cache_hashes WHERE hash_name = ? AND field IN ({placeholders})",
            [name, *fields],
        )

    def close(self) -> None:
        if self._connection is not None:
            self._purge_expired()
            self._connection.close()
            self._connection = None

    def metadata_df(self):
        """Return a ``polars.DataFrame`` with all live cache entries."""
        import polars as pl

        self._purge_expired()
        now = _now_iso()
        rows = self._connection.execute(
            """
            SELECT cache_id, created_at, expires_at
            FROM   cache_entries
            WHERE  expires_at IS NULL OR expires_at >= ?
            ORDER  BY created_at
            """,
            [now],
        ).fetchall()
        return pl.DataFrame(
            rows,
            schema=["cache_id", "created_at", "expires_at"],
            orient="row",
        )

    def __enter__(self) -> "DuckDBBackend":
        return self

    def __exit__(self, *_) -> None:
        self.close()
