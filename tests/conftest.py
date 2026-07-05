import sys
from pathlib import Path

import fakeredis
import pytest

# src/ contains the framecache package directory
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from framecache import FrameCache, CacheConfig, BackendFactory  # noqa: E402
from framecache.backends import RedisBackend, SQLiteBackend, DuckDBBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Raw storage client fixtures  (one Redis, one SQLite, one DuckDB per test)
# ---------------------------------------------------------------------------

@pytest.fixture
def redis_client():
    """Fresh in-memory fakeredis client."""
    client = fakeredis.FakeRedis()
    client.flushall()
    return client


@pytest.fixture
def sqlite_backend():
    """Fresh in-memory SQLite backend."""
    backend = SQLiteBackend(":memory:")
    yield backend
    backend.close()


@pytest.fixture
def duckdb_backend():
    """Fresh in-memory DuckDB backend."""
    backend = DuckDBBackend(":memory:")
    yield backend
    backend.close()


# ---------------------------------------------------------------------------
# FrameCache fixtures parametrized over all backends
# ---------------------------------------------------------------------------

@pytest.fixture(params=["redis", "sqlite", "duckdb"])
def fc(request, redis_client, sqlite_backend, duckdb_backend):
    if request.param == "redis":
        return FrameCache(RedisBackend(redis_client))
    if request.param == "sqlite":
        return FrameCache(sqlite_backend)
    return FrameCache(duckdb_backend)


@pytest.fixture(params=["redis", "sqlite", "duckdb"])
def fc_hashed(request, redis_client, sqlite_backend, duckdb_backend):
    if request.param == "redis":
        return FrameCache(RedisBackend(redis_client), use_hash_keys=True)
    if request.param == "sqlite":
        return FrameCache(sqlite_backend, use_hash_keys=True)
    return FrameCache(duckdb_backend, use_hash_keys=True)


# ---------------------------------------------------------------------------
# Shared-storage fixture for cross-instance round-trip tests
# ---------------------------------------------------------------------------

@pytest.fixture(params=["redis", "sqlite", "duckdb"])
def shared_client(request):
    """Storage that two separate FrameCache instances can connect to."""
    if request.param == "redis":
        client = fakeredis.FakeRedis()
        client.flushall()
        yield RedisBackend(client)
        client.close()
    elif request.param == "sqlite":
        backend = SQLiteBackend(":memory:")
        yield backend
        backend.close()
    else:
        backend = DuckDBBackend(":memory:")
        yield backend
        backend.close()
