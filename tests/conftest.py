import sys
from pathlib import Path

import fakeredis
import pytest

# src/ contains the framecache package directory
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from framecache import FrameCache, SQLiteBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Raw storage client fixtures  (one Redis, one SQLite per test)
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


# ---------------------------------------------------------------------------
# FrameCache fixtures parametrized over both backends
# ---------------------------------------------------------------------------

@pytest.fixture(params=["redis", "sqlite"])
def fc(request, redis_client, sqlite_backend):
    if request.param == "redis":
        return FrameCache(redis_client)
    return FrameCache(sqlite_backend)


@pytest.fixture(params=["redis", "sqlite"])
def fc_hashed(request, redis_client, sqlite_backend):
    if request.param == "redis":
        return FrameCache(redis_client, use_hash_keys=True)
    return FrameCache(sqlite_backend, use_hash_keys=True)


# ---------------------------------------------------------------------------
# Shared-storage fixture for cross-instance round-trip tests
# ---------------------------------------------------------------------------

@pytest.fixture(params=["redis", "sqlite"])
def shared_client(request):
    """Storage that two separate FrameCache instances can connect to."""
    if request.param == "redis":
        client = fakeredis.FakeRedis()
        client.flushall()
        yield client
        client.close()
    else:
        backend = SQLiteBackend(":memory:")
        yield backend
        backend.close()
