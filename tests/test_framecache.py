import time

import numpy as np
import polars as pl
import pytest

from framecache import FrameCache, CacheConfig, BackendFactory
from framecache.backends import RedisBackend, SQLiteBackend, DuckDBBackend


def make_counter_fn(values=None):
    """Build a function that records how many times it ran.

    The returned DataFrame embeds the call index so successive calls produce
    distinct values (useful for asserting historical retrieval).
    """
    state = {"calls": 0}

    def fn(seed: int = 0):
        state["calls"] += 1
        return pl.DataFrame({"seed": [seed], "call": [state["calls"]]})

    fn.state = state
    return fn


# --------------------------------------------------------------------------- #
# Default memoization (1, False)
# --------------------------------------------------------------------------- #
def test_default_memoizes(fc):
    fn = make_counter_fn()
    cached = fc.cache(method="pyarrow", func=fn)

    first = cached(5)
    second = cached(5)

    assert fn.state["calls"] == 1, "second identical call must hit the cache"
    assert first.equals(second)


def test_default_distinguishes_arguments(fc):
    fn = make_counter_fn()
    cached = fc.cache(method="pyarrow", func=fn)

    cached(1)
    cached(2)

    assert fn.state["calls"] == 2
    ids = fc.list_cache_instance_ids(func=fn)
    assert len(ids) == 2


# --------------------------------------------------------------------------- #
# Caching disabled (0, ...)
# --------------------------------------------------------------------------- #
def test_disabled_caching_never_stores(fc):
    fn = make_counter_fn()
    cached = fc.cache(method="pyarrow", func=fn, calls=(0, False))

    cached(1)
    cached(1)

    assert fn.state["calls"] == 2, "disabled cache must always recompute"
    assert fc.list_cache_instance_ids(func=fn) == []


# --------------------------------------------------------------------------- #
# History with counter tokens (n>1, False)
# --------------------------------------------------------------------------- #
def test_counter_history_keeps_last_n(fc):
    fn = make_counter_fn()
    cached = fc.cache(method="pyarrow", func=fn, calls=(3, False))

    for _ in range(5):
        cached(7)

    assert fn.state["calls"] == 5, "history mode records every call"

    versions = fc.cached_versions(fn, "pyarrow", 7)
    assert len(versions) == 3, "only the last 3 calls are retained"
    assert all(FrameCache.VERSION_SEPARATOR + "c" in v for v in versions)

    # oldest retained == call #3, newest == call #5
    assert fc.deserialize(versions[0])["call"].item() == 3
    assert fc.deserialize(versions[-1])["call"].item() == 5

    latest_id = fc.latest_cache_instance_id(fn, "pyarrow", 7)
    assert latest_id == versions[-1]
    assert fc.get_latest(fn, "pyarrow", 7)["call"].item() == 5


# --------------------------------------------------------------------------- #
# History with timestamp tokens (n, True)
# --------------------------------------------------------------------------- #
def test_timestamp_history(fc):
    fn = make_counter_fn()
    cached = fc.cache(method="pyarrow", func=fn, calls=(2, True))

    for _ in range(3):
        cached(9)
        time.sleep(0.002)

    versions = fc.cached_versions(fn, "pyarrow", 9)
    assert len(versions) == 2
    assert all(FrameCache.VERSION_SEPARATOR + "t" in v for v in versions)
    # versions are chronologically sorted; newest is the 3rd call
    assert fc.deserialize(versions[-1])["call"].item() == 3
    assert fc.deserialize(versions[0])["call"].item() == 2


# --------------------------------------------------------------------------- #
# Listing & regex discovery
# --------------------------------------------------------------------------- #
def test_list_with_regex(fc):
    fn = make_counter_fn()
    cached = fc.cache(method="pyarrow", func=fn)
    cached(11)
    cached(22)

    matched = fc.list_cache_instance_ids(func=fn, regex=r"\|11$")
    assert len(matched) == 1
    assert matched[0].endswith("|11")

    all_for_fn = fc.list_cache_instance_ids(func=fn)
    assert len(all_for_fn) == 2


def test_hashed_keys_listing_by_func_name(fc_hashed):
    fn = make_counter_fn()
    cached = fc_hashed.cache(method="pyarrow", func=fn, calls=(2, True))
    cached(1)
    time.sleep(0.002)
    cached(1)

    # function name remains visible in hashed keys, so discovery still works
    versions = fc_hashed.cached_versions(fn, "pyarrow", 1)
    assert len(versions) == 2
    assert fc_hashed.get_latest(fn, "pyarrow", 1)["call"].item() == 2


# --------------------------------------------------------------------------- #
# Serialization round-trips
# --------------------------------------------------------------------------- #
def test_polars_dataframe_roundtrip(fc):
    expected = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    def fn():
        return expected

    cached = fc.cache(method="pyarrow", func=fn)
    assert cached().equals(expected)


def test_polars_series_roundtrip_pickle(fc):
    expected = pl.Series("x", [1, 2, 3])

    def fn():
        return expected

    cached = fc.cache(method="pickle", func=fn)
    assert cached().to_list() == [1, 2, 3]


def test_polars_series_roundtrip_under_pyarrow(shared_client):
    """Series are supported under the pyarrow method (via pickle fallback)."""
    expected = pl.Series("x", [4, 5, 6])

    def fn():
        return expected

    FrameCache(shared_client).cache(method="pyarrow", func=fn)()

    reader = FrameCache(shared_client)
    assert reader.get_latest(fn, "pyarrow").to_list() == [4, 5, 6]


def test_numpy_uses_pickle_fallback(fc):
    arr = np.arange(6).reshape(2, 3)

    def fn():
        return arr

    cached = fc.cache(method="pyarrow", func=fn)
    out = cached()
    assert np.array_equal(out, arr)

    cid = fc.latest_cache_instance_id(fn, "pyarrow")
    fc.refresh()
    assert fc.cache_formats[cid] == "pickle"


def test_numpy_roundtrip_across_instances(shared_client):
    """A fresh instance must deserialize numpy-under-pyarrow correctly.

    Regression test: the format is persisted, not inferred from the key's
    method segment (which would wrongly say 'pyarrow' for pickled numpy bytes).
    """
    arr = np.arange(6).reshape(2, 3)
    writer = FrameCache(shared_client)

    def fn():
        return arr

    writer.cache(method="pyarrow", func=fn)()

    reader = FrameCache(shared_client)
    out = reader.get_latest(fn, "pyarrow")
    assert np.array_equal(out, arr)


def test_dataframe_roundtrip_across_instances(shared_client):
    expected = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    writer = FrameCache(shared_client)

    def fn():
        return expected

    writer.cache(method="pyarrow", func=fn)()

    reader = FrameCache(shared_client)
    assert reader.get_latest(fn, "pyarrow").equals(expected)


def test_json_roundtrip(fc):
    def fn():
        return {"a": [1, 2, 3], "b": "hello"}

    cached = fc.json_cache(fn)
    out = cached()
    assert out == {"a": [1, 2, 3], "b": "hello"}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Return-type validation
# --------------------------------------------------------------------------- #
def test_unsupported_return_type_raises(fc):
    @fc.cache(method="pyarrow")
    def fn():
        return "not a frame"

    with pytest.raises(TypeError):
        fn()


def test_unsupported_return_type_raises_in_history_mode(fc):
    @fc.cache(method="pyarrow", calls=(3, True))
    def fn():
        return 123

    with pytest.raises(TypeError):
        fn()


def test_json_rejects_frame_return(fc):
    @fc.json_cache
    def fn():
        return pl.DataFrame({"a": [1]})

    with pytest.raises(TypeError):
        fn()


def test_json_accepts_json_types(fc):
    @fc.json_cache
    def fn():
        return [1, 2, {"k": "v"}]

    assert fn() == [1, 2, {"k": "v"}]


def test_validate_return_can_be_disabled(fc):
    @fc.cache(method="pickle", validate_return=False)
    def fn():
        return {"arbitrary": "object", "n": 7}

    assert fn() == {"arbitrary": "object", "n": 7}


def test_none_return_is_allowed_and_not_cached(fc):
    calls = {"n": 0}

    @fc.cache(method="pyarrow")
    def fn():
        calls["n"] += 1
        return None

    assert fn() is None
    assert fn() is None
    assert calls["n"] == 2, "None is not cached, so it recomputes each time"


def test_supported_return_types_helper():
    assert FrameCache.supported_return_types("json") == FrameCache.SUPPORTED_JSON_TYPES
    assert FrameCache.supported_return_types("pyarrow") == FrameCache.SUPPORTED_FRAME_TYPES
    assert FrameCache.supported_return_types("pickle") == FrameCache.SUPPORTED_FRAME_TYPES


@pytest.mark.parametrize("bad", [(-1, False), (101, False), "nope", (1,), (1, 2, 3)])
def test_invalid_calls_raise(fc, bad):
    fn = make_counter_fn()
    with pytest.raises((ValueError, TypeError)):
        fc.cache(method="pyarrow", func=fn, calls=bad)


# --------------------------------------------------------------------------- #
# Decorator usage
# --------------------------------------------------------------------------- #
def test_parametrized_decorator(fc):
    calls = {"n": 0}

    @fc.cache(method="pyarrow")
    def fn(seed):
        calls["n"] += 1
        return pl.DataFrame({"seed": [seed]})

    first = fn(3)
    second = fn(3)

    assert calls["n"] == 1, "decorated function must be memoized"
    assert first.equals(second)
    assert fn.__name__ == "fn", "functools.wraps metadata is preserved"


def test_bare_decorator(fc):
    calls = {"n": 0}

    @fc.cache
    def fn(seed):
        calls["n"] += 1
        return pl.DataFrame({"seed": [seed]})

    fn(1)
    fn(1)

    assert calls["n"] == 1
    assert fn(1).equals(pl.DataFrame({"seed": [1]}))


def test_decorator_with_calls_history(fc):
    calls = {"n": 0}

    @fc.cache(method="pyarrow", calls=(3, True))
    def fn(seed):
        calls["n"] += 1
        return pl.DataFrame({"seed": [seed], "call": [calls["n"]]})

    for _ in range(4):
        fn(2)
        time.sleep(0.002)

    assert calls["n"] == 4, "history mode records every call"
    versions = fc.cached_versions(fn, "pyarrow", 2)
    assert len(versions) == 3
    assert fc.get_latest(fn, "pyarrow", 2)["call"].item() == 4


def test_pyarrow_cache_decorator_forms(fc):
    @fc.pyarrow_cache
    def bare(seed):
        return pl.DataFrame({"seed": [seed]})

    @fc.pyarrow_cache(calls=(1, False))
    def parametrized(seed):
        return pl.DataFrame({"seed": [seed]})

    assert bare(5).equals(pl.DataFrame({"seed": [5]}))
    assert parametrized(6).equals(pl.DataFrame({"seed": [6]}))


def test_json_cache_decorator_forms(fc):
    @fc.json_cache
    def bare():
        return {"a": 1}

    @fc.json_cache(calls=(2, True))
    def parametrized():
        return {"b": 2}

    assert bare() == {"a": 1}
    assert parametrized() == {"b": 2}


# --------------------------------------------------------------------------- #
# SQLiteBackend specifics
# --------------------------------------------------------------------------- #

def test_sqlite_ttl_expiry():
    """Entries with an expired TTL must not be retrievable."""
    backend = SQLiteBackend(":memory:")
    backend.set("k1", b"hello", ttl=__import__("datetime").timedelta(milliseconds=50))
    assert backend.exists("k1")
    time.sleep(0.1)
    assert not backend.exists("k1")
    assert backend.get("k1") is None
    backend.close()


def test_sqlite_scan_glob():
    backend = SQLiteBackend(":memory:")
    backend.set("ns-foo-bar", b"1")
    backend.set("ns-foo-baz", b"2")
    backend.set("other-key", b"3")
    results = sorted(backend.scan("ns-foo-*"))
    assert results == ["ns-foo-bar", "ns-foo-baz"]
    backend.close()


def test_sqlite_hash_operations():
    backend = SQLiteBackend(":memory:")
    backend.hset("myhash", "field1", "val1")
    backend.hset("myhash", "field2", "val2")
    assert backend.hgetall("myhash") == {"field1": "val1", "field2": "val2"}
    backend.hdel("myhash", "field1")
    assert backend.hgetall("myhash") == {"field2": "val2"}
    backend.close()


def test_sqlite_metadata_df():
    backend = SQLiteBackend(":memory:")
    fc = FrameCache(backend)

    @fc.cache(method="pyarrow")
    def fn(x):
        return pl.DataFrame({"x": [x]})

    fn(1)
    fn(2)
    df = backend.metadata_df()
    assert len(df) == 2
    assert "cache_id" in df.columns
    backend.close()


# --------------------------------------------------------------------------- #
# DuckDBBackend specifics
# --------------------------------------------------------------------------- #

def test_duckdb_ttl_expiry():
    backend = DuckDBBackend(":memory:")
    backend.set("k1", b"hello", ttl=__import__("datetime").timedelta(milliseconds=50))
    assert backend.exists("k1")
    time.sleep(0.1)
    assert not backend.exists("k1")
    assert backend.get("k1") is None
    backend.close()


def test_duckdb_scan_glob():
    backend = DuckDBBackend(":memory:")
    backend.set("ns-foo-bar", b"1")
    backend.set("ns-foo-baz", b"2")
    backend.set("other-key", b"3")
    results = sorted(backend.scan("ns-foo-*"))
    assert results == ["ns-foo-bar", "ns-foo-baz"]
    backend.close()


def test_duckdb_hash_operations():
    backend = DuckDBBackend(":memory:")
    backend.hset("myhash", "field1", "val1")
    backend.hset("myhash", "field2", "val2")
    assert backend.hgetall("myhash") == {"field1": "val1", "field2": "val2"}
    backend.hdel("myhash", "field1")
    assert backend.hgetall("myhash") == {"field2": "val2"}
    backend.close()


def test_duckdb_metadata_df():
    backend = DuckDBBackend(":memory:")
    fc = FrameCache(backend)

    @fc.cache(method="pyarrow")
    def fn(x):
        return pl.DataFrame({"x": [x]})

    fn(1)
    fn(2)
    df = backend.metadata_df()
    assert len(df) == 2
    assert "cache_id" in df.columns
    backend.close()


# --------------------------------------------------------------------------- #
# BackendFactory
# --------------------------------------------------------------------------- #

def test_backend_factory_supported_backends():
    assert BackendFactory.supported_backends() == frozenset({"redis", "sqlite", "duckdb"})


def test_backend_factory_create_sqlite():
    cfg = CacheConfig(backend_type="sqlite", db_path=":memory:")
    backend = BackendFactory.create(cfg)
    assert isinstance(backend, SQLiteBackend)
    backend.close()


def test_backend_factory_create_duckdb():
    cfg = CacheConfig(backend_type="duckdb", db_path=":memory:")
    backend = BackendFactory.create(cfg)
    assert isinstance(backend, DuckDBBackend)
    backend.close()


def test_backend_factory_create_framecache(tmp_path):
    cfg = CacheConfig(backend_type="duckdb", db_path=str(tmp_path / "fc.duckdb"))
    fc = BackendFactory.create_framecache(cfg)

    @fc.cache(method="pyarrow")
    def fn():
        return pl.DataFrame({"v": [1]})

    assert fn()["v"].item() == 1


def test_framecache_rejects_raw_redis_client(redis_client):
    with pytest.raises(TypeError, match="CacheBackend"):
        FrameCache(redis_client)


# --------------------------------------------------------------------------- #
# CacheConfig and YAML round-trip
# --------------------------------------------------------------------------- #

def test_cache_config_from_dict_redis():
    cfg = CacheConfig.from_dict({"backend_type": "redis", "framecache_key": "Test", "port": 6380})
    assert cfg.backend_type == "redis"
    assert cfg.framecache_key == "Test"
    assert cfg.port == 6380


def test_cache_config_from_dict_sqlite():
    cfg = CacheConfig.from_dict({"backend_type": "sqlite", "db_path": "./test.db", "default_ttl_hours": 48.0})
    assert cfg.backend_type == "sqlite"
    assert cfg.default_ttl_hours == 48.0


def test_cache_config_from_dict_duckdb():
    cfg = CacheConfig.from_dict({"backend_type": "duckdb", "db_path": "./test.duckdb"})
    assert cfg.backend_type == "duckdb"
    assert cfg.db_path.endswith("test.duckdb")


def test_cache_config_invalid_backend():
    with pytest.raises(ValueError, match="backend_type"):
        CacheConfig.from_dict({"backend_type": "memcached"})


def test_cache_config_yaml_round_trip(tmp_path):
    cfg = CacheConfig(
        backend_type="sqlite",
        framecache_key="YAMLTest",
        use_hash_keys=True,
        default_ttl_hours=12.0,
        db_path=str(tmp_path / "test.db"),
    )
    yaml_path = tmp_path / "cache.yaml"
    cfg.to_yaml(yaml_path)
    reloaded = CacheConfig.from_yaml(yaml_path)
    assert reloaded.backend_type == "sqlite"
    assert reloaded.framecache_key == "YAMLTest"
    assert reloaded.use_hash_keys is True
    assert reloaded.default_ttl_hours == 12.0


def test_framecache_from_config_sqlite(tmp_path):
    cfg = CacheConfig(
        backend_type="sqlite",
        framecache_key="CFGTest",
        db_path=str(tmp_path / "fc.db"),
        default_ttl_hours=24.0,
    )
    fc_cfg = FrameCache.from_config(cfg)
    assert fc_cfg.framecache_key == "CFGTest"

    @fc_cfg.cache(method="pyarrow")
    def fn():
        return pl.DataFrame({"v": [42]})

    assert fn()["v"].item() == 42
    assert fn()["v"].item() == 42  # served from cache


def test_framecache_from_yaml(tmp_path):
    cfg = CacheConfig(
        backend_type="sqlite",
        db_path=str(tmp_path / "fc2.db"),
    )
    cfg.to_yaml(tmp_path / "cfg.yaml")
    fc_yaml = FrameCache.from_yaml(tmp_path / "cfg.yaml")

    @fc_yaml.pyarrow_cache
    def fn():
        return pl.DataFrame({"n": [1, 2, 3]})

    result = fn()
    assert result["n"].to_list() == [1, 2, 3]
