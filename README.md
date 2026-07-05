# framecache

Memoization for functions that return a `polars.DataFrame`[^1], `polars.Series`,
`numpy.ndarray`, `numpy.matrix`, or `numpy.recarray`, backed by **Redis**,
**SQLite**, or **DuckDB**. Results are serialized and stored keyed by the function name, its
arguments, and the serialization method, then reused on subsequent identical
calls. On top of plain memoization, framecache can retain a rolling history of
the last *N* calls and let you retrieve historical values.

Inspired by [randas_cache](https://github.com/emmc15/Randas_Cache).


## Installation

This project uses [uv](https://github.com/astral-sh/uv) as its package manager, offering faster dependency resolution and installation compared to traditional tools like pip. Below are the installation instructions for common operating systems:

## macOS
```bash
brew install uv
```

## Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Windows (PowerShell)
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

After installing `uv`, verify it’s available:
```bash
uv --version
```

Retrieve the repository:

```bash
git clone https://github.com/FulgentMcGuffin/FrameCache.git
```

You can install the `framecache` package by running one of the
following from the repository root:

```bash
# Recommended: install in editable mode with all dependencies (uv)
uv pip install -e .

# Or sync the full project environment (installs deps + package in one step)
uv sync

# With plain pip (inside any virtualenv)
pip install .

# Editable / development install with pip
pip install -e .
```

After installation, import it directly in any Python environment:

```python
from framecache import FrameCache, CacheConfig, BackendFactory
```

## Backends & Configuration

framecache ships three storage backends behind a common interface. Clients
should use `BackendFactory` (or `FrameCache.from_config` /
`FrameCache.from_yaml`) and never need to import concrete backend classes.

| `backend_type` | When to use                                         |
| -------------- | --------------------------------------------------- |
| `redis`        | Shared / distributed cache; TTL managed by Redis    |
| `sqlite`       | Local / file-based cache; TTL enforced via a column |
| `duckdb`       | Local / file-based cache; same layout as SQLite     |

### YAML configuration (recommended)

Create a YAML file and use `FrameCache.from_yaml()`:

```yaml
# cache_redis.yaml
backend_type: redis
framecache_key: MyCache    # key namespace (optional, default "FrameCache")
use_hash_keys: false       # SHA-256 argument hashing (optional)
default_ttl_hours: 1.0

host: localhost
port: 6379
db: 0
# password: null           # optional
```

```yaml
# cache_sqlite.yaml
backend_type: sqlite
framecache_key: MyCache
use_hash_keys: false
default_ttl_hours: 24.0    # null / omit for no expiry

db_path: ./cache/framecache.db    # ":memory:" for tests/in-memory
```

```yaml
# cache_duckdb.yaml
backend_type: duckdb
framecache_key: MyCache
use_hash_keys: false
default_ttl_hours: 24.0

db_path: ./cache/framecache.duckdb
```

```python
from framecache import FrameCache

fc = FrameCache.from_yaml("cache_sqlite.yaml")
```

### BackendFactory (recommended)

Use `BackendFactory` to obtain a backend without importing implementation
classes:

```python
from framecache import BackendFactory, CacheConfig, FrameCache

config = CacheConfig.from_yaml("cache.yaml")
backend = BackendFactory.create(config)
fc = FrameCache(backend)

# Or in one step:
fc = BackendFactory.create_framecache(config)
# equivalent to:
fc = FrameCache.from_config(config)
```

### Programmatic construction

```python
from framecache import FrameCache, CacheConfig

config = CacheConfig(backend_type="duckdb", db_path="./my.duckdb", default_ttl_hours=12.0)
fc = FrameCache.from_config(config)
```

`FrameCache` accepts a `CacheBackend` instance only (typically from
`BackendFactory.create()`). 

## Quick start

```python
import polars as pl
from framecache import FrameCache

fc = FrameCache.from_yaml("cache.yaml")

@fc.cache(method="pyarrow")
def load_sales(region: str) -> pl.DataFrame:
    # ... expensive query ...
    return pl.DataFrame({"region": [region], "total": [123]})

df = load_sales("EU")   # first call: computed and cached
df = load_sales("EU")   # subsequent calls: served from cache
```

### Usage as a decorator

`cache` works as a bare decorator, a parametrized decorator, or via the
`pyarrow_cache` / `json_cache` shortcuts:

```python
@fc.cache(method="pyarrow")
def load_sales(region: str) -> pl.DataFrame: ...

@fc.cache(method="pyarrow", calls=(5, True))   # keep last 5 calls, timestamped
def load_metrics(day: str) -> pl.DataFrame: ...

@fc.cache                                       # bare: defaults to pyarrow
def load_users() -> pl.DataFrame: ...

@fc.pyarrow_cache
def load_orders() -> pl.DataFrame: ...

@fc.json_cache(calls=(2, True))
def load_config() -> dict: ...
```

`functools.wraps` is applied so the function object retains its original
metadata and can be passed to the retrieval helpers below.

### Serialization methods

| `method`    | Best for                                | Notes                                                        |
| ----------- | --------------------------------------- | ------------------------------------------------------------ |
| `"pyarrow"` | `polars.DataFrame` (default)            | Arrow IPC stream. `Series` / numpy arrays fall back to pickle|
| `"pickle"`  | `polars.Series`, numpy arrays, anything | General purpose                                              |
| `"json"`    | JSON-serializable values (dicts/lists)  | Human-readable                                               |

## The `calls` argument

`cache()` (and `json_cache` / `pyarrow_cache`) take an optional
`calls: tuple[int, bool]` that defaults to `(1, False)`:

```python
fc.cache(method="pyarrow", func=fn, calls=(n_last_calls, add_timestamp))
```

- **`n_last_calls`** (`0`–`100`): how many of the most recent cached calls to
  retain per function/arguments combination.
  - `1` (default) → classic single-value memoization.
  - `0` → caching disabled (always recompute, store nothing).
  - `> 1` → record every invocation and keep a rolling window of the last *n*.
- **`add_timestamp`**: when `True` a millisecond-accuracy timestamp is embedded
  in the key, so each call becomes its own historical record. When `False`
  history entries use a sortable zero-padded counter instead.

### Return-type validation

By default, the value returned by a cached function is checked against the
allowed types for its `method`, raising a `TypeError` on mismatch:

- `"pyarrow"` / `"pickle"` → `pl.DataFrame`, `pl.Series`, `np.ndarray`,
  `np.matrix`, `np.recarray` (or `None`, which is not cached).
- `"json"` → `dict`, `list`, `tuple`, `str`, `int`, `float`, `bool` (or `None`).

```python
@fc.cache(method="pyarrow")
def bad() -> str:
    return "not a frame"   # raises TypeError at call time

@fc.cache(method="pickle", validate_return=False)
def anything():
    return {"any": "picklable object"}   # bypass check
```

The accepted types are accessible via `FrameCache.SUPPORTED_FRAME_TYPES`,
`FrameCache.SUPPORTED_JSON_TYPES`, and `FrameCache.supported_return_types(method)`.

### How keys encode history

Historical calls live under versioned `cache_instance_id`s appended with `@@`:

- `…@@t20260623T234512.345` — timestamp token (`add_timestamp=True`)
- `…@@c000000000001` — zero-padded counter (`add_timestamp=False`)

Both are lexicographically sortable so the most-recent call is always the
maximum. **Default retrieval always returns the last cached call.**

## Retrieving cached values (including history)

```python
# Most recent cached call (default behaviour)
df = fc.get_latest(load_sales, "pyarrow", "EU")
cid = fc.latest_cache_instance_id(load_sales, "pyarrow", "EU")

# Every retained version of an exact func/args/method call, oldest first
versions = fc.cached_versions(load_sales, "pyarrow", "EU")
oldest_df = fc.deserialize(versions[0])

# Broad discovery — combine func, method, glob, and/or Python regex
for cid in fc.list_cache_instance_ids(func=load_sales, regex=r"region-EU"):
    historical_df = fc.deserialize(cid)
```

`list_cache_instance_ids` accepts any combination of:

- `glob_match` — a raw glob pattern forwarded to the backend.
- `func` / `method` — narrows the scan to a function (and optionally a method).
  The function name stays visible in the key even when `use_hash_keys=True`.
- `regex` — a Python regex applied after the scan, useful when not all
  arguments are known or several func/arg combinations are wanted.

## SQL file backend extras

SQLite and DuckDB backends expose `metadata_df()` on the backend instance
returned by `BackendFactory.create()` — useful for inspection and debugging:

```python
from framecache import BackendFactory, CacheConfig, FrameCache

config = CacheConfig(backend_type="sqlite", db_path="./my_cache.db")
backend = BackendFactory.create(config)
fc = FrameCache(backend)

# ... run some cached functions ...

df = backend.metadata_df()
# shape: (n_entries, 3) — columns: cache_id, created_at, expires_at
print(df)
```

## Notes on correctness

The actual serialization format is persisted in a companion hash/table
(`<framecache_key>::__formats__`) rather than inferred from the key. This is
required because numpy arrays stored under the `pyarrow` method transparently
fall back to pickle — a fresh process / new `FrameCache` instance can still
deserialize them correctly.

## Testing

Tests run without a live Redis server or on-disk database files; all three
backends use in-memory instances during the test suite. Every functional test
is parametrized and runs against **redis**, **sqlite**, and **duckdb**:

```bash
uv run pytest
```

[^1]: For `pandas.DataFrame` caching, convert first: `pl.DataFrame(df_pandas)`.
