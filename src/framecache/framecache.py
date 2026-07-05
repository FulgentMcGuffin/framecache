import os, sys, re
import polars as pl
import pyarrow as pa
import numpy as np
import json
import hashlib
import pickle
import functools
from datetime import timedelta, datetime, date
from io import BytesIO
from pathlib import Path
from typing import Literal, Callable, Tuple

from framecache.backends import CacheBackend  # noqa: E402

# Inspired by: https://github.com/emmc15/Randas_Cache


CacheFormat = Literal["pickle", "pyarrow", "json"]

# (number_of_last_calls_to_retain, add_millisecond_timestamp_to_key)
CallsSpec = Tuple[int, bool]

class FrameCache:

    DEFAULT_KEY_TTL = timedelta(hours=1)

    # Separates the base cache_instance_id from a per-call version token.
    VERSION_SEPARATOR = "@@"
    # Upper bound for the number of last calls that can be retained per key.
    MAX_RETAINED_CALLS = 100
    # Zero-padding width for counter based version tokens (keeps them
    # lexicographically sortable in chronological order).
    _COUNTER_WIDTH = 12

    # Return types a cached function may produce. The frame/array types apply to
    # the "pyarrow" and "pickle" methods; the "json" method expects plain
    # JSON-serializable values instead.
    SUPPORTED_FRAME_TYPES = (pl.DataFrame, pl.Series, np.ndarray, np.matrix, np.recarray)
    SUPPORTED_JSON_TYPES = (dict, list, tuple, str, int, float, bool)

    def __init__(
        self,
        backend: CacheBackend,
        framecache_key: str = None,
        use_hash_keys: bool = False,
        default_ttl: timedelta = None,
    ):
        """Construct a FrameCache.

        Args:
            backend:        A :class:`~framecache.backends.CacheBackend` instance.
                            Use :class:`~framecache.backend_factory.BackendFactory`
                            or :meth:`from_config` / :meth:`from_yaml` to obtain one.
            framecache_key: Namespace prefix for all cache keys.
                            Defaults to the class name ``"FrameCache"``.
            use_hash_keys:  When ``True``, SHA-256-hash the argument portion
                            of every cache_instance_id to keep key lengths
                            bounded.
            default_ttl:    Lifetime of cached entries.  Overrides
                            :attr:`DEFAULT_KEY_TTL`.  ``None`` uses the class
                            default (1 hour unless overridden in config).

        Prefer :meth:`from_config` / :meth:`from_yaml` for new code.
        """
        # redis.Redis satisfies @runtime_checkable CacheBackend by method name but
        # is not compatible (e.g. scan() signature differs). Reject it explicitly.
        try:
            import redis as _redis

            if isinstance(backend, _redis.client.Redis):
                raise TypeError(
                    "Expected a CacheBackend instance, got redis.Redis. "
                    "Use BackendFactory.create(config) or FrameCache.from_config(config)."
                )
        except ImportError:
            pass

        if not isinstance(backend, CacheBackend):
            raise TypeError(
                f"Expected a CacheBackend instance, got {type(backend).__name__}. "
                f"Use BackendFactory.create(config) or FrameCache.from_config(config)."
            )

        self.cache_container: CacheBackend = backend
        self.use_hash_keys = use_hash_keys
        self.framecache_key = framecache_key if framecache_key is not None else self.__class__.__name__

        # Instance-level TTL (falls back to class DEFAULT_KEY_TTL when None).
        self._default_ttl = default_ttl

        # Companion hash that records the *actual* serialization format used for
        # each cache_instance_id. This matters because numpy values stored under
        # the "pyarrow" method silently fall back to pickle, so the format cannot
        # always be inferred from the key alone.
        self._formats_key = f"{self.framecache_key}::__formats__"

        self.cache_formats = {}
        self.refresh()

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "CacheConfig") -> "FrameCache":
        """Build a :class:`FrameCache` from a :class:`~framecache.cache_config.CacheConfig`.

        Example::

            config = CacheConfig.from_yaml("cache.yaml")
            fc = FrameCache.from_config(config)
        """
        from framecache.backend_factory import BackendFactory  # noqa: E402

        backend = BackendFactory.create(config)
        return cls(
            backend,
            framecache_key=config.framecache_key,
            use_hash_keys=config.use_hash_keys,
            default_ttl=config.default_ttl,
        )

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "FrameCache":
        """Build a :class:`FrameCache` directly from a YAML config file.

        Example::

            fc = FrameCache.from_yaml("cache.yaml")
        """
        from framecache.cache_config import CacheConfig  # noqa: E402
        return cls.from_config(CacheConfig.from_yaml(path))

    def _effective_ttl(self, ttl: timedelta | None) -> timedelta | None:
        """Resolve a TTL value: explicit > instance default > class default."""
        if ttl is not None:
            return ttl
        if self._default_ttl is not None:
            return self._default_ttl
        return self.DEFAULT_KEY_TTL

    @staticmethod
    def deserialize_arrow_to_polars(pull_value):
        if pull_value is None or len(pull_value) == 0:
            return None
        reader = pa.ipc.open_stream(pull_value)
        arrow_table = reader.read_all()
        return pl.from_arrow(arrow_table)

    def refresh(self, user_suffix="*"):
        stored = dict(self.cache_container.hgetall(self._formats_key))

        formats: dict[str, str] = {}
        live_keys: set[str] = set()

        for k in self.cache_container.scan(
            f"{self.framecache_key}-{user_suffix}"
        ):
            live_keys.add(k)
            if k in stored:
                formats[k] = stored[k]
            else:
                parts = k.split("-")
                if len(parts) >= 3:
                    formats[k] = parts[1].lower().strip()

        stale = [cid for cid in stored if cid not in live_keys]
        if stale:
            self.cache_container.hdel(self._formats_key, *stale)

        self.cache_formats = formats

    def cache_instance_id(self, func: Callable, method: CacheFormat, *args, **kwargs):
        prefix_string = f"{self.framecache_key}-{method}-{func.__name__}"
        parameters_as_strings = []

        if args:
            parameters_as_strings.append("|".join(str(i) for i in args))

        if kwargs:
            for k, v in kwargs.items():
                parameters_as_strings.append(f"({k}-{v})")

        if self.use_hash_keys:
            if len(parameters_as_strings) > 0:                        
                parameters_as_strings = hashlib.sha256("".join(parameters_as_strings).encode()).hexdigest()
                return "-".join([prefix_string, parameters_as_strings])
            else:
                return "-".join([prefix_string, hashlib.sha256(prefix_string.encode()).hexdigest()])

        parameters_as_strings.insert(0, prefix_string)
        return "|".join(parameters_as_strings)

    @staticmethod
    def _normalize_calls(calls: CallsSpec) -> CallsSpec:
        """Validate/normalize the ``calls`` tuple into ``(n_last_calls, add_timestamp)``."""
        if calls is None:
            return (1, False)
        try:
            n_last_calls, add_timestamp = calls
        except (TypeError, ValueError):
            raise ValueError(
                f"`calls` must be a (int, bool) tuple, received {calls!r}"
            )
        n_last_calls = int(n_last_calls)
        if not (0 <= n_last_calls <= FrameCache.MAX_RETAINED_CALLS):
            raise ValueError(
                f"`calls` count must be between 0 and {FrameCache.MAX_RETAINED_CALLS}, "
                f"received {n_last_calls}"
            )
        return (n_last_calls, bool(add_timestamp))

    @classmethod
    def supported_return_types(cls, method: CacheFormat):
        """The return types accepted for a given serialization ``method``."""
        return cls.SUPPORTED_JSON_TYPES if method == "json" else cls.SUPPORTED_FRAME_TYPES

    def _check_return_type(self, value, method: CacheFormat, func: Callable = None):
        """Raise ``TypeError`` if ``value`` is not supported for ``method``.

        ``None`` is allowed: it is treated as a no-op (nothing gets cached).
        """
        if value is None:
            return
        allowed = self.supported_return_types(method)
        if not isinstance(value, allowed):
            name = getattr(func, "__name__", "decorated function")
            allowed_names = ", ".join(t.__name__ for t in allowed)
            raise TypeError(
                f"{name} returned {type(value).__name__!r}, which is not a supported "
                f"return type for method '{method}'. Expected one of: {allowed_names} "
                f"(or None). Pass validate_return=False to skip this check."
            )

    @staticmethod
    def _glob_escape(text: str) -> str:
        """Escape Redis glob meta characters so a literal string matches itself."""
        return re.sub(r"([\\*?\[\]])", r"\\\1", text)

    def _decoded_scan(self, match: str):
        yield from self.cache_container.scan(match)

    def _version_ids(self, base_id: str):
        """All versioned cache_instance_ids for ``base_id``, oldest first."""
        match = f"{self._glob_escape(base_id)}{self.VERSION_SEPARATOR}*"
        return sorted(self._decoded_scan(match))

    def _latest_version_id(self, base_id: str):
        versions = self._version_ids(base_id)
        return versions[-1] if versions else None

    def _new_version_id(self, base_id: str, add_timestamp: bool) -> str:
        """Build a fresh, chronologically-sortable versioned cache_instance_id."""
        sep = self.VERSION_SEPARATOR

        if add_timestamp:
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S.%f")[:-3]  # millisecond accuracy
            candidate = f"{base_id}{sep}t{stamp}"
            unique = candidate
            collision = 1
            while self.cache_container.exists(unique):
                unique = f"{candidate}.{collision}"
                collision += 1
            return unique

        last = 0
        for key in self._version_ids(base_id):
            token = key.rsplit(sep, 1)[-1]
            if token.startswith("c"):
                try:
                    last = max(last, int(token[1:]))
                except ValueError:
                    continue
        width = self._COUNTER_WIDTH
        return f"{base_id}{sep}c{(last + 1):0{width}d}"

    def _prune_versions(self, base_id: str, keep: int):
        """Keep only the ``keep`` most-recent versions of ``base_id``."""
        if keep <= 0:
            return
        versions = self._version_ids(base_id)
        excess = len(versions) - keep
        if excess <= 0:
            return
        for old_key in versions[:excess]:
            self.cache_container.delete(old_key)
            self.cache_container.hdel(self._formats_key, old_key)
            self.cache_formats.pop(old_key, None)

    def cached_versions(self, func: Callable, method: CacheFormat, *args, **kwargs):
        """List every cache_instance_id stored for an exact func/args/method, oldest first.

        Works for both hashed and non-hashed keys since the base id is computed
        exactly. Each entry can be passed straight to :meth:`deserialize`.
        """
        base = self.cache_instance_id(func, method, *args, **kwargs)
        ids = self._version_ids(base)
        if self.cache_container.exists(base):
            ids = [base] + ids
        return ids

    def latest_cache_instance_id(self, func: Callable, method: CacheFormat, *args, **kwargs):
        """The most recently cached call for the given func/args/method (or ``None``)."""
        base = self.cache_instance_id(func, method, *args, **kwargs)
        latest_version = self._latest_version_id(base)
        if latest_version is not None:
            return latest_version
        if self.cache_container.exists(base):
            return base
        return None

    def get_latest(self, func: Callable, method: CacheFormat, *args, **kwargs):
        """Deserialize the most recent cached call (default retrieval behaviour)."""
        cache_instance_id = self.latest_cache_instance_id(func, method, *args, **kwargs)
        if cache_instance_id is None:
            raise ValueError(
                f"No cached call found for {getattr(func, '__name__', func)} with the given arguments"
            )
        return self.deserialize(cache_instance_id)

    def list_cache_instance_ids(
        self,
        func: Callable = None,
        method: CacheFormat = None,
        regex: str = None,
        glob_match: str = None,
    ):
        """List cached cache_instance_ids, sorted chronologically (oldest first).

        Provide any combination of:
          * ``glob_match`` - a raw Redis glob pattern (highest precedence).
          * ``func`` / ``method`` - narrow the scan to a function (and optionally a
            serialization method); the function name is always visible in the key,
            even when ``use_hash_keys`` is enabled, so this still works for hashed keys.
          * ``regex`` - a Python regex applied to the scanned keys for fine-grained
            filtering when not all arguments are known or several func/arg
            combinations are wanted.

        With no arguments, every cache_instance_id under this instance is returned.
        """
        if glob_match is not None:
            scan_match = glob_match
        else:
            fkey = self._glob_escape(self.framecache_key)
            meth = self._glob_escape(method) if method is not None else "*"
            if func is not None:
                fname = self._glob_escape(func.__name__)
                scan_match = f"{fkey}-{meth}-{fname}*"
            elif method is not None:
                scan_match = f"{fkey}-{meth}-*"
            else:
                scan_match = f"{fkey}-*"

        keys = list(self._decoded_scan(scan_match))

        if regex is not None:
            pattern = re.compile(regex)
            keys = [k for k in keys if pattern.search(k)]

        return sorted(keys)

    def deserialize(self, cache_instance_id):
        pull_value = self.cache_container.get(cache_instance_id)

        self.refresh()

        if cache_instance_id not in self.cache_formats:
            return pickle.loads(pull_value)

        method = self.cache_formats[cache_instance_id]

        if method == "pickle":
            return pickle.loads(pull_value)

        if method == "pyarrow":
            return FrameCache.deserialize_arrow_to_polars(pull_value)

        if method == "json":
            return json.loads(pull_value)

        return pickle.loads(pull_value)

    def serialize(
        self,
        cache_instance_id,
        value: pl.DataFrame | pl.Series | np.ndarray | np.matrix | np.recarray,
        method: CacheFormat = "pyarrow",
        ttl: timedelta = None,
    ):
        if value is None:
            return

        method = method.lower().strip()

        if method == "pickle":
            hashed_value = pickle.dumps(value)
            self.cache_formats[cache_instance_id] = "pickle"
        elif method == "pyarrow":
            # Only polars DataFrames map cleanly onto the Arrow IPC stream; Series
            # and numpy arrays fall back to pickle (the real format is persisted).
            if isinstance(value, pl.DataFrame):
                arrow_table = value.to_arrow()

                sink = BytesIO()
                writer = pa.ipc.new_stream(sink, arrow_table.schema)
                writer.write_table(arrow_table)
                writer.close()

                hashed_value = sink.getvalue()
                self.cache_formats[cache_instance_id] = "pyarrow"
            else:
                hashed_value = pickle.dumps(value)
                self.cache_formats[cache_instance_id] = "pickle"
        elif method == "json":
            hashed_value = json.dumps(value)
            self.cache_formats[cache_instance_id] = "json"
        else:
            raise ValueError(f"Unsupported serialization method: {method}")

        self.cache_container.set(cache_instance_id, hashed_value, ttl=self._effective_ttl(ttl))
        # Persist the actual format so it can be recovered by any process/refresh,
        # even when it differs from the key's method segment (numpy -> pickle).
        self.cache_container.hset(
            self._formats_key, cache_instance_id, self.cache_formats[cache_instance_id]
        )

    def get(self, cache_instance_id: str):
        if self.cache_container.exists(cache_instance_id):
            return self.deserialize(cache_instance_id)

        raise ValueError(f"No cache instance id {cache_instance_id} found in object")

    def post(
        self,
        cache_instance_id: str,
        value: pl.DataFrame | pl.Series | np.ndarray | np.matrix | np.recarray,
        serialization: CacheFormat = "pyarrow",
        # add_prefix: bool = False,
    ):
        # if add_prefix:
        #     cache_instance_id = f"{self.framecache_key}-{serialization}-{cache_instance_id}"
        self.serialize(cache_instance_id, value, method=serialization)

    def cache(
        self,
        method: CacheFormat = "pyarrow",
        func: Callable = None,
        cache_instance_id: str = None,
        ttl: timedelta = None,
        calls: CallsSpec = (1, False),
        validate_return: bool = True,
    ):
        """Memoize ``func`` via the configured cache backend.

        ``calls`` is a ``(n_last_calls, add_timestamp)`` tuple:
          * ``n_last_calls`` (0-100): how many of the most recent cached calls to
            retain for each func/args combination.
              - ``1`` (default) behaves exactly like a classic memoize: the value
                is computed once and reused on subsequent identical calls.
              - ``0`` disables caching entirely (always recompute, store nothing).
              - ``>1`` records every invocation as its own historical entry and
                keeps a rolling window of the last ``n_last_calls`` of them.
          * ``add_timestamp``: when ``True`` (which also forces history mode), a
            millisecond-accuracy timestamp is embedded in the key so each call is
            recorded individually; when ``False`` history entries use a sortable
            counter token instead.

        Retrieval defaults to the most recent cached call; use
        :meth:`latest_cache_instance_id`, :meth:`cached_versions` or
        :meth:`list_cache_instance_ids` together with :meth:`deserialize` to reach
        historical values.

        Can be used directly (``fc.cache(method="pyarrow", func=fn)``) or as a
        decorator, both bare (``@fc.cache``) and parametrized
        (``@fc.cache(method="pyarrow", calls=(5, True))``).

        When ``validate_return`` is ``True`` (default) the value returned by the
        decorated function is checked against the supported types for ``method``
        (see :attr:`SUPPORTED_FRAME_TYPES` / :attr:`SUPPORTED_JSON_TYPES`) and a
        ``TypeError`` is raised on mismatch. Set it to ``False`` to cache
        arbitrary (picklable) objects.
        """
        # Support bare-decorator usage: ``@fc.cache`` passes the function as the
        # first positional argument (``method``).
        if callable(method) and func is None:
            func = method
            method = "pyarrow"

        if func is not None:
            return self._apply_cache_decorator(
                method, func, cache_instance_id=cache_instance_id, ttl=ttl,
                calls=calls, validate_return=validate_return,
            )

        # Parametrized-decorator usage: ``@fc.cache(method=...)`` returns a
        # decorator that receives the function being decorated.
        def decorator(target: Callable):
            return self._apply_cache_decorator(
                method, target, cache_instance_id=cache_instance_id, ttl=ttl,
                calls=calls, validate_return=validate_return,
            )

        return decorator

    def _apply_cache_decorator(
        self,
        method: CacheFormat,
        func: Callable,
        cache_instance_id: str = None,
        ttl: timedelta = None,
        calls: CallsSpec = (1, False),
        validate_return: bool = True,
    ):
        n_last_calls, add_timestamp = self._normalize_calls(calls)
        fixed_instance_id = cache_instance_id

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            base_id = (
                fixed_instance_id
                if fixed_instance_id is not None
                else self.cache_instance_id(func, method, *args, **kwargs)
            )

            if n_last_calls == 0:
                return func(*args, **kwargs)

            if n_last_calls == 1 and not add_timestamp:
                if self.cache_container.exists(base_id):
                    return self.deserialize(base_id)
                value = func(*args, **kwargs)
                if validate_return:
                    self._check_return_type(value, method, func)
                self.serialize(base_id, value, method=method, ttl=ttl)
                return value

            value = func(*args, **kwargs)
            if validate_return:
                self._check_return_type(value, method, func)
            version_id = self._new_version_id(base_id, add_timestamp=add_timestamp)
            self.serialize(version_id, value, method=method, ttl=ttl)
            self._prune_versions(base_id, keep=n_last_calls)
            return value

        return wrapper

    def json_cache(
        self, func: Callable = None, calls: CallsSpec = (1, False),
        ttl: timedelta = None, validate_return: bool = True,
    ):
        return self.cache(
            method="json", func=func, calls=calls, ttl=ttl, validate_return=validate_return
        )

    def pyarrow_cache(
        self, func: Callable = None, calls: CallsSpec = (1, False),
        ttl: timedelta = None, validate_return: bool = True,
    ):
        return self.cache(
            method="pyarrow", func=func, calls=calls, ttl=ttl, validate_return=validate_return
        )
