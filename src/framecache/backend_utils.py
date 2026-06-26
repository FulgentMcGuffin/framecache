import redis
from datetime import date
import polars as pl
import subprocess
import time


def is_redis_installed():
    """Check if Redis is installed"""
    try:
        result = subprocess.run(
            ["redis-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=7,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"Error checking if Redis is installed: {e}")
        return False


_DEFAULT_REDIS_INSTANCE = (
    redis.Redis(host="localhost", port=6379, db=0) if is_redis_installed() else None
)


def is_redis_running(redis_instance: redis.Redis | None = None):
    """Check if Redis is running"""
    try:
        r = _DEFAULT_REDIS_INSTANCE if redis_instance is None else redis_instance
        if r is not None:
            r.ping()
            return True
        else:
            return False
        return True
    except (redis.ConnectionError, redis.TimeoutError, FileNotFoundError):
        return False
    return False


def start_redis():
    """Start Redis server"""
    try:
        # Start Redis in background without reloading or saving to disk
        process = subprocess.Popen(
            ["redis-server", "--daemonize", "yes", "--save", "--appendonly", "no"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait a moment for Redis to start
        time.sleep(10)
        # Check if it started successfully
        if is_redis_running():
            print("Redis started successfully")
            return True
        else:
            print("Failed to start Redis")
            return False
    except FileNotFoundError:
        print("redis-server command not found")
        return False
    except Exception as e:
        print(f"Error starting Redis: {e}")
        return False
