# utils/timers.py
import time
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.time()
    result = {"elapsed_ms": 0}
    yield result
    end = time.time()
    result["elapsed_ms"] = (end - start) * 1000# Retrieval and generation timers
