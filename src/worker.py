"""Queue backend: Redis + RQ when REDIS_URL is set, otherwise an in-process thread pool.
Run extra workers with:  rq worker --url $REDIS_URL"""
import os
from concurrent.futures import ThreadPoolExecutor

from .pipeline import process

_POOL = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))
_Q = None


def submit(job_id: str, path: str, caption: str):
    global _Q
    url = os.getenv("REDIS_URL")
    if url:
        if _Q is None:
            from redis import Redis
            from rq import Queue
            _Q = Queue(connection=Redis.from_url(url))
        _Q.enqueue("src.pipeline.process", job_id, path, caption, job_timeout=600)
    else:
        _POOL.submit(process, job_id, path, caption)
