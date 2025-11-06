from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import os

try:
    import ray  # type: ignore
except Exception:
    ray = None  # Optional

try:
    from celery import Celery  # type: ignore
except Exception:
    Celery = None  # Optional


class BaseExecutor:
    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def map(self, fn: Callable[..., Any], iterable: Iterable[Any]) -> Iterable[Any]:
        # Simple default map using submit and as_completed
        futures: List[Any] = [self.submit(fn, item) for item in iterable]
        for f in futures:
            yield self.result(f)

    def result(self, f: Any) -> Any:
        # Override for backends
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


class ThreadExecutor(BaseExecutor):
    def __init__(self, max_workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        return self._pool.submit(fn, *args, **kwargs)

    def result(self, f: Future) -> Any:
        return f.result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


class ProcessExecutor(BaseExecutor):
    def __init__(self, max_workers: int) -> None:
        self._pool = ProcessPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        return self._pool.submit(fn, *args, **kwargs)

    def result(self, f: Future) -> Any:
        return f.result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


class CeleryExecutor(BaseExecutor):
    def __init__(self, broker_url: str, backend_url: Optional[str] = None, app_name: str = "hls_generator") -> None:
        if Celery is None:
            raise RuntimeError("Celery is not installed. pip install celery redis")
        self._app = Celery(app_name, broker=broker_url, backend=backend_url)

    def submit(self, fn_path: str, *args: Any, **kwargs: Any):
        # fn_path is the dotted path to a celery task, e.g. "hls.orchestrator.process_film_task"
        return self._app.send_task(fn_path, args=args, kwargs=kwargs)

    def result(self, async_result: Any) -> Any:
        return async_result.get()


class RayExecutor(BaseExecutor):
    def __init__(self, max_workers: int) -> None:
        if ray is None:
            raise RuntimeError("Ray is not installed. pip install ray")
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        remote_fn = ray.remote(fn)  # type: ignore[attr-defined]
        return remote_fn.remote(*args, **kwargs)  # type: ignore[attr-defined]

    def result(self, object_ref: Any) -> Any:
        return ray.get(object_ref)  # type: ignore


def make_executor(name: str, max_workers: int, *, broker_url: Optional[str] = None, backend_url: Optional[str] = None) -> BaseExecutor:
    name = (name or "").lower()
    if name in ("threadpool", "thread"):
        return ThreadExecutor(max_workers=max_workers)
    if name in ("local-multiproc", "process", "multiprocess"):
        return ProcessExecutor(max_workers=max_workers)
    if name == "celery":
        if not broker_url:
            raise ValueError("broker_url is required for Celery executor")
        return CeleryExecutor(broker_url=broker_url, backend_url=backend_url)
    if name == "ray":
        return RayExecutor(max_workers=max_workers)
    raise ValueError(f"Unknown executor: {name}")
