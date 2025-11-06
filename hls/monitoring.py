from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import Counter, Histogram, start_http_server  # type: ignore
except Exception:
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    start_http_server = None  # type: ignore


class Metrics:
    def __init__(self) -> None:
        if Counter is None or Histogram is None:
            self.enabled = False
            return
        self.enabled = True
        self.films_processed = Counter("films_processed", "Number of films processed")
        self.encode_time_seconds = Histogram("encode_time_seconds", "Per-variant encode time (s)")

    def start_server(self, port: int = 8000) -> None:
        if self.enabled and start_http_server is not None:
            start_http_server(port)

    def inc_films(self) -> None:
        if self.enabled:
            self.films_processed.inc()

    def observe_encode_time(self, duration: float) -> None:
        if self.enabled:
            self.encode_time_seconds.observe(duration)


metrics = Metrics()
