from __future__ import annotations

from typing import Any, Dict, Optional
import os
import sys

try:
    from celery import Celery  # type: ignore
except Exception:
    Celery = None  # type: ignore


def get_celery_app(broker_url: str, backend_url: Optional[str] = None) -> Any:
    if Celery is None:
        raise RuntimeError("Celery not installed. pip install celery redis")
    app = Celery("hls_generator", broker=broker_url, backend=backend_url)
    return app


# Lazy task definition helper; users can import and register tasks in their worker
def register_process_task(app: Any, func) -> None:
    app.task(name="hls.process_film_task")(func)


# Default Celery app for `celery -A hls.orchestrator worker`
if Celery is not None:
    app = Celery(
        "hls_generator",
        broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        backend=os.getenv("CELERY_BACKEND_URL"),
    )

    @app.task(name="hls.process_film_task")
    def process_film_task(payload: Dict[str, Any]) -> str:
        """Celery task wrapper that calls hls_generator_v2.process_film for one film.
        Expects a payload dict with the keys set by the enqueueing code.
        """
        # Ensure repository root is importable inside worker (container-safe)
        repo_root = os.environ.get("APP_ROOT", "/app")
        if repo_root and repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from pathlib import Path
        from hls_generator_v2 import process_film

        film_name = payload["film_name"]
        inputs_by_res = {int(k): Path(v) for k, v in payload["inputs_by_res"].items()}
        output_root = Path(payload["output_root"]) 
        segment_time = int(payload["segment_time"]) 
        overwrite = bool(payload["overwrite"]) 
        prefer_copy = bool(payload["prefer_copy"]) 
        thumb_interval = int(payload["thumb_interval"]) 
        thumb_width = int(payload["thumb_width"]) 
        variant_workers = int(payload["variant_workers"]) 
        audio_mode = str(payload["audio_mode"]) 
        hw_encoder = payload.get("hw_encoder")
        quality_mode = payload.get("quality_mode", "preset")
        per_title = bool(payload.get("per_title", False))
        explicit_bitrates = payload.get("explicit_bitrates")

        process_film(
            film_name,
            inputs_by_res,
            output_root,
            segment_time,
            overwrite,
            prefer_copy,
            thumb_interval,
            thumb_width,
            variant_workers,
            audio_mode,
            hw_encoder=hw_encoder,
            quality_mode=quality_mode,
            per_title=per_title,
            explicit_bitrates=explicit_bitrates,
        )
        return f"done:{film_name}"
