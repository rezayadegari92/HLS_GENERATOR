from __future__ import annotations

import subprocess
import os
from contextlib import contextmanager
from threading import Semaphore, Lock
from typing import Dict, Iterator, List, Optional, Tuple


def detect_nvidia_gpus() -> List[int]:
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader"
        ], stderr=subprocess.DEVNULL, text=True)
        indices = [int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
        return indices
    except Exception:
        return []


def encoder_name_for(hw: str, codec: str) -> Optional[str]:
    hw = (hw or "").lower()
    codec = (codec or "h264").lower()
    if hw == "nvenc":
        return f"{codec}_nvenc"
    if hw == "qsv":
        return f"{codec}_qsv"
    if hw == "amf":
        return f"{codec}_amf"
    return None


class GPUManager:
    def __init__(self, max_sessions_per_gpu: int = 2) -> None:
        self.gpu_indices = detect_nvidia_gpus()
        self._locks: Dict[int, Semaphore] = {idx: Semaphore(max_sessions_per_gpu) for idx in self.gpu_indices}
        self._rr_lock = Lock()
        self._rr_ptr = 0

    def available(self) -> bool:
        return len(self.gpu_indices) > 0

    def _next_gpu(self) -> Optional[int]:
        if not self.gpu_indices:
            return None
        with self._rr_lock:
            idx = self.gpu_indices[self._rr_ptr % len(self.gpu_indices)]
            self._rr_ptr += 1
            return idx

    @contextmanager
    def allocate(self) -> Iterator[Optional[int]]:
        if not self.gpu_indices:
            yield None
            return
        gpu = self._next_gpu()
        if gpu is None:
            yield None
            return
        sem = self._locks[gpu]
        acquired = sem.acquire(blocking=False)
        try:
            if not acquired:
                # Saturated; caller should fall back to CPU
                yield None
            else:
                yield gpu
        finally:
            if acquired:
                sem.release()
