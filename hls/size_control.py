from __future__ import annotations

from typing import Dict, List, Optional


def allocate_bitrates_for_target_size(
    duration_seconds: float,
    resolutions: List[int],
    target_size_bytes: int,
    audio_kbps: int = 128,
) -> Dict[int, int]:
    """Compute approximate per-variant video kbps to hit target size across variants.
    Very rough split: weight by pixel count.
    """
    if duration_seconds <= 0:
        duration_seconds = 1.0
    weights = {r: r for r in resolutions}  # proxy for pixel count
    total_w = sum(weights.values()) or 1
    # total video budget kbps from target size minus audio budget (for a single audio)
    total_kbits = max(1, int((target_size_bytes * 8) / 1000))
    video_budget_kbps = max(1, total_kbits // (duration_seconds / 1.0) - audio_kbps)
    per_variant: Dict[int, int] = {}
    for r in resolutions:
        per_variant[r] = max(300, int(video_budget_kbps * (weights[r] / total_w)))
    return per_variant
