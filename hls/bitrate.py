from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import ffmpeg  # type: ignore
except Exception:
    ffmpeg = None

# Default preset ladder in kbps
PRESET_LADDER_KBPS: Dict[int, int] = {
    1080: 5500,
    720: 3000,
    480: 1200,
}


def analyze_complexity(input_path: Path) -> float:
    """Return a very rough complexity score [0..1] based on source bitrate.
    0 = very simple, 1 = very complex.
    """
    if ffmpeg is None:
        return 0.5
    try:
        probe = ffmpeg.probe(str(input_path))
        fmt = probe.get("format", {})
        bit_rate = float(fmt.get("bit_rate", 0.0))
        duration = float(fmt.get("duration", 1.0)) or 1.0
        # bitrate per 1080p minute baseline ~ 8 Mbps assumed
        bps = bit_rate or 4_000_000.0
        normalized = min(1.0, max(0.0, bps / 8_000_000.0))
        return normalized
    except Exception:
        return 0.5


def choose_bitrate_ladder(
    input_path: Path,
    target_resolutions: List[int],
    mode: str = "preset",
    preset_ladder_kbps: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    """Return kbps bitrate targets per resolution.
    mode: preset | per-title | abr
    """
    preset = dict(preset_ladder_kbps or PRESET_LADDER_KBPS)
    res_sorted = sorted(set(target_resolutions), reverse=True)

    if mode == "preset":
        return {r: preset.get(r, max(600, int(r * 4))) for r in res_sorted}

    if mode == "per-title":
        complexity = analyze_complexity(input_path)
        # Scale preset up/down by complexity factor between 0.6x and 1.2x
        scale = 0.6 + 0.6 * complexity
        ladder = {r: max(400, int(preset.get(r, r * 4) * scale)) for r in res_sorted}
        return ladder

    if mode == "abr":
        # If ABR mode requested without explicit targets, fall back to preset
        return {r: preset.get(r, max(600, int(r * 4))) for r in res_sorted}

    return {r: preset.get(r, max(600, int(r * 4))) for r in res_sorted}
