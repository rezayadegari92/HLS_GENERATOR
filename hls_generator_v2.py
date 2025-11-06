#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
import sys
import time
import subprocess
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# New modules for scaling and quality control
try:
    from hls.executors import make_executor, BaseExecutor
except Exception:
    make_executor = None  # type: ignore
    BaseExecutor = object  # type: ignore

try:
    from hls.gpu_manager import GPUManager, encoder_name_for
except Exception:
    GPUManager = None  # type: ignore
    def encoder_name_for(hw: str, codec: str) -> Optional[str]:  # type: ignore
        return None

try:
    from hls.bitrate import choose_bitrate_ladder
except Exception:
    def choose_bitrate_ladder(*args, **kwargs):  # type: ignore
        # Fallback to preset-like values
        resolutions = kwargs.get('target_resolutions') or (args[1] if len(args) > 1 else [1080, 720, 480])
        return {r: {1080: 5500, 720: 3000, 480: 1200}.get(r, max(600, r * 4)) for r in resolutions}

try:
    from hls.size_control import allocate_bitrates_for_target_size
except Exception:
    def allocate_bitrates_for_target_size(*args, **kwargs):  # type: ignore
        return {}

try:
    from hls.monitoring import metrics
except Exception:
    class _Dummy:
        def inc_films(self):
            pass
        def observe_encode_time(self, duration: float):
            pass
    metrics = _Dummy()  # type: ignore

try:
    import ffmpeg
except ImportError:
    print("Error: ffmpeg-python is required. Install with: pip install ffmpeg-python")
    sys.exit(1)

# ------------------------------
# Configuration defaults
# ------------------------------
DEFAULT_RESOLUTIONS = [1080, 720, 480]
DEFAULT_SEGMENT_TIME = 6
DEFAULT_HLS_PLAYLIST_NAME = "index.m3u8"
DEFAULT_HLS_SEGMENT_PATTERN = "segment_%05d.ts"
THUMB_INTERVAL = 20  # seconds between thumbnails

# Encoding presets tuned for speed while keeping quality reasonable
# Change: use 'veryfast' instead of 'ultrafast' to avoid bitrate bloat at same CRF
RESOLUTION_PRESETS = {
    1080: {"bandwidth": 6_000_000, "resolution": "1920x1080", "crf": 21, "preset": "veryfast", "threads": 0},
    720: {"bandwidth": 3_500_000, "resolution": "1280x720", "crf": 23, "preset": "veryfast", "threads": 0},
    # Fix: keep CRF within 18-23 for H.264 to prevent size bloat while preserving visual quality
    480: {"bandwidth": 1_200_000, "resolution": "854x480", "crf": 23, "preset": "veryfast", "threads": 0},
}

# Expanded supported extensions to improve discovery
SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".ts", ".m2ts"}

# Global GPU manager (initialized in main based on CLI)
GPU_MGR: Optional["GPUManager"] = None

# ------------------------------
# Timer decorator
# ------------------------------
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@dataclass
class VariantResult:
    film_name: str
    resolution: int
    input_path: Path
    output_dir: Path
    playlist_path: Path
    success: bool
    attempted_transcode: bool
    error_message: Optional[str] = None

@dataclass
class SubtitleTrack:
    index: int
    language: str
    title: str
    codec: str

@dataclass
class AudioTrack:
    index: int
    language: str
    title: str
    codec: str
    channels: int
    sample_rate: int

# ------------------------------
# Logging
# ------------------------------
def configure_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

# ------------------------------
# FFmpeg capability helpers
# ------------------------------

@lru_cache(maxsize=1)
def _ffmpeg_filters_output() -> str:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"], stderr=subprocess.STDOUT, text=True)
        return out
    except Exception:
        return ""

def ffmpeg_supports_filter(filter_name: str) -> bool:
    out = _ffmpeg_filters_output()
    return (f" {filter_name} " in out) or (f" {filter_name}\n" in out) or (f"\n{filter_name} " in out)

@lru_cache(maxsize=1)
def _ffmpeg_encoders_output() -> str:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True)
        return out
    except Exception:
        return ""

def ffmpeg_supports_encoder(encoder_name: str) -> bool:
    if not encoder_name:
        return False
    out = _ffmpeg_encoders_output()
    return (f" {encoder_name} " in out) or (f" {encoder_name}\n" in out) or (f"\n{encoder_name} " in out)

@lru_cache(maxsize=1)
def _ffmpeg_hwaccels_output() -> str:
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-hwaccels"], stderr=subprocess.STDOUT, text=True)
        return out
    except Exception:
        return ""

def ffmpeg_supports_hwaccel(name: str) -> bool:
    out = _ffmpeg_hwaccels_output()
    return (f" {name}\n" in out) or (f"\n{name}\n" in out) or (out.strip().endswith(name))

# ------------------------------
# FFmpeg helpers using ffmpeg-python
# ------------------------------
def probe_file(input_path: Path) -> Dict:
    """Probe file and return stream information."""
    try:
        probe = ffmpeg.probe(str(input_path))
        return probe
    except ffmpeg.Error as e:
        logging.warning("ffprobe failed for %s: %s", input_path, e.stderr.decode() if e.stderr else str(e))
        return {}

def get_video_info(input_path: Path) -> Tuple[Optional[int], Optional[float]]:
    """Get video height and duration."""
    try:
        probe = probe_file(input_path)
        video_stream = next((s for s in probe.get('streams', []) if s['codec_type'] == 'video'), None)
        if video_stream:
            height = int(video_stream.get('height', 0))
            duration = float(probe.get('format', {}).get('duration', 0))
            return height, duration
        return None, None
    except Exception as e:
        logging.debug("Failed to get video info for %s: %s", input_path, e)
        return None, None

def get_video_fps(input_path: Path) -> Optional[float]:
    """Best-effort FPS detection using avg_frame_rate or r_frame_rate."""
    try:
        probe = probe_file(input_path)
        video_stream = next((s for s in probe.get('streams', []) if s.get('codec_type') == 'video'), None)
        if not video_stream:
            return None
        fps_str = video_stream.get('avg_frame_rate') or video_stream.get('r_frame_rate') or "0/1"
        if fps_str and fps_str != "N/A":
            num, den = fps_str.split('/')
            num_i = int(num or 0)
            den_i = int(den or 1)
            if den_i == 0:
                return None
            return float(Fraction(num_i, den_i))
        return None
    except Exception as e:
        logging.debug("Failed to get FPS for %s: %s", input_path, e)
        return None

def get_subtitle_tracks(input_path: Path) -> List[SubtitleTrack]:
    """Extract subtitle track information."""
    probe = probe_file(input_path)
    subtitle_tracks = []
    
    for stream in probe.get('streams', []):
        if stream.get('codec_type') == 'subtitle':
            tags = stream.get('tags', {})
            language = tags.get('language', 'und').lower()
            title = tags.get('title', language.upper())
            codec = stream.get('codec_name', 'unknown')
            index = int(stream.get('index', 0))
            
            subtitle_tracks.append(SubtitleTrack(
                index=index,
                language=language,
                title=title,
                codec=codec
            ))
    
    return subtitle_tracks

def get_audio_tracks(input_path: Path) -> List[AudioTrack]:
    """Extract audio track information."""
    probe = probe_file(input_path)
    audio_tracks = []
    
    for stream in probe.get('streams', []):
        if stream.get('codec_type') == 'audio':
            tags = stream.get('tags', {})
            language = tags.get('language', 'und').lower()
            title = tags.get('title', language.upper())
            codec = stream.get('codec_name', 'unknown')
            index = int(stream.get('index', 0))
            channels = int(stream.get('channels', 2))
            sample_rate = int(stream.get('sample_rate', 48000))
            
            audio_tracks.append(AudioTrack(
                index=index,
                language=language,
                title=title,
                codec=codec,
                channels=channels,
                sample_rate=sample_rate
            ))
    
    return audio_tracks

def input_has_audio(input_path: Path) -> bool:
    """Check if input has audio streams."""
    probe = probe_file(input_path)
    return any(s.get('codec_type') == 'audio' for s in probe.get('streams', []))

def map_height_to_resolution(height: Optional[int], resolutions: List[int]) -> Optional[int]:
    """Map video height to target resolution."""
    if not resolutions:
        return None
    sorted_res = sorted(resolutions)
    if height is None:
        return sorted_res[-1]
    candidates = [r for r in sorted_res if r <= height]
    return candidates[-1] if candidates else sorted_res[0]

# ------------------------------
# Discovery
# ------------------------------
def discover_films(source_dir: Path, resolutions: List[int]) -> Dict[str, Dict[int, Path]]:
    """Discover films and their resolutions."""
    film_map: Dict[str, Dict[int, Path]] = {}
    res_dir_pattern = re.compile(r"^(?P<num>\d{3,4})(?:[pP])?$")

    for root, dirs, files in os.walk(source_dir):
        parent = Path(root)
        base = parent.name.strip()
        m = res_dir_pattern.match(base)
        
        if m:
            try:
                res_val = int(m.group("num"))
            except (TypeError, ValueError):
                continue
            if res_val not in resolutions:
                continue
            
            film_name = parent.parent.name.strip()
            if not film_name:
                continue
                
            for entry in sorted(os.listdir(root)):
                p = Path(root) / entry
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    film_map.setdefault(film_name, {})[res_val] = p
                    break
            continue

        if parent.parent.resolve() != Path(source_dir).resolve():
            continue

        film_name = parent.name.strip()
        if not film_name:
            continue

        media_files = [
            Path(root) / e for e in sorted(os.listdir(root))
            if (Path(root) / e).is_file() and (Path(root) / e).suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        if media_files:
            first_media = media_files[0]
            height, _ = get_video_info(first_media)
            mapped_res = map_height_to_resolution(height, resolutions)
            if mapped_res is not None:
                film_map.setdefault(film_name, {})[mapped_res] = first_media

    return film_map

# ------------------------------
# HLS generation
# ------------------------------
def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def extract_subtitles_to_vtt(input_path: Path, film_output_dir: Path, overwrite: bool) -> List[Tuple[str, str, Path]]:
    """Extract all subtitle streams to WebVTT files."""
    subtitle_tracks = get_subtitle_tracks(input_path)
    if not subtitle_tracks:
        logging.debug("No subtitle tracks found in %s", input_path)
        return []

    subs_dir = film_output_dir / "subtitles"
    ensure_dir(subs_dir)
    results: List[Tuple[str, str, Path]] = []

    for i, track in enumerate(subtitle_tracks):
        # Skip image-based subtitles that can't be converted to WebVTT
        if track.codec in ['dvd_subtitle', 'pgs', 'hdmv_pgs_subtitle']:
            logging.debug("Skipping image-based subtitle track %d (%s)", track.index, track.codec)
            continue

        lang_suffix = f"_{track.language}" if track.language != 'und' else f"_{i}"
        out_vtt = subs_dir / f"subtitle{lang_suffix}.vtt"
        
        if out_vtt.exists() and not overwrite:
            results.append((track.language, track.title, out_vtt.relative_to(film_output_dir)))
            continue

        try:
            input_stream = ffmpeg.input(str(input_path))
            output_stream = ffmpeg.output(
                input_stream[f's:{i}'],  # Use subtitle stream ordinal
                str(out_vtt),
                format='webvtt'
            )
            
            if overwrite:
                output_stream = ffmpeg.overwrite_output(output_stream)
            
            ffmpeg.run(output_stream, quiet=True, overwrite_output=overwrite)
            results.append((track.language, track.title, out_vtt.relative_to(film_output_dir)))
            logging.debug("Extracted subtitle: %s (%s)", out_vtt.name, track.language)
            
        except ffmpeg.Error as e:
            logging.warning("Failed to extract subtitle %d: %s", i, e.stderr.decode() if e.stderr else str(e))
        except Exception as e:
            logging.warning("Unexpected error extracting subtitle %d: %s", i, str(e))

    return results

def generate_thumbnails_vtt(input_path: Path, film_output_dir: Path, 
                          overwrite: bool, interval: int = THUMB_INTERVAL, 
                          width: int = 160) -> None:
    """Generate single thumbnails.vtt for all qualities."""
    thumb_dir = film_output_dir / "thumbnails"
    ensure_dir(thumb_dir)
    vtt_path = film_output_dir / "thumbnails.vtt"
    
    if vtt_path.exists() and not overwrite:
        logging.info("Thumbnails already exist, skipping: %s", vtt_path)
        return

    try:
        # Get video duration
        _, duration = get_video_info(input_path)
        if not duration:
            logging.warning("Could not determine video duration for %s", input_path)
            return

        # Extract thumbnails using ffmpeg-python
        thumb_pattern = str(thumb_dir / "thumb_%04d.jpg")
        
        (
            ffmpeg
            .input(str(input_path))
            .filter('fps', f'1/{interval}')
            .filter('scale', width, -1)
            .output(thumb_pattern)
            .overwrite_output()
            .run(quiet=True)
        )

        # Build VTT file
        lines = ["WEBVTT\n"]
        thumb_files = sorted(thumb_dir.glob("thumb_*.jpg"))
        
        for i, img in enumerate(thumb_files):
            start = i * interval
            end = min((i + 1) * interval, duration)
            start_time = time.strftime('%H:%M:%S.000', time.gmtime(start))
            end_time = time.strftime('%H:%M:%S.000', time.gmtime(end))
            lines.append(f"{start_time} --> {end_time}")
            lines.append(f"thumbnails/{img.name}\n")

        vtt_path.write_text("\n".join(lines), encoding="utf-8")
        logging.info("Generated thumbnails.vtt: %s (%d thumbnails)", vtt_path, len(thumb_files))

    except ffmpeg.Error as e:
        logging.warning("Thumbnail generation failed: %s", e.stderr.decode() if e.stderr else str(e))
    except Exception as e:
        logging.warning("Unexpected error generating thumbnails: %s", str(e))

def _deterministic_lang_tag(language: str, index: int) -> str:
    lang = (language or "und").lower()
    return f"{lang}-{index:02d}" if lang != "und" else f"und-{index:02d}"


def extract_all_audio_tracks_to_hls(input_path: Path, film_output_dir: Path, segment_time: int, overwrite: bool) -> List[Tuple[str, str, Path]]:
    """Extract all audio tracks as separate HLS audio-only streams."""
    audio_tracks = get_audio_tracks(input_path)
    if not audio_tracks:
        logging.debug("No audio tracks found in %s", input_path)
        return []

    audio_dir = film_output_dir / "audio"
    ensure_dir(audio_dir)
    results: List[Tuple[str, str, Path]] = []

    for i, track in enumerate(audio_tracks):
        # Deterministic folder: audio/<lang>-<ordinal>
        tag = _deterministic_lang_tag(track.language, i)
        audio_subdir = audio_dir / f"{tag}"
        ensure_dir(audio_subdir)
        
        audio_playlist = audio_subdir / "index.m3u8"
        audio_segment_pattern = str(audio_subdir / "segment_%05d.ts")
        
        if audio_playlist.exists() and not overwrite:
            results.append((track.language, track.title, audio_playlist.relative_to(film_output_dir)))
            continue

        try:
            input_stream = ffmpeg.input(str(input_path))
            
            # Extract audio stream and create HLS segments
            audio_stream = input_stream[f'a:{i}']
            
            # Prefer stream copy for AAC to speed up processing
            use_copy = (track.codec or "").lower() in ("aac",)
            output_args = {
                'format': 'hls',
                'hls_time': segment_time,
                'hls_playlist_type': 'vod',
                'hls_segment_filename': audio_segment_pattern,
                'vn': None  # No video
            }
            if use_copy:
                output_args['c:a'] = 'copy'
            else:
                output_args.update({
                    'acodec': 'aac',
                    'b:a': '128k',
                    'ac': min(track.channels, 2),  # Stereo max for HLS compatibility
                    'ar': '48000',
                })
            
            output_stream = ffmpeg.output(audio_stream, str(audio_playlist), **output_args)
            
            if overwrite:
                output_stream = ffmpeg.overwrite_output(output_stream)
            
            ffmpeg.run(output_stream, quiet=True, overwrite_output=overwrite)
            results.append((track.language, track.title, audio_playlist.relative_to(film_output_dir)))
            logging.debug("Extracted audio HLS: %s (%s)", audio_playlist.name, track.language)
            
        except ffmpeg.Error as e:
            logging.warning("Failed to extract audio HLS %d: %s", i, e.stderr.decode() if e.stderr else str(e))
            # Still register the intended rendition path for playlist wiring
            results.append((track.language, track.title, audio_playlist.relative_to(film_output_dir)))
        except Exception as e:
            logging.warning("Unexpected error extracting audio HLS %d: %s", i, str(e))
            # Still register the intended rendition path for playlist wiring
            results.append((track.language, track.title, audio_playlist.relative_to(film_output_dir)))

    return results

def generate_hls_variant_parallel_segments(
    film_name: str,
    resolution: int,
    input_path: Path,
    output_root: Path,
    segment_time: int,
    overwrite: bool,
    prefer_copy: bool = True,
    audio_mode: str = "separate",
    *,
    hw_encoder: Optional[str] = None,
    video_codec: str = "h264",
    quality_mode: str = "preset",
    bitrate_kbps: Optional[int] = None,
) -> VariantResult:
    """Generate HLS variant using parallel segment processing for maximum speed."""
    output_dir = output_root / film_name / str(resolution)
    ensure_dir(output_dir)

    playlist_path = output_dir / DEFAULT_HLS_PLAYLIST_NAME
    if playlist_path.exists() and not overwrite:
        return VariantResult(film_name, resolution, input_path, output_dir, playlist_path, True, False)

    # Get video duration for segment calculation
    _, duration = get_video_info(input_path)
    if not duration:
        logging.warning("Could not determine duration for %s", input_path)
        duration = 3600  # fallback

    # Use single-pass encoding with FFmpeg's built-in parallel processing
    try:
        # Detect FPS to set GOP size aligned with segment time for faster HLS
        fps = get_video_fps(input_path) or 25.0
        gop_size = max(1, int(round(fps * segment_time)))

        # Choose encoder with capability checks
        encoder_candidate = encoder_name_for(hw_encoder, video_codec) if hw_encoder else 'libx264'
        is_nvenc_candidate = bool(encoder_candidate and 'nvenc' in encoder_candidate)
        nvenc_available = is_nvenc_candidate and ffmpeg_supports_encoder(encoder_candidate)
        if is_nvenc_candidate and not nvenc_available:
            encoder = 'libx264'
            is_nvenc = False
        else:
            encoder = encoder_candidate
            is_nvenc = is_nvenc_candidate

        # Use CUDA hwaccel and GPU scaling only if both are supported
        use_cuda_hw = is_nvenc and ffmpeg_supports_hwaccel('cuda') and ffmpeg_supports_filter('scale_npp')
        input_stream = (
            ffmpeg.input(str(input_path), hwaccel='cuda', hwaccel_output_format='cuda')
            if use_cuda_hw else ffmpeg.input(str(input_path))
        )

        # Proper resolution scaling
        video_stream = input_stream['v:0']
        if resolution == 1080:
            target_w, target_h = 1920, 1080
        elif resolution == 720:
            target_w, target_h = 1280, 720
        elif resolution == 480:
            target_w, target_h = 854, 480
        else:
            # Fallback: best-effort maintain aspect via height
            target_w, target_h = -2, resolution

        if use_cuda_hw:
            # GPU scale for speed
            video_stream = ffmpeg.filter(video_stream, 'scale_npp', target_w, target_h)
        else:
            video_stream = ffmpeg.filter(video_stream, 'scale', target_w, target_h)

        # Get ultra-fast settings
        preset_info = RESOLUTION_PRESETS.get(resolution, {})

        output_args = {
            'format': 'hls',
            'hls_time': segment_time,
            'hls_playlist_type': 'vod',
            'hls_segment_filename': str(output_dir / DEFAULT_HLS_SEGMENT_PATTERN),
            'c:v': encoder,
            'pix_fmt': 'yuv420p',
            # Use ffmpeg '-threads 0' behavior to utilize all cores when value is 0
            'threads': preset_info.get('threads', 0),
            'hls_flags': 'independent_segments',
            'g': gop_size,
        }

        # Encoding options
        # Problem: Previous settings could inflate size (ultrafast + weak GOP control).
        # Fix: Use CRF within 18–23 range and veryfast preset; set keyint/min-keyint=scenecut=0 to align with 6s segments.
        # Impact: Maintains perceptual quality while keeping bitrate reasonable; prevents multi-GB inflation.
        if is_nvenc:
            target_kbps = bitrate_kbps or (preset_info.get('bandwidth', 3_000_000) // 1000)
            output_args.update({
                'preset': 'p1',
                'rc': 'cbr',
                'b:v': f'{target_kbps}k',
                'rc-lookahead': '0',
                'spatial_aq': '0',
                'temporal_aq': '0',
                'delay': '0',
            })
        else:
            # CPU encoding tuned for speed without excessive size bloat
            output_args.update({
                'crf': str(preset_info.get('crf', 23)),
                'preset': preset_info.get('preset', 'veryfast'),
                'tune': 'zerolatency',
                # Keep keyframes aligned to segment boundaries and avoid scene-cut drift
                'x264opts': f'keyint={gop_size}:min-keyint={gop_size}:scenecut=0:fast-pskip:no-mixed-refs:no-8x8dct:no-cabac',
            })

        # Audio handling for separate mode
        if audio_mode == "separate":
            output_args['an'] = None  # No audio in video stream
            streams_to_output = [video_stream]
        else:
            # Include first audio track
            audio_stream = input_stream['a:0']
            streams_to_output = [video_stream, audio_stream]
            output_args.update({
                'c:a': 'aac',
                'b:a': '128k',
                'ac': '2',
                'ar': '48000'
            })

        # Execute with hardware acceleration if available
        if is_nvenc and GPU_MGR:
            with GPU_MGR.allocate() as gpu_id:
                if gpu_id is not None:
                    # Bind to specific GPU for NVENC
                    final_args = dict(output_args)
                    final_args['gpu'] = str(gpu_id)
                    output_stream = ffmpeg.output(*streams_to_output, str(playlist_path), **final_args)
                    if overwrite:
                        output_stream = ffmpeg.overwrite_output(output_stream)
                    ffmpeg.run(output_stream, quiet=True)
                else:
                    # Fallback to CPU
                    output_args['c:v'] = 'libx264'
                    # Remove NVENC-only options and switch to fast CPU settings
                    for k in ('preset', 'rc', 'rc-lookahead', 'spatial_aq', 'temporal_aq', 'gpu', 'delay'):
                        if k in output_args:
                            output_args.pop(k, None)
                    output_args.update({
                        'crf': str(preset_info.get('crf', 23)),
                        'preset': RESOLUTION_PRESETS.get(resolution, {}).get('preset', 'veryfast'),
                        'tune': 'zerolatency',
                        'x264opts': f'keyint={gop_size}:min-keyint={gop_size}:scenecut=0:fast-pskip:no-mixed-refs:no-8x8dct:no-cabac',
                    })
                    output_stream = ffmpeg.output(*streams_to_output, str(playlist_path), **output_args)
                    if overwrite:
                        output_stream = ffmpeg.overwrite_output(output_stream)
                    ffmpeg.run(output_stream, quiet=True)
        else:
            output_stream = ffmpeg.output(*streams_to_output, str(playlist_path), **output_args)
            if overwrite:
                output_stream = ffmpeg.overwrite_output(output_stream)
            ffmpeg.run(output_stream, quiet=True)

        logging.debug("Fast parallel encoding successful for %s %dp", film_name, resolution)
        return VariantResult(film_name, resolution, input_path, output_dir, playlist_path, True, True)

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logging.error("Fast HLS generation failed for %s %dp: %s", film_name, resolution, error_msg)
        return VariantResult(film_name, resolution, input_path, output_dir, playlist_path, False, True, error_msg)
    except Exception as e:
        error_msg = str(e)
        logging.error("Unexpected error for %s %dp: %s", film_name, resolution, error_msg)
        return VariantResult(film_name, resolution, input_path, output_dir, playlist_path, False, True, error_msg)


def generate_hls_variant_with_audio(
    film_name: str,
    resolution: int,
    input_path: Path,
    output_root: Path,
    segment_time: int,
    overwrite: bool,
    prefer_copy: bool = True,
    audio_mode: str = "separate",
    *,
    hw_encoder: Optional[str] = None,
    video_codec: str = "h264",
    quality_mode: str = "preset",
    bitrate_kbps: Optional[int] = None,
) -> VariantResult:
    """Generate HLS variant with configurable audio handling."""
    output_dir = output_root / film_name / str(resolution)
    ensure_dir(output_dir)

    playlist_path = output_dir / DEFAULT_HLS_PLAYLIST_NAME
    if playlist_path.exists() and not overwrite:
        return VariantResult(film_name, resolution, input_path, output_dir, playlist_path, True, False)

    segment_path = str(output_dir / DEFAULT_HLS_SEGMENT_PATTERN)
    has_audio = input_has_audio(input_path)
    
    try:
        # Detect FPS to set GOP size aligned with segment time for faster HLS
        fps = get_video_fps(input_path) or 25.0
        gop_size = max(1, int(round(fps * segment_time)))

        encoder = encoder_name_for(hw_encoder, video_codec) if hw_encoder else f'libx264'
        is_nvenc = bool(encoder and 'nvenc' in encoder)
        input_stream = (
            ffmpeg.input(str(input_path), hwaccel='cuda', hwaccel_output_format='cuda')
            if is_nvenc else ffmpeg.input(str(input_path))
        )

        # Try copy first if preferred and audio mode allows
        if prefer_copy and audio_mode == "muxed":
            try:
                output_args = {
                    'format': 'hls',
                    'hls_time': segment_time,
                    'hls_playlist_type': 'vod',
                    'hls_segment_filename': segment_path,
                    'c': 'copy'  # Copy streams without re-encoding
                }
                
                # Select video and audio streams for muxed mode
                video_stream = input_stream['v:0']
                streams_to_map = [video_stream]
                
                if has_audio:
                    # For muxed mode, include first audio track only
                    audio_stream = input_stream['a:0']
                    streams_to_map.append(audio_stream)
                
                output_stream = ffmpeg.output(*streams_to_map, str(playlist_path), **output_args)
                
                if overwrite:
                    output_stream = ffmpeg.overwrite_output(output_stream)
                    
                ffmpeg.run(output_stream, quiet=True)
                logging.debug("Copy successful for %s %dp (muxed audio)", film_name, resolution)
                return VariantResult(film_name, resolution, input_path, output_dir, playlist_path, True, False)
                
            except ffmpeg.Error as e:
                logging.debug("Copy failed for %s %dp, trying transcode: %s", film_name, resolution, 
                            e.stderr.decode() if e.stderr else str(e))

        # Transcode with optimized settings
        video_stream = input_stream['v:0']

        # Proper resolution scaling
        if resolution == 1080:
            target_w, target_h = 1920, 1080
        elif resolution == 720:
            target_w, target_h = 1280, 720
        elif resolution == 480:
            target_w, target_h = 854, 480
        else:
            target_w, target_h = -2, resolution

        if is_nvenc:
            video_stream = ffmpeg.filter(video_stream, 'scale_npp', target_w, target_h)
        else:
            video_stream = ffmpeg.filter(video_stream, 'scale', target_w, target_h)
        
        # Get resolution-specific settings for quality differentiation
        preset_info = RESOLUTION_PRESETS.get(resolution, {})
        encoder_preset = preset_info.get('preset', 'ultrafast')
        target_crf = preset_info.get('crf', 23)
        thread_count = preset_info.get('threads', 4)
        
        # Choose encoder
        output_args = {
            'format': 'hls',
            'hls_time': segment_time,
            'hls_playlist_type': 'vod',
            'hls_segment_filename': segment_path,
            'c:v': encoder,
                'pix_fmt': 'yuv420p',
            'movflags': '+faststart',
            # Use ffmpeg '-threads 0' behavior to utilize all cores when value is 0
            'threads': (0 if thread_count in (None, 0) else thread_count),
            'g': gop_size,
        }
        
        # Hardware vs software encoder settings
        # Problem: Using too aggressive presets can cause huge files at same CRF.
        # Fix: Prefer veryfast + CRF (18–23) and enforce GOP alignment with 6s segments.
        # Impact: Similar perceptual quality; controlled bitrate.
        if is_nvenc or (hw_encoder and encoder_name_for(hw_encoder, video_codec)):
            # Hardware encoder settings - optimized for speed
            if is_nvenc:
                output_args.update({
                    'preset': 'p1',  # Fastest NVENC preset
                    'rc': 'cbr',  # Constant bitrate for speed
                    'b:v': f'{bitrate_kbps or preset_info.get("bandwidth", 3000000) // 1000}k',
                    'rc-lookahead': '0',
                    'spatial_aq': '0',
                    'temporal_aq': '0',
                    # gpu id applied at execution time if available
                    'delay': '0',  # No delay for speed
                    # Note: tune=zerolatency not supported by NVENC
                })
            else:
                # QSV/AMF settings - fastest possible
                output_args.update({
                    'preset': 'veryfast',
                    'b:v': f'{bitrate_kbps or preset_info.get("bandwidth", 3000000) // 1000}k',
                })
        else:
            # Software encoder settings - ultrafast for maximum speed
            output_args['tune'] = 'zerolatency'  # Only add tune for software encoders
            if bitrate_kbps:
                output_args.update({
                    'b:v': f'{bitrate_kbps}k',
                    'preset': preset_info.get('preset', 'veryfast'),
                    'x264opts': 'no-scenecut',  # Disable scene detection for speed
                })
            else:
                output_args.update({
                    'crf': str(target_crf),
                    'preset': preset_info.get('preset', 'veryfast'),
                    'x264opts': f'keyint={gop_size}:min-keyint={gop_size}:scenecut=0:fast-pskip:no-mixed-refs',
                })
        
        streams_to_output = [video_stream]
        
        # Handle audio based on mode
        if has_audio and audio_mode == "muxed":
            # Include first audio track in video stream for muxed mode
            audio_stream = input_stream['a:0']
            streams_to_output.append(audio_stream)
            output_args.update({
                'c:a': 'aac',
                'b:a': '128k',
                'ac': '2',
                'ar': '48000'
            })
        elif audio_mode == "separate":
            # Video-only stream for separate audio mode
            output_args['an'] = None  # No audio in video stream

        output_stream = ffmpeg.output(*streams_to_output, str(playlist_path), **output_args)
        
        if overwrite:
            output_stream = ffmpeg.overwrite_output(output_stream)
            
        if is_nvenc and encoder_name_for(hw_encoder, video_codec):
            # Attempt GPU with best-effort; use session limiter if available
            mgr = GPU_MGR
            if mgr is not None:
                with mgr.allocate() as gpu_id:
                    if gpu_id is None:
                        # Saturated; fall back to CPU
                        logging.debug("GPU sessions saturated; falling back to CPU for %s %dp", film_name, resolution)
                        output_args['c:v'] = f'libx264'
                        # Remove NVENC-only flags when falling back
                        for k in ('preset', 'rc', 'rc-lookahead', 'spatial_aq', 'temporal_aq', 'gpu', 'delay'):
                            if k in output_args:
                                output_args.pop(k, None)
                        output_args.update({
                            'crf': str(target_crf),
                            'preset': RESOLUTION_PRESETS.get(resolution, {}).get('preset', 'veryfast'),
                            'tune': 'zerolatency',
                            'x264opts': f'keyint={gop_size}:min-keyint={gop_size}:scenecut=0:fast-pskip:no-mixed-refs:no-8x8dct:no-cabac',
                        })
                        output_stream = ffmpeg.output(*streams_to_output, str(playlist_path), **output_args)
                        ffmpeg.run(output_stream, quiet=True)
                    else:
                        try:
                            # Bind to a specific GPU for NVENC via encoder arg
                            final_args = dict(output_args)
                            final_args['gpu'] = str(gpu_id)
                            output_stream = ffmpeg.output(*streams_to_output, str(playlist_path), **final_args)
                            if overwrite:
                                output_stream = ffmpeg.overwrite_output(output_stream)
                            ffmpeg.run(output_stream, quiet=True)
                        except ffmpeg.Error:
                            logging.debug("HW encode failed on GPU %s; fallback CPU for %s %dp", gpu_id, film_name, resolution)
                            output_args['c:v'] = f'libx264'
                            for k in ('preset', 'rc', 'rc-lookahead', 'spatial_aq', 'temporal_aq', 'gpu', 'delay'):
                                if k in output_args:
                                    output_args.pop(k, None)
                            output_args.update({
                                'crf': str(target_crf),
                                'preset': RESOLUTION_PRESETS.get(resolution, {}).get('preset', 'veryfast'),
                                'tune': 'zerolatency',
                                'x264opts': f'keyint={gop_size}:min-keyint={gop_size}:scenecut=0:fast-pskip:no-mixed-refs:no-8x8dct:no-cabac',
                            })
                            output_stream = ffmpeg.output(*streams_to_output, str(playlist_path), **output_args)
                            ffmpeg.run(output_stream, quiet=True)
            else:
                try:
                    output_stream = ffmpeg.output(*streams_to_output, str(playlist_path), **output_args)
                    if overwrite:
                        output_stream = ffmpeg.overwrite_output(output_stream)
                    ffmpeg.run(output_stream, quiet=True)
                except ffmpeg.Error:
                    logging.debug("HW encode failed, falling back to CPU for %s %dp", film_name, resolution)
                    output_args['c:v'] = f'libx264'
                    for k in ('preset', 'rc', 'rc-lookahead', 'spatial_aq', 'temporal_aq', 'gpu', 'delay'):
                        if k in output_args:
                            output_args.pop(k, None)
                    output_args.update({
                        'crf': str(target_crf),
                        'preset': RESOLUTION_PRESETS.get(resolution, {}).get('preset', 'veryfast'),
                        'tune': 'zerolatency',
                        'x264opts': f'keyint={gop_size}:min-keyint={gop_size}:scenecut=0:fast-pskip:no-mixed-refs:no-8x8dct:no-cabac',
                    })
                    output_stream = ffmpeg.output(*streams_to_output, str(playlist_path), **output_args)
                    ffmpeg.run(output_stream, quiet=True)
        else:
            ffmpeg.run(output_stream, quiet=True)
        logging.debug("Transcode successful for %s %dp (%s audio)", film_name, resolution, audio_mode)
        return VariantResult(film_name, resolution, input_path, output_dir, playlist_path, True, True)

    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logging.error("HLS generation failed for %s %dp: %s", film_name, resolution, error_msg)
        return VariantResult(film_name, resolution, input_path, output_dir, playlist_path, False, True, error_msg)
    except Exception as e:
        error_msg = str(e)
        logging.error("Unexpected error for %s %dp: %s", film_name, resolution, error_msg)
        return VariantResult(film_name, resolution, input_path, output_dir, playlist_path, False, True, error_msg)

# ------------------------------
# Master playlist
# ------------------------------
def write_master_playlist(film_output_dir: Path, variants: List[Tuple[int, Path]], 
                         subtitles: Optional[List[Tuple[str, str, Path]]] = None,
                         audio_tracks: Optional[List[Tuple[str, str, Path]]] = None) -> Path:
    """Write master playlist with subtitles and audio tracks."""
    ensure_dir(film_output_dir)
    master_path = film_output_dir / "master.m3u8"
    lines = ["#EXTM3U", "#EXT-X-VERSION:3"]

    # Add audio renditions - include ALL audio tracks for language switching
    if audio_tracks:
        has_en_audio = any((lang or "").lower() in ("en", "eng") for lang, _, _ in audio_tracks)
        for i, (lang, name, rel_path) in enumerate(audio_tracks):
            rel_str = str(rel_path).replace("\\", "/")
            lang_attr = (lang or "und").lower()
            
            # Default: first English track if available, otherwise first track
            is_default = False
            if has_en_audio and lang_attr in ("en", "eng"):
                # First English track is default
                is_default = not any(lang_attr in ("en", "eng") for lang, _, _ in audio_tracks[:i])
            elif not has_en_audio and i == 0:
                # First track is default if no English
                is_default = True
            
            default = "YES" if is_default else "NO"
            autoselect = "YES" if lang_attr in ("en", "eng") else "NO"
            safe_name = name.replace('"', "'") if name else f"{lang_attr.upper()}"
            
            lines.append(
                f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="{safe_name}",'
                f'LANGUAGE="{lang_attr}",AUTOSELECT={autoselect},DEFAULT={default},URI="{rel_str}"'
            )

    # Add subtitle renditions
    if subtitles:
        has_en = any((lang or "").lower() in ("en", "eng") for lang, _, _ in subtitles)
        for lang, name, rel_path in subtitles:
            rel_str = str(rel_path).replace("\\", "/")
            lang_attr = (lang or "und").lower()
            default = "YES" if (has_en and lang_attr in ("en", "eng")) else "NO"
            safe_name = name.replace('"', "'") if name else lang_attr.upper()
            lines.append(
                f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="{safe_name}",'
                f'LANGUAGE="{lang_attr}",AUTOSELECT=YES,DEFAULT={default},URI="{rel_str}"'
            )

    # Add video variants
    for res, rel_playlist in sorted(variants, key=lambda x: x[0], reverse=True):
        preset = RESOLUTION_PRESETS.get(res)
        bandwidth = preset["bandwidth"] if preset else 1_000_000
        resolution_str = preset["resolution"] if preset else f"{res}x{res}"
        
        inf_line = f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},RESOLUTION={resolution_str}"
        
        # Add group references if present
        if audio_tracks:
            inf_line += ',AUDIO="audio"'
        if subtitles:
            inf_line += ',SUBTITLES="subs"'
            
        lines.append(inf_line)
        
        rel_str = str(rel_playlist).replace("\\", "/")
        lines.append(rel_str)

    master_content = "\n".join(lines) + "\n"
    master_path.write_text(master_content, encoding="utf-8")
    return master_path

# ------------------------------
# Orchestration
# ------------------------------
def process_film(
    film_name: str,
    inputs_by_res: Dict[int, Path],
    output_root: Path,
    segment_time: int,
    overwrite: bool,
    prefer_copy: bool,
    thumb_interval: int,
    thumb_width: int,
    max_variant_workers: int = 3,
    audio_mode: str = "separate",
    *,
    hw_encoder: Optional[str] = None,
    quality_mode: str = "preset",
    per_title: bool = False,
    explicit_bitrates: Optional[Dict[int, int]] = None,
) -> None:
    """Process a single film with all its variants in parallel."""
    logging.info("Processing film: %s with %d resolutions (audio: %s)", film_name, len(inputs_by_res), audio_mode)
    
    variant_results: List[VariantResult] = []

    # Compute per-title bitrate ladder if requested and not explicitly provided
    bitrates: Optional[Dict[int, int]] = explicit_bitrates
    if bitrates is None and per_title:
        try:
            # Use the highest resolution input as a proxy for analysis
            top_res = max(inputs_by_res.keys())
            source_for_analysis = inputs_by_res[top_res]
            bitrates = choose_bitrate_ladder(
                source_for_analysis,
                list(inputs_by_res.keys()),
                mode="per-title",
            )
        except Exception:
            bitrates = None
    
    # Process variants in parallel for faster processing
    # Note: ThreadPool is OK here because ffmpeg jobs are external processes (I/O bound).
    # If CPU becomes saturated, consider ProcessPool or a task queue executor.
    with ThreadPoolExecutor(max_workers=min(max_variant_workers, len(inputs_by_res))) as executor:
        futures = {
            executor.submit(
                generate_hls_variant_parallel_segments,  # Use the ultra-fast version
                film_name,
                res,
                input_path,
                output_root,
                segment_time,
                overwrite,
                prefer_copy,
                audio_mode,
                hw_encoder=hw_encoder,
                quality_mode=("per-title" if per_title else quality_mode),
                bitrate_kbps=(bitrates or {}).get(res),
            ): (res, input_path) for res, input_path in inputs_by_res.items()
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                variant_results.append(result)
                # Record per-variant timing is already logged by timer decorator; we can add metrics here if desired
                res, _ = futures[future]
                if result.success:
                    logging.info("✓ Completed %s %dp", film_name, res)
                else:
                    logging.error("✗ Failed %s %dp: %s", film_name, res, result.error_message)
            except Exception as e:
                res, _ = futures[future]
                logging.exception("Error processing %s %dp: %s", film_name, res, e)

    # Collect successful variants
    successful_variants: List[Tuple[int, Path]] = []
    highest_res_variant = None
    
    for r in variant_results:
        if r.success and r.playlist_path.exists():
            rel = r.playlist_path.relative_to(output_root / film_name)
            successful_variants.append((r.resolution, rel))
            if highest_res_variant is None or r.resolution > highest_res_variant.resolution:
                highest_res_variant = r

    if successful_variants:
        film_output_dir = output_root / film_name
        
        # Extract audio tracks and subtitles based on audio mode
        audio_tracks: Optional[List[Tuple[str, str, Path]]] = None
        subs: Optional[List[Tuple[str, str, Path]]] = None
        
        if highest_res_variant:
            if audio_mode == "separate":
                # Use ThreadPoolExecutor for parallel extraction
                with ThreadPoolExecutor(max_workers=2) as executor:
                    audio_future = executor.submit(extract_all_audio_tracks_to_hls, highest_res_variant.input_path, film_output_dir, segment_time, overwrite)
                    subs_future = executor.submit(extract_subtitles_to_vtt, highest_res_variant.input_path, film_output_dir, overwrite)
                    
                    audio_tracks = audio_future.result()
                    subs = subs_future.result()
            else:
                # For muxed mode, only extract subtitles
                subs = extract_subtitles_to_vtt(highest_res_variant.input_path, film_output_dir, overwrite)
            
        # Write master playlist with audio and subtitle references
        write_master_playlist(film_output_dir, successful_variants, subtitles=subs, audio_tracks=audio_tracks)
        
        # Generate single thumbnails.vtt for all qualities
        if highest_res_variant:
            generate_thumbnails_vtt(
                highest_res_variant.input_path, film_output_dir, 
                overwrite, interval=thumb_interval, width=thumb_width
            )
        
        audio_count = len(audio_tracks) if audio_tracks else 0
        subs_count = len(subs) if subs else 0
        if audio_mode == "separate":
            logging.info("✓ Successfully processed %s with %d variants, %d audio tracks, and %d subtitles", 
                        film_name, len(successful_variants), audio_count, subs_count)
            try:
                metrics.inc_films()
            except Exception:
                pass
        else:
            logging.info("✓ Successfully processed %s with %d variants (muxed audio) and %d subtitles", 
                        film_name, len(successful_variants), subs_count)
            try:
                metrics.inc_films()
            except Exception:
                pass
    else:
        logging.warning("✗ No successful variants for %s", film_name)

# ------------------------------
# CLI
# ------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast HLS generator with configurable audio handling using ffmpeg-python."
    )
    parser.add_argument("--source", "-s", default=str(Path.cwd() / "movies"),
                       help="Source directory containing movies")
    parser.add_argument("--dest", "-d", default=str(Path.cwd() / "hls_output"),
                       help="Destination directory for HLS output")
    parser.add_argument("--segment-time", "-t", type=int, default=6,
                       help="HLS segment duration in seconds (default: 6s)")
    parser.add_argument("--resolutions", "-r", type=str, default=",".join(map(str, DEFAULT_RESOLUTIONS)),
                       help="Comma-separated list of target resolutions")
    parser.add_argument("--workers", "-w", type=int, default=4,
                       help="Number of workers for film processing (threads or processes depending on executor)")
    parser.add_argument("--variant-workers", type=int, default=8,
                       help="Number of parallel workers per film for variant processing (max speed)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing files")
    parser.add_argument("--no-copy", action="store_true",
                       help="Always transcode, never copy streams")
    parser.add_argument("--audio-mode", choices=["separate", "muxed"], default="separate",
                       help="Audio handling: separate HLS streams or muxed with video")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--discover-only", action="store_true",
                       help="Only discover films, don't process")
    parser.add_argument("--thumb-interval", type=int, default=THUMB_INTERVAL,
                       help="Seconds between thumbnail images")
    parser.add_argument("--thumb-width", type=int, default=160,
                       help="Width of thumbnail images in pixels")
    # New scaling and quality options
    parser.add_argument("--executor", type=str, default="threadpool",
                       choices=["threadpool", "local-multiproc", "celery", "ray"],
                       help="Execution backend for film-level parallelism")
    parser.add_argument("--hw-encoder", type=str, default=None,
                       choices=["nvenc", "qsv", "amf", "auto"],
                       help="Hardware encoder to use when available")
    parser.add_argument("--max-encodes-per-gpu", type=int, default=2,
                       help="Maximum concurrent encode sessions per GPU (NVENC)")
    parser.add_argument("--quality-mode", type=str, default="preset",
                       choices=["preset", "crf", "abr", "per-title"],
                       help="Quality/bitrate control mode")
    parser.add_argument("--per-title", action="store_true",
                       help="Enable per-title bitrate ladder selection")
    parser.add_argument("--target-size", type=str, default=None,
                       help="Approximate target size per film, e.g. 2G, 1500M")
    parser.add_argument("--celery-broker", type=str, default=os.getenv("CELERY_BROKER_URL"),
                       help="Celery broker URL (for executor=celery)")
    parser.add_argument("--celery-backend", type=str, default=os.getenv("CELERY_BACKEND_URL"),
                       help="Celery backend URL (optional)")
    return parser.parse_args(argv)

def parse_resolution_list(res_str: str) -> List[int]:
    """Parse comma-separated resolution list."""
    values: List[int] = []
    for tok in re.split(r"[ ,]+", res_str.strip()):
        if not tok:
            continue
        try:
            values.append(int(tok))
        except ValueError:
            pass
    
    # Remove duplicates while preserving order
    seen = set()
    ordered: List[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered

# ------------------------------
# Main
# ------------------------------
@timer
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    source_dir = Path(args.source).expanduser().resolve()
    output_root = Path(args.dest).expanduser().resolve()
    ensure_dir(output_root)

    resolutions = parse_resolution_list(args.resolutions)
    if not resolutions:
        logging.error("No valid resolutions specified")
        return 2

    # Initialize GPU manager if requested
    global GPU_MGR
    hw_encoder = args.hw_encoder
    if hw_encoder == "auto":
        # Prefer NVENC if available
        if GPUManager is not None:
            mgr = GPUManager(max_sessions_per_gpu=max(1, int(getattr(args, 'max_encodes_per_gpu', 2))))
            GPU_MGR = mgr if mgr.available() else None
            hw_encoder = "nvenc" if GPU_MGR else None
        else:
            hw_encoder = None
    else:
        if GPUManager is not None and hw_encoder in ("nvenc", "qsv", "amf"):
            mgr = GPUManager(max_sessions_per_gpu=max(1, int(getattr(args, 'max_encodes_per_gpu', 2))))
            GPU_MGR = mgr if mgr.available() else None
        else:
            GPU_MGR = None

    # Discover films
    logging.info("Discovering films in %s...", source_dir)
    film_inputs = discover_films(source_dir, resolutions)
    if not film_inputs:
        logging.info("No films found in %s", source_dir)
        return 0

    logging.info("Found %d films to process", len(film_inputs))
    
    if args.discover_only:
        for film, by_res in sorted(film_inputs.items()):
            logging.info("Film: %s", film)
            for res, p in sorted(by_res.items(), reverse=True):
                logging.info("  %dp -> %s", res, p)
        return 0

    workers = max(1, args.workers)
    prefer_copy = not args.no_copy
    variant_workers = max(1, args.variant_workers)

    logging.info("Starting processing with %d film workers, %d variant workers per film, audio mode: %s", 
                workers, variant_workers, args.audio_mode)

    # Compute per-title bitrates if requested
    explicit_bitrates_by_film: Dict[str, Dict[int, int]] = {}
    if args.target_size:
        # Parse sizes like 2G, 1500M
        size_str = args.target_size.strip().upper()
        multiplier = 1
        if size_str.endswith('G'):
            multiplier = 1024 * 1024 * 1024
            size_val = float(size_str[:-1])
        elif size_str.endswith('M'):
            multiplier = 1024 * 1024
            size_val = float(size_str[:-1])
        elif size_str.endswith('K'):
            multiplier = 1024
            size_val = float(size_str[:-1])
        else:
            size_val = float(size_str)
        target_bytes = int(size_val * multiplier)
        for film_name, by_res in film_inputs.items():
            # best-effort duration from first input
            any_input = next(iter(by_res.values()))
            _, duration = get_video_info(any_input)
            duration = duration or 3600
            explicit_bitrates_by_film[film_name] = allocate_bitrates_for_target_size(duration, list(by_res.keys()), target_bytes)

    # Select execution backend
    executor_name = args.executor
    if executor_name == "threadpool":
        from concurrent.futures import ThreadPoolExecutor as TPE
        pool = TPE(max_workers=workers)
        futures = []
        for film_name, inputs_by_res in sorted(film_inputs.items()):
            futures.append(pool.submit(
                process_film,
                film_name,
                inputs_by_res,
                output_root,
                args.segment_time,
                args.overwrite,
                prefer_copy,
                args.thumb_interval,
                args.thumb_width,
                variant_workers,
                args.audio_mode,
                hw_encoder=hw_encoder,
                quality_mode=args.quality_mode,
                per_title=(args.per_title or args.quality_mode == 'per-title'),
                explicit_bitrates=explicit_bitrates_by_film.get(film_name),
            ))
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as ex:
                logging.exception("Worker error: %s", ex)
        pool.shutdown(wait=True)
    elif executor_name == "local-multiproc":
        try:
            from concurrent.futures import ProcessPoolExecutor as PPE
        except Exception:
            logging.error("ProcessPoolExecutor unavailable; falling back to threadpool")
            return main([*(argv or []), "--executor", "threadpool"])  # type: ignore
        pool = PPE(max_workers=workers)
        futures = []
        for film_name, inputs_by_res in sorted(film_inputs.items()):
            futures.append(pool.submit(
                process_film,
                film_name,
                inputs_by_res,
                output_root,
                args.segment_time,
                args.overwrite,
                prefer_copy,
                args.thumb_interval,
                args.thumb_width,
                variant_workers,
                args.audio_mode,
                hw_encoder=hw_encoder,
                quality_mode=args.quality_mode,
                per_title=(args.per_title or args.quality_mode == 'per-title'),
                explicit_bitrates=explicit_bitrates_by_film.get(film_name),
            ))
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as ex:
                    logging.exception("Worker error: %s", ex)
        pool.shutdown(wait=True)
    elif executor_name == "celery":
        if make_executor is None:
            logging.error("Celery executor not available. Install celery and redis-py.")
            return 2
        execu = make_executor("celery", max_workers=workers, broker_url=args.celery_broker, backend_url=args.celery_backend)  # type: ignore
        # Enqueue tasks: assume workers register 'hls.process_film_task'
        tasks = []
        for film_name, inputs_by_res in sorted(film_inputs.items()):
            payload = {
                "film_name": film_name,
                "inputs_by_res": {int(k): str(v) for k, v in inputs_by_res.items()},
                "output_root": str(output_root),
                "segment_time": args.segment_time,
                "overwrite": args.overwrite,
                "prefer_copy": prefer_copy,
                "thumb_interval": args.thumb_interval,
                "thumb_width": args.thumb_width,
                "variant_workers": variant_workers,
                "audio_mode": args.audio_mode,
                "hw_encoder": hw_encoder,
                "quality_mode": args.quality_mode,
                "per_title": (args.per_title or args.quality_mode == 'per-title'),
                "explicit_bitrates": explicit_bitrates_by_film.get(film_name),
            }
            tasks.append(execu.submit("hls.process_film_task", payload))
        for t in tasks:
            execu.result(t)
    elif executor_name == "ray":
        if make_executor is None:
            logging.error("Ray executor not available. Install ray.")
            return 2
        execu = make_executor("ray", max_workers=workers)  # type: ignore
        futs = []
        for film_name, inputs_by_res in sorted(film_inputs.items()):
            futs.append(execu.submit(
                process_film,
                film_name,
                inputs_by_res,
                output_root,
                args.segment_time,
                args.overwrite,
                prefer_copy,
                args.thumb_interval,
                args.thumb_width,
                variant_workers,
                args.audio_mode,
                hw_encoder=hw_encoder,
                quality_mode=args.quality_mode,
                per_title=(args.per_title or args.quality_mode == 'per-title'),
                explicit_bitrates=explicit_bitrates_by_film.get(film_name),
            ))
        for f in futs:
            execu.result(f)
    else:
        logging.error("Unknown executor: %s", executor_name)
        return 2

    logging.info("✓ Processing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 