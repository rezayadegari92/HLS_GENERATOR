<<<<<<< HEAD
# HLS Generator

Generate adaptive HLS (HTTP Live Streaming) outputs from a movie library. The tool scans your movies folder, produces per-resolution HLS playlists and segments, and builds a master playlist for adaptive streaming.

### Variants
- `hls_generator_v2.py` (recommended): uses `ffmpeg-python`, supports parallel variant processing, optional separate audio HLS tracks, WebVTT subtitles extraction, and thumbnails.
- `hls_generator.py` (legacy): standard-library only; relies on shelling out to ffmpeg/ffprobe.

#### v2 scaling additions
- Execution backends: `--executor` supports `threadpool`, `local-multiproc`, `celery`, `ray`.
- GPU encoding: `--hw-encoder {auto,nvenc,qsv,amf}` with safe fallback to CPU.
- Quality control: `--quality-mode {preset,crf,abr,per-title}`, `--per-title`, and `--target-size`.
- Deterministic audio naming for separate renditions (e.g., `audio/en-00`, `audio/fa-01`).
- Celery PoC with sample `deploy/docker-compose.celery.yml` and `deploy/k8s-worker.yaml`.
- Metrics (Prometheus) via `hls/monitoring.py` (opt-in).

## Requirements
- Python 3.8+
- FFmpeg and FFprobe installed and on PATH
  - Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y ffmpeg`
  - macOS (Homebrew): `brew install ffmpeg`
  - Windows (winget): `winget install Gyan.FFmpeg.Full`
  - Windows (Chocolatey): `choco install ffmpeg`
- Python packages (for v2): install via `requirements.txt`

### Dependency notes (global vs pip)
- System/global: Requires `ffmpeg` and `ffprobe` binaries available on PATH. Install via your OS package manager (see above).
- Python/pip: v2 uses `ffmpeg-python==0.2.0` (listed in `requirements.txt`). v1 uses only the Python standard library.
- Contributing: If you add dependencies, list pip packages in `requirements.txt` and document any new system binaries here.

### Verify installation
```bash
ffmpeg -version
ffprobe -version
python -c "import ffmpeg; print('ffmpeg-python OK')"
```

## Install
```bash
# From project root
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you only plan to use the legacy script `hls_generator.py`, Python packages are not required; you still need system FFmpeg/FFprobe.

## Input folder structure
The tool supports both styles; you can mix them:

1) Film with resolution subfolders (preferred):
```
movies/
  film1/
    1080/film1.mkv
    720/film1.mp4
    480/film1.mkv
```
Accepted resolution folder names: `1080`, `1080p`, `720`, `720p` (case-insensitive `p`).

2) Film with media file directly inside the film folder:
```
movies/
  film2/
    Film2.mkv
```
Video height is probed and mapped to the closest configured resolution (e.g., 480, 720, 1080).

## Output folder structure
For each film:
```
hls_output/
  {film_name}/
    master.m3u8
    1080/
      index.m3u8
      segment_00001.ts
    720/
      index.m3u8
    480/
      index.m3u8
    subtitles/        # when extracted (v2)
    audio/            # when audio-mode=separate (v2)
    thumbnails/       # thumbnails and thumbnails.vtt (v2)
```
The master playlist references all successfully generated variants and optional renditions.

## Quick start (recommended v2)
```bash
source .venv/bin/activate
python3 hls_generator_v2.py \
  --source /path/to/movies \
  --dest /path/to/hls_output \
  --workers 2 \
  --variant-workers 3 \
  --overwrite \
  --audio-mode separate \
  --executor local-multiproc \
  --hw-encoder auto \
  --quality-mode per-title \
  --log-level INFO
```

### v2 Command-line options
- `-s, --source`: Source movies directory (default: `./movies`)
- `-d, --dest`: Destination output directory (default: `./hls_output`)
- `-t, --segment-time`: HLS segment duration (default: `6`)
- `-r, --resolutions`: Comma-separated list (default: `1080,720,480`)
- `-w, --workers`: Number of films to process concurrently (default: `2`)
- `--variant-workers`: Parallel workers per film for variants (default: `3`)
- `--overwrite`: Overwrite existing playlists/segments
- `--no-copy`: Always transcode; otherwise try stream copy first
- `--audio-mode`: `separate` (audio-only HLS renditions) or `muxed` (first audio track included in video)
- `--thumb-interval`: Seconds between thumbnails (default: `20`)
- `--thumb-width`: Width of thumbnails in pixels (default: `160`)
- `--discover-only`: Only scan and print what would be processed
- `--log-level`: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)
 - `--executor`: `threadpool`, `local-multiproc`, `celery`, `ray`
 - `--hw-encoder`: `auto`, `nvenc`, `qsv`, `amf` (falls back to CPU)
 - `--quality-mode`: `preset`, `crf`, `abr`, `per-title`
 - `--per-title`: Enable per-title ladder selection
 - `--target-size`: e.g., `2G`, `1500M` to budget bitrates across variants
 - `--celery-broker`, `--celery-backend`: Celery connection parameters

## Legacy usage (v1)
```bash
python3 hls_generator.py \
  -s /path/to/movies \
  -d /path/to/hls_output \
  -w 4 \
  --overwrite \
  --log-level INFO
```

### v1 Command-line options
- `-s, --source`: Source movies directory (default: `./movies`)
- `-d, --dest`: Destination output directory (default: `./hls_output`)
- `-t, --segment-time`: HLS segment duration (default: `6`)
- `-r, --resolutions`: Comma-separated list (default: `1080,720,480`)
- `-w, --workers`: Number of films to process concurrently (default: `1`)
- `--ffmpeg`: Path to ffmpeg binary (default: `ffmpeg`)
- `--ffprobe`: Path to ffprobe binary (default: `ffprobe`)
- `--overwrite`: Overwrite existing playlists/segments
- `--no-copy`: Always transcode to H.264/AAC instead of attempting stream copy
- `--discover-only`: Only list discovered films and resolutions; don’t run ffmpeg
- `--log-level`: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)

## Notes on encoding
- HLS VOD: `-hls_time 6 -hls_playlist_type vod`
- Variant generation tries `copy` first where applicable, then falls back to H.264/AAC
- Master playlist uses reasonable bandwidth presets and resolution tags for 1080p/720p/480p

## Troubleshooting
- ffmpeg/ffprobe not found: install FFmpeg and ensure it is on PATH
- No films discovered: check folder structure and `--source` path; try `--discover-only`
- Permission denied: ensure write access to `--dest`
- Slow encoding: increase `--workers` or `--variant-workers`, or prefer copy when inputs are already H.264/AAC

## Example (one-liner)
```bash
source .venv/bin/activate && \
python3 hls_generator_v2.py \
  --source /home/red/Desktop/movies/ \
  --dest /home/red/Desktop/hls_output_v2 \
  --overwrite \
  --workers 2 \
  --variant-workers 3 \
  --thumb-interval 20
```
=======
# HLS_GENERATOR
>>>>>>> 95ac836a93f85b8c1fd34774dfdebb3960f96943
