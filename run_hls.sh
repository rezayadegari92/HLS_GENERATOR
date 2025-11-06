#!/usr/bin/env bash
set -euo pipefail

# Simple runner for hls_generator_v2.py with preflight checks and post-run verification

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
GEN="${SCRIPT_DIR}/hls_generator_v2.py"

# Defaults (override via env or flags)
SRC="${SRC:-$HOME/movies}"
DEST="${DEST:-$HOME/hls_output}"
SEG_TIME="${SEG_TIME:-6}"
WORKERS="${WORKERS:-4}"
VARIANT_WORKERS="${VARIANT_WORKERS:-8}"
AUDIO_MODE="${AUDIO_MODE:-separate}"
EXECUTOR="${EXECUTOR:-threadpool}"
HW_ENCODER="${HW_ENCODER:-auto}"
RESOLUTIONS="${RESOLUTIONS:-1080,720,480}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--source PATH] [--dest PATH] [--segment-time N] [--workers N] [--variant-workers N]
                         [--audio-mode separate|muxed] [--executor threadpool|local-multiproc] [--hw-encoder auto|nvenc|qsv|amf|none]
                         [--resolutions "1080,720,480"] [--discover-only]

Env overrides: SRC, DEST, SEG_TIME, WORKERS, VARIANT_WORKERS, AUDIO_MODE, EXECUTOR, HW_ENCODER, RESOLUTIONS
Examples:
  $(basename "$0") --source /home/reza/movies --dest /home/reza/hls_output
  SEG_TIME=6 VARIANT_WORKERS=6 $(basename "$0") --source /data/movies --dest /data/hls
EOF
}

DISCOVER_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--source) SRC="$2"; shift 2;;
    -d|--dest) DEST="$2"; shift 2;;
    -t|--segment-time) SEG_TIME="$2"; shift 2;;
    -w|--workers) WORKERS="$2"; shift 2;;
    --variant-workers) VARIANT_WORKERS="$2"; shift 2;;
    --audio-mode) AUDIO_MODE="$2"; shift 2;;
    --executor) EXECUTOR="$2"; shift 2;;
    --hw-encoder) HW_ENCODER="$2"; shift 2;;
    -r|--resolutions) RESOLUTIONS="$2"; shift 2;;
    --discover-only) DISCOVER_ONLY=true; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Error: python or python3 not found in PATH" >&2
  exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found. Install it (e.g., sudo apt-get install -y ffmpeg)." >&2
  exit 2
fi
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "Error: ffprobe not found. Install it (e.g., sudo apt-get install -y ffmpeg)." >&2
  exit 2
fi

if [[ ! -f "${GEN}" ]]; then
  echo "Error: generator not found at ${GEN}" >&2
  exit 2
fi

if [[ ! -d "${SRC}" ]]; then
  echo "Error: source directory does not exist: ${SRC}" >&2
  echo "Create it and place videos as either:"
  echo "  - ${SRC}/FilmName/MovieFile.mkv"
  echo "  - ${SRC}/FilmName/1080p/MovieFile.mp4 (or 720/480)"
  exit 3
fi

mkdir -p "${DEST}"

# Quick content check
mapfile -t SAMPLE_VIDS < <(find "${SRC}" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.mov' \) | head -n 5)
if [[ ${#SAMPLE_VIDS[@]} -eq 0 ]]; then
  echo "No supported video files found under ${SRC}. Supported: mp4, mkv, avi, mov" >&2
  echo "Expected layouts:"
  echo "  - ${SRC}/FilmName/MovieFile.mkv"
  echo "  - ${SRC}/FilmName/1080p/MovieFile.mp4"
  exit 3
fi

echo "Found sample videos:" >&2
printf '  %s\n' "${SAMPLE_VIDS[@]}" >&2

echo "Running discovery..." >&2
"${PYTHON_BIN}" "${GEN}" \
  --source "${SRC}" \
  --dest "${DEST}" \
  --resolutions "${RESOLUTIONS}" \
  --discover-only || true

if [[ "${DISCOVER_ONLY}" == true ]]; then
  echo "Discovery-only mode complete." >&2
  exit 0
fi

# Decide hw-encoder argument
HW_ARG=()
case "${HW_ENCODER}" in
  auto|nvenc|qsv|amf) HW_ARG=(--hw-encoder "${HW_ENCODER}");;
  none|"") HW_ARG=();;
  *) HW_ARG=(--hw-encoder auto);;
esac

echo "Starting HLS generation..." >&2
"${PYTHON_BIN}" "${GEN}" \
  --source "${SRC}" \
  --dest "${DEST}" \
  --segment-time "${SEG_TIME}" \
  --workers "${WORKERS}" \
  --variant-workers "${VARIANT_WORKERS}" \
  --audio-mode "${AUDIO_MODE}" \
  --executor "${EXECUTOR}" \
  --resolutions "${RESOLUTIONS}" \
  --overwrite \
  "${HW_ARG[@]}"

echo "Generation finished. Verifying one film's variants if present..." >&2

# Pick first film dir under DEST
FILM_DIR="$(find "${DEST}" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
if [[ -z "${FILM_DIR}" ]]; then
  echo "No film directories found under ${DEST}." >&2
  exit 0
fi

probe_variant() {
  local film_dir="$1"; shift
  local res="$1"; shift
  local vdir="${film_dir}/${res}"
  if [[ ! -d "${vdir}" ]]; then
    echo "[${res}] directory missing: ${vdir}" >&2
    return 0
  fi
  local seg
  seg="$(ls -1 "${vdir}"/segment_0000*.ts 2>/dev/null | sort | head -n 1 || true)"
  if [[ -z "${seg}" ]]; then
    seg="$(ls -1 "${vdir}"/segment_*.ts 2>/dev/null | sort | head -n 1 || true)"
  fi
  if [[ -z "${seg}" ]]; then
    echo "[${res}] no segments found in ${vdir}" >&2
    return 0
  fi
  local wh
  wh="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "${seg}" || true)"
  echo "[${res}] ${seg##*/}: ${wh}" >&2
}

echo "Film: ${FILM_DIR}" >&2
probe_variant "${FILM_DIR}" 1080 || true
probe_variant "${FILM_DIR}" 720 || true
probe_variant "${FILM_DIR}" 480 || true

echo "Done." >&2


