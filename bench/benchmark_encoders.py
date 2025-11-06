#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path


def run_ffmpeg(cmd):
    start = time.time()
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = True
    except subprocess.CalledProcessError:
        ok = False
    duration = time.time() - start
    return ok, duration


def main():
    p = argparse.ArgumentParser(description="Benchmark CPU vs GPU encoders on a short sample.")
    p.add_argument("input", type=Path, help="Input media file")
    p.add_argument("--seconds", type=int, default=30, help="Duration to encode for")
    p.add_argument("--resolution", type=int, default=720, help="Target height")
    p.add_argument("--output-dir", type=Path, default=Path("./bench_out"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # CPU libx264 CRF 23
    cpu_out = args.output_dir / "cpu_x264.mp4"
    cmd_cpu = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(args.input),
        "-t", str(args.seconds),
        "-vf", f"scale=-2:{args.resolution}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(cpu_out),
    ]
    ok, sec = run_ffmpeg(cmd_cpu)
    results["cpu_x264"] = {"ok": ok, "seconds": sec, "bytes": cpu_out.stat().st_size if cpu_out.exists() else 0}

    # NVENC h264 if available
    nvenc_out = args.output_dir / "nvenc_h264.mp4"
    cmd_nvenc = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(args.input),
        "-t", str(args.seconds),
        "-vf", f"scale=-2:{args.resolution}",
        "-c:v", "h264_nvenc", "-preset", "p5", "-b:v", "3000k",
        "-c:a", "aac", "-b:a", "128k",
        str(nvenc_out),
    ]
    ok, sec = run_ffmpeg(cmd_nvenc)
    results["nvenc_h264"] = {"ok": ok, "seconds": sec, "bytes": nvenc_out.stat().st_size if nvenc_out.exists() else 0}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
