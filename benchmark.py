#!/usr/bin/env python3
"""
Benchmark script for transcribe.py server.
Tests both diarization and plain transcription on a selection of WAV files.

Usage:
    # Start the server first:
    python transcribe.py

    # Then in another terminal:
    python benchmark.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

PORT = int(os.getenv("TRANSCRIBE_PORT", "8765"))
URL  = f"http://127.0.0.1:{PORT}/transcribe"

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")

# A handful of files spanning different sizes (small → large)
FILES = [
    "hqyok.wav",   # ~688K
    "szsyz.wav",   # ~1.6M
    "rtvuw.wav",   # ~1.8M
    "oklol.wav",   # ~30M
    "ldnro.wav",   # ~34M
]


def transcribe(filepath: str, diarize: bool) -> tuple[float, str]:
    payload = json.dumps({"filename": filepath, "diarize": diarize}).encode()
    req = urllib.request.Request(
        URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - t0
    return elapsed, result.get("text", "")


def fmt_size(path: str) -> str:
    mb = os.path.getsize(path) / 1_048_576
    return f"{mb:.1f}MB"


def main(files=None):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)

    fpaths = files if files is not None else [os.path.join(BENCHMARK_DIR, f) for f in FILES]

    print(f"Benchmarking against {URL}\n")
    print(f"{'File':<12} {'Size':>6}  {'Mode':<14} {'Time':>7}  Preview")
    print("-" * 80)

    for fpath in fpaths:
        fname = os.path.basename(fpath)
        fpath = os.path.join(BENCHMARK_DIR, fname)
        if not os.path.exists(fpath):
            print(f"{fname:<12}  NOT FOUND — skipping")
            continue

        size = fmt_size(fpath)


        for diarize in [False, True]:
            mode = "diarization" if diarize else "plain"
            try:
                elapsed, text = transcribe(fpath, diarize)
                preview = text[:60].replace("\n", " ")
                if len(text) > 60:
                    preview += "…"
                print(f"{fname:<12} {size:>6}  {mode:<14} {elapsed:>6.2f}s  {preview}")

                stem = os.path.splitext(fname)[0]
                out_path = os.path.join(out_dir, f"{run_id}_{stem}_{mode}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)

            except urllib.error.URLError as e:
                print(f"{fname:<12} {size:>6}  {mode:<14}  ERROR: {e.reason}")
            except Exception as e:
                print(f"{fname:<12} {size:>6}  {mode:<14}  ERROR: {e}")

        print()

    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single file mode: resolve relative to benchmark dir if not absolute
        arg = sys.argv[1]
        if not os.path.isabs(arg):
            arg = os.path.join(BENCHMARK_DIR, arg)
        main(files=[arg])
    else:
        main()
