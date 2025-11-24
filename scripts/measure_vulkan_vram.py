#!/usr/bin/env python3
"""
Measure VRAM usage of the Vulkan tokenizer by polling DRM mem_info counters while it runs.

Usage:
  python scripts/measure_vulkan_vram.py --model /path/to/model.gguf --chars 400000

Notes:
- Relies on /sys/class/drm/card*/device/mem_info_vram_used (AMD/ROCm drivers).
- Uses LLAMA_BPE_VK=1 to force Vulkan tokenizer path.
"""

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List


def find_vram_file() -> Path:
    candidates: List[Path] = sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"))
    if not candidates:
        raise SystemExit("Could not locate mem_info_vram_used under /sys/class/drm")
    return candidates[0]


def read_vram_bytes(path: Path) -> int:
    return int(path.read_text().strip())


def build_prompt(chars: int) -> str:
    sources = [Path("README.md"), Path("docs")]
    text = ""
    for src in sources:
        if src.is_file():
            text += src.read_text(encoding="utf-8", errors="ignore") + "\n\n"
        elif src.is_dir():
            for f in src.rglob("*.md"):
                try:
                    text += f.read_text(encoding="utf-8", errors="ignore") + "\n\n"
                except Exception:
                    continue
    if not text:
        text = "Llama tokenizer VRAM measurement prompt. " * 100
    while len(text) < chars:
        text += text
    return text[:chars]


def measure(binary: Path, model: Path, prompt: str, vram_file: Path) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        tmp.write(prompt)
        tmp_path = tmp.name
    env = os.environ.copy()
    env["LLAMA_BPE_VK"] = "1"
    cmd = [
        str(binary),
        "--log-disable",
        "--ids",
        "--show-count",
        "-m",
        str(model),
        "-f",
        tmp_path,
    ]
    baseline = read_vram_bytes(vram_file)
    max_vram = baseline
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        while proc.poll() is None:
            time.sleep(0.05)
            max_vram = max(max_vram, read_vram_bytes(vram_file))
        _, err = proc.communicate()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    end_vram = read_vram_bytes(vram_file)
    delta = max_vram - baseline
    print(f"Baseline VRAM: {baseline} bytes")
    print(f"Peak VRAM while tokenizing: {max_vram} bytes")
    print(f"Delta: {delta} bytes")
    if proc.returncode != 0:
        print("Process stderr:\n", err)
        raise SystemExit(proc.returncode)
    else:
        print("llama-tokenize completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure VRAM usage of Vulkan tokenizer.")
    parser.add_argument("--binary", default="build/bin/llama-tokenize", help="Path to llama-tokenize binary.")
    parser.add_argument("--model", required=True, help="Path to model GGUF file.")
    parser.add_argument("--chars", type=int, default=400000, help="Prompt size in characters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vram_file = find_vram_file()
    prompt = build_prompt(args.chars)
    measure(Path(args.binary), Path(args.model), prompt, vram_file)


if __name__ == "__main__":
    main()
