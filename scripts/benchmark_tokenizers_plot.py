#!/usr/bin/env python3
"""
Benchmark default vs Vulkan tokenizer across growing prompt sizes and plot with matplotlib.

Run with the venv Python to ensure matplotlib is available:
  . .venv/bin/activate
  python scripts/benchmark_tokenizers_plot.py --runs 3 --warmup 1
"""

import argparse
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


BIN_DEFAULT = Path("build/bin/llama-tokenize")
MODEL_DEFAULT = Path("/var/lib/ollama/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29")


def load_corpus(min_chars: int = 200_000) -> str:
    """Collect real repo text to serve as prompt material."""
    candidates = [
        Path("README.md"),
        Path("docs"),
        Path("examples"),
        Path("benches"),
    ]
    pieces: List[str] = []
    for path in candidates:
        if path.is_file():
            pieces.append(path.read_text(encoding="utf-8", errors="ignore"))
        elif path.is_dir():
            for file in path.rglob("*"):
                if file.suffix.lower() in {".md", ".txt"} and file.is_file():
                    try:
                        pieces.append(file.read_text(encoding="utf-8", errors="ignore"))
                    except Exception:
                        continue
    corpus = "\n\n".join(pieces)
    if not corpus:
        corpus = "Llama tokenizer benchmark prompt. " * 10
    while len(corpus) < min_chars:
        corpus += "\n\n" + corpus
    return corpus


def make_prompt(corpus: str, target_chars: int) -> str:
    prompt = (corpus * (target_chars // len(corpus) + 1))[:target_chars]
    cut = prompt.rfind(" ", 0, target_chars)
    return prompt[:cut] if cut > target_chars * 0.8 else prompt


def tokenize_once(
    binary: Path,
    model: Path,
    prompt: str,
    env_extra: Dict[str, str],
) -> Tuple[float, int]:
    env = os.environ.copy()
    env.update(env_extra)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        tmp.write(prompt)
        tmp_path = tmp.name

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
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    elapsed = time.perf_counter() - start
    merged_out = (completed.stdout or "") + "\n" + (completed.stderr or "")
    m = re.search(r"Total number of tokens:\s*(\d+)", merged_out)
    tokens = int(m.group(1)) if m else -1
    return elapsed, tokens


def collect_series(
    binary: Path,
    model: Path,
    corpus: str,
    target_char_lengths: Iterable[int],
    runs: int,
    warmup: int,
) -> List[Dict[str, float]]:
    rows = []
    for target in target_char_lengths:
        prompt = make_prompt(corpus, target)
        for _ in range(warmup):
            tokenize_once(binary, model, prompt, env_extra={})
            tokenize_once(binary, model, prompt, env_extra={"LLAMA_BPE_VK": "1"})

        times_def: List[float] = []
        times_vk: List[float] = []
        tokens_ref = None
        for _ in range(runs):
            t_def, tokens = tokenize_once(binary, model, prompt, env_extra={})
            t_vk, tokens_vk = tokenize_once(binary, model, prompt, env_extra={"LLAMA_BPE_VK": "1"})
            if tokens_ref is None:
                tokens_ref = tokens
            if tokens != tokens_vk or tokens_ref != tokens:
                raise RuntimeError(f"Token count mismatch at chars={target}: {tokens} vs {tokens_vk} (ref {tokens_ref})")
            times_def.append(t_def)
            times_vk.append(t_vk)

        rows.append(
            {
                "target_chars": target,
                "tokens": tokens_ref or -1,
                "t_default": mean(times_def),
                "t_vk": mean(times_vk),
            }
        )
        print(
            f"chars ~{target:>7}, tokens {tokens_ref:>7} | "
            f"default avg {mean(times_def):.3f}s | vulkan avg {mean(times_vk):.3f}s "
            f"(runs={runs}, warmup={warmup})"
        )
    return rows


def plot(rows: List[Dict[str, float]], out_path: Path) -> None:
    tokens = [r["tokens"] for r in rows]
    t_def = [r["t_default"] for r in rows]
    t_vk = [r["t_vk"] for r in rows]

    plt.figure(figsize=(9, 5))
    plt.scatter(tokens, t_def, color="#1f77b4", label="default")
    plt.scatter(tokens, t_vk, color="#d62728", label="vulkan (LLAMA_BPE_VK=1)")
    plt.plot(tokens, t_def, color="#1f77b4", linestyle="--", alpha=0.6)
    plt.plot(tokens, t_vk, color="#d62728", linestyle="--", alpha=0.6)
    plt.xlabel("tokens (prompt)")
    plt.ylabel("wall time (s)")
    plt.title("Tokenizer performance: default vs Vulkan")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark tokenizer default vs Vulkan and plot with matplotlib.")
    parser.add_argument("--binary", default=str(BIN_DEFAULT), help="Path to llama-tokenize binary.")
    parser.add_argument("--model", default=str(MODEL_DEFAULT), help="Path to model GGUF file.")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per point.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per point (not measured).")
    parser.add_argument(
        "--targets",
        type=str,
        default="1000,4000,8000,20000,40000,70000,100000",
        help="Comma-separated target token counts (approx; chars use x4).",
    )
    parser.add_argument("--output", default="benchmarks/tokenizer_vulkan_vs_default.png", help="Output plot path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binary = Path(args.binary)
    model = Path(args.model)
    if not binary.exists():
        raise SystemExit(f"Missing tokenizer binary at {binary}")
    if not model.exists():
        raise SystemExit(f"Missing model at {model}")

    targets_tokens = [int(x) for x in args.targets.split(",")]
    target_chars = [t * 4 for t in targets_tokens]
    corpus = load_corpus()
    print("Collecting timings (default vs LLAMA_BPE_VK) ...")
    rows = collect_series(binary, model, corpus, target_chars, runs=args.runs, warmup=args.warmup)
    plot(rows, Path(args.output))

    print("\nSummary (tokens ~ avg time seconds):")
    for r in rows:
        print(
            f"{r['tokens']:>8} tokens | default {r['t_default']:.3f}s | "
            f"vulkan {r['t_vk']:.3f}s | chars target {r['target_chars']}"
        )


if __name__ == "__main__":
    main()
