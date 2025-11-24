#!/usr/bin/env python3
"""
Small helper to compare the Vulkan tokenizer (LLAMA_BPE_VK=1) against the default tokenizer.

It shells out to llama-tokenize for each variant, captures timings, and checks that both
tokenizers return identical token ID sequences.
"""

import argparse
import os
import statistics
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_prompt(prompt: str, prompt_file: Optional[str], repeat: int) -> str:
    if prompt_file:
        content = Path(prompt_file).read_text(encoding="utf-8")
    else:
        content = prompt
    # Repeat the content to amplify tokenization cost if requested.
    repeated = " ".join([content] * repeat).strip()
    return repeated if repeated else content


def _check_paths(binary_path: Path, model_path: Path) -> None:
    if not binary_path.is_file():
        raise FileNotFoundError(f"llama-tokenize binary not found: {binary_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")


def _run_once(
    binary_path: Path,
    model_path: Path,
    prompt: str,
    extra_env: Dict[str, str],
) -> Tuple[float, str]:
    env = os.environ.copy()
    env.update(extra_env)
    cmd = [
        str(binary_path),
        "--log-disable",
        "--ids",
        "--show-count",
        "-m",
        str(model_path),
        "-p",
        prompt,
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.perf_counter() - start
    return elapsed, completed.stdout.strip()


def benchmark_variant(
    name: str,
    binary_path: Path,
    model_path: Path,
    prompt: str,
    runs: int,
    warmup: int,
    extra_env: Dict[str, str],
) -> Tuple[List[float], str]:
    for _ in range(warmup):
        _run_once(binary_path, model_path, prompt, extra_env)
    times: List[float] = []
    tokens: Optional[str] = None
    for _ in range(runs):
        elapsed, out = _run_once(binary_path, model_path, prompt, extra_env)
        if tokens is None:
            tokens = out
        elif out != tokens:
            raise RuntimeError(
                f"Token mismatch detected while benchmarking {name}. "
                "Ensure both tokenizers produce identical outputs."
            )
        times.append(elapsed)
    assert tokens is not None
    return times, tokens


def summarize_times(times: List[float]) -> str:
    return (
        f"avg {statistics.mean(times):.3f}s | "
        f"min {min(times):.3f}s | "
        f"max {max(times):.3f}s over {len(times)} runs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Vulkan tokenizer (LLAMA_BPE_VK) vs default.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example:
              python scripts/benchmark_tokenizers.py \\
                --model /var/lib/ollama/blobs/sha256-... \\
                --prompt "hello world" --repeat 100 --runs 3 --warmup 1
            """
        ),
    )
    parser.add_argument(
        "--binary",
        default="build/bin/llama-tokenize",
        help="Path to llama-tokenize binary (default: build/bin/llama-tokenize).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model GGUF file to load.",
    )
    parser.add_argument(
        "--prompt",
        default="hello world",
        help="Prompt text. Ignored if --prompt-file is provided.",
    )
    parser.add_argument(
        "--prompt-file",
        help="Optional file containing the prompt.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the prompt N times to increase tokenization work (default: 1).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per variant (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unmeasured warmup runs per variant (default: 1).",
    )
    parser.add_argument(
        "--vk-env",
        default="LLAMA_BPE_VK",
        help="Environment variable that enables the Vulkan tokenizer (default: LLAMA_BPE_VK).",
    )
    args = parser.parse_args()

    binary_path = Path(args.binary).expanduser()
    model_path = Path(args.model).expanduser()
    _check_paths(binary_path, model_path)

    prompt = _read_prompt(args.prompt, args.prompt_file, args.repeat)
    print(f"Binary: {binary_path}")
    print(f"Model:  {model_path}")
    print(f"Prompt length: {len(prompt)} chars (repeat={args.repeat})")
    print(f"Runs: {args.runs} | Warmup: {args.warmup}")
    print()

    default_times, sample_tokens = benchmark_variant(
        name="default",
        binary_path=binary_path,
        model_path=model_path,
        prompt=prompt,
        runs=args.runs,
        warmup=args.warmup,
        extra_env={},
    )
    vk_times, vk_tokens = benchmark_variant(
        name="vulkan",
        binary_path=binary_path,
        model_path=model_path,
        prompt=prompt,
        runs=args.runs,
        warmup=args.warmup,
        extra_env={args.vk_env: "1"},
    )

    print("Timings:")
    print(f"- default: {summarize_times(default_times)}")
    print(f"- vulkan ({args.vk_env}=1): {summarize_times(vk_times)}")
    print()
    print("Sample tokens (should match across variants):")
    print(sample_tokens)


if __name__ == "__main__":
    main()
