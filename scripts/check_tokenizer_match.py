#!/usr/bin/env python3
"""
Verify that Vulkan tokenizer output matches the default tokenizer exactly.

Usage (after building llama-tokenize):
  python scripts/check_tokenizer_match.py --model /path/to/model.gguf

It builds several real-text prompts from repo content (scales up to ~100k tokens),
tokenizes with and without LLAMA_BPE_VK=1, and fails if any mismatch is found.
"""

import argparse
import ast
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


BIN_DEFAULT = Path("build/bin/llama-tokenize")
MODEL_DEFAULT = Path("/var/lib/ollama/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29")


def load_corpus(min_chars: int = 200_000) -> str:
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
        corpus = "Llama tokenizer consistency check prompt. " * 10
    while len(corpus) < min_chars:
        corpus += "\n\n" + corpus
    return corpus


def make_prompt(corpus: str, target_chars: int) -> str:
    prompt = (corpus * (target_chars // len(corpus) + 1))[:target_chars]
    cut = prompt.rfind(" ", 0, target_chars)
    return prompt[:cut] if cut > target_chars * 0.8 else prompt


def run_tokenize(binary: Path, model: Path, prompt: str, env_extra: Dict[str, str]) -> Tuple[List[int], int]:
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
    # stdout format: "[1, 2]\nTotal number of tokens: X\n"
    lines = (completed.stdout or "").strip().splitlines()
    if not lines:
        raise RuntimeError("No stdout from llama-tokenize")
    tokens = ast.literal_eval(lines[0])
    total = None
    for line in lines[1:]:
        if line.lower().startswith("total number of tokens"):
            try:
                total = int(line.split(":")[1].strip())
            except Exception:
                pass
    if total is None:
        total = len(tokens)
    return tokens, total


def check_matches(
    binary: Path,
    model: Path,
    corpus: str,
    target_char_lengths: Iterable[int],
) -> None:
    failures = []
    for target in target_char_lengths:
        prompt = make_prompt(corpus, target)
        tokens_def, total_def = run_tokenize(binary, model, prompt, env_extra={})
        tokens_vk, total_vk = run_tokenize(binary, model, prompt, env_extra={"LLAMA_BPE_VK": "1"})
        match = tokens_def == tokens_vk and total_def == total_vk
        status = "OK" if match else "MISMATCH"
        print(
            f"chars ~{target:>8} | tokens default={total_def:>7} vulkan={total_vk:>7} | {status}"
        )
        if not match:
            failures.append(
                {
                    "target": target,
                    "default": total_def,
                    "vulkan": total_vk,
                    "first_diff": next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(tokens_def, tokens_vk))
                            if a != b
                        ),
                        None,
                    ),
                }
            )
    if failures:
        lines = ["\nFound mismatches:"]
        for f in failures:
            lines.append(
                f"- chars~{f['target']}: default={f['default']} vulkan={f['vulkan']} first_diff_index={f['first_diff']}"
            )
        raise SystemExit("\n".join(lines))
    print("\nAll prompts matched between default and Vulkan tokenizers.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that Vulkan tokenizer matches default output exactly.")
    parser.add_argument("--binary", default=str(BIN_DEFAULT), help="Path to llama-tokenize binary.")
    parser.add_argument("--model", default=str(MODEL_DEFAULT), help="Path to model GGUF file.")
    parser.add_argument(
        "--targets",
        type=str,
        default="10,100,1000,5000,20000,50000,100000",
        help="Comma-separated target token counts (approx; chars use x4).",
    )
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
    check_matches(binary, model, corpus, target_chars)


if __name__ == "__main__":
    main()
