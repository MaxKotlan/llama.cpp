#!/usr/bin/env bash
set -euo pipefail

# Run tokenizer benchmark across all large Ollama blobs using the matplotlib script.
# Uses the venv Python and writes plots under benchmarks/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS=$(find /var/lib/ollama/blobs -maxdepth 1 -type f -name 'sha256-*' -print | sort | paste -sd, -)

if [[ -z "${MODELS}" ]]; then
  echo "No models found under /var/lib/ollama/blobs." >&2
  exit 1
fi

source "${REPO_ROOT}/.venv/bin/activate"
export MPLCONFIGDIR=/tmp/mplcfg

python "${REPO_ROOT}/scripts/benchmark_tokenizers_plot.py" \
  --models "${MODELS}" \
  --runs 1 --warmup 0 \
  --cold-runs 1 --cold-warmup 0 \
  --targets 1000,4000,8000,20000,40000,70000,100000,250000,500000,750000,1000000 \
  --output-dir "${REPO_ROOT}/benchmarks"

echo "Benchmark plots written to ${REPO_ROOT}/benchmarks"
