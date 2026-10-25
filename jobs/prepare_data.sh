#!/bin/bash
# Download + tokenize a dataset into data/<dataset>/{train,val}.bin (CPU only).
# <dataset> is openwebtext (default) or slimpajama.
#
# Usage (from the repo root):
#   mkdir -p logs                                   # once; SLURM needs it to exist
#   sbatch jobs/prepare_data.sh                     # openwebtext, num_docs default from prepare.py (100k docs)
#   sbatch jobs/prepare_data.sh --num_docs 1000000
#   sbatch jobs/prepare_data.sh --num_docs all      # full openwebtext: ~54GB cache + 17GB train.bin
#   sbatch jobs/prepare_data.sh slimpajama --num_docs 1000000
#   sbatch jobs/prepare_data.sh slimpajama --num_docs all   # full SlimPajama-6B: ~14GB download + ~12GB train.bin
#
# Then train on it with --dataset=<dataset>, e.g. jobs/submit.sh NAME config/cluster_test.py --dataset=slimpajama
#
# The huggingface cache goes to ~/.cache/huggingface by default. If your home quota is small,
# point it at scratch space, e.g. export HF_HOME=/scratch/$USER/hf

#SBATCH --job-name=prep-data
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

DATASET="openwebtext"
if [[ $# -gt 0 && "$1" != --* ]]; then
    DATASET="$1"
    shift                            # everything after the dataset goes to prepare.py
fi

if [[ ! -f "data/$DATASET/prepare.py" ]]; then
    echo "data/$DATASET/prepare.py not found" >&2
    exit 1
fi

export PATH="$HOME/.local/bin:$PATH" # where uv usually lives
export PYTHONUNBUFFERED=1

echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "dataset: $DATASET  args: $*"

uv run --frozen python "data/$DATASET/prepare.py" "$@"

echo "finished at $(date)"
