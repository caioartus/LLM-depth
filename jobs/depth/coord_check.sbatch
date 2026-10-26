#!/bin/bash
# Coordinate check of the parameterization (CLAUDE.md "Before sweeps"), one GPU job.
# 10-step runs of SP and CompleteP, 3 seeds each, over
#   depth L = 2, 4, 8, 16, 32, 64 at width 256    (checks the depth scaling)
#   width d = 256, 512, 1024 at depth 2           (checks the width scaling)
# Each run records the mean |output| of the embedding, attention, MLP, last block and lm_head
# at every step (config/depth/coord_check_common.py). Under CompleteP these should stay roughly
# flat as L and d grow; under SP they drift with depth.
#
# Usage (from the repo root, after data/slimpajama is prepared):
#   mkdir -p logs
#   sbatch jobs/depth/coord_check.sbatch
#   uv run --frozen python analysis/plot_coord_check.py      # -> results/coord_check/*.png
#
# Output: out/coord_check/<sp|completep>-<d>x<L>-seed<s>/eval.csv

#SBATCH --job-name=coord-check
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH" # where uv usually lives
export PYTHONUNBUFFERED=1

if [[ ! -f data/slimpajama/train.bin ]]; then
    echo "data/slimpajama/train.bin not found, run: sbatch jobs/prepare_data.sh slimpajama" >&2
    exit 1
fi

echo "job $SLURM_JOB_ID on $(hostname) at $(date)"

for PARAM in sp completep; do
    for SHAPE in 256x2 256x4 256x8 256x16 256x32 256x64 512x2 1024x2; do
        for SEED in 1 2 3; do
            D=${SHAPE%x*}
            L=${SHAPE#*x}
            OUT_DIR="out/coord_check/$PARAM-$SHAPE-seed$SEED"
            echo "=== $OUT_DIR"
            # the shape goes *before* the config file, which derives the head count and multipliers from it
            uv run --frozen python train.py --n_embd="$D" --n_layer="$L" --batch_size=8 --seed="$SEED" \
                "config/depth/$PARAM/coord_check.py" --out_dir="$OUT_DIR"
        done
    done
done

echo "finished at $(date)"
