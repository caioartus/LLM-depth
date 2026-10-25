#!/bin/bash
# Train a model with nanoGPT's train.py on one GPU.
#
# Usage (from the repo root):
#   mkdir -p logs                                   # once; SLURM needs it to exist
#   sbatch jobs/prepare_data.sh                     # once, to create data/openwebtext/{train,val}.bin
#   sbatch jobs/train.sbatch                        # default config below
#   sbatch jobs/train.sbatch config/cluster_test.py --max_iters=500
#   sbatch jobs/train.sbatch --n_layer=12           # default config + overrides
#   sbatch --gres=gpu:a100:1 jobs/train.sbatch      # pick a GPU model (name is cluster-specific)
#
# Flags given to sbatch on the command line override the #SBATCH lines below.
# Checkpoints and csv logs go to out/<job name>-<job id> unless you pass --out_dir=...
#
# To name a run, use jobs/submit.sh (or sbatch --job-name=<name> jobs/train.sbatch ...).
# The name is used for the log file, the out dir, the wandb run name and in squeue.

#SBATCH --job-name=train
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job id

set -euo pipefail

DEFAULT_CONFIG="config/cluster_test.py"
if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG="$1"
    shift                            # everything after the config goes to train.py
else
    CONFIG="$DEFAULT_CONFIG"         # no config given, only --key=value overrides
fi

export PATH="$HOME/.local/bin:$PATH" # where uv usually lives
export PYTHONUNBUFFERED=1
# export WANDB_MODE=offline          # uncomment if compute nodes have no internet

# dataset: --dataset=... on the command line, else dataset = '...' in the config, else train.py's default
DATASET=$(printf '%s\n' "$@" | sed -n 's/^--dataset=//p' | tail -1)
[[ -z "$DATASET" ]] && DATASET=$(sed -n "s/^dataset *= *['\"]\([^'\"]*\)['\"].*/\1/p" "$CONFIG" | tail -1)
[[ -z "$DATASET" ]] && DATASET="openwebtext"
if [[ ! -f "data/$DATASET/train.bin" ]]; then
    echo "data/$DATASET/train.bin not found, run: sbatch jobs/prepare_data.sh $DATASET" >&2
    exit 1
fi

# train.py's configurator would turn a name like "123" or "True" into a number/bool and reject it
if [[ ! "$SLURM_JOB_NAME" =~ ^[A-Za-z][A-Za-z0-9._-]*$ || "$SLURM_JOB_NAME" =~ ^(True|False|None)$ ]]; then
    echo "job name '$SLURM_JOB_NAME' must start with a letter and use only letters, digits, . _ -" >&2
    exit 1
fi

OUT_DIR="out/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) at $(date)"
echo "config: $CONFIG  extra args: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# record what this run was, so out/ dirs can be told apart later
{
    echo "name:    $SLURM_JOB_NAME"
    echo "job id:  $SLURM_JOB_ID"
    echo "started: $(date)"
    if COMMIT=$(git rev-parse --short HEAD 2>/dev/null); then
        git diff --quiet HEAD || COMMIT="$COMMIT (+ uncommitted changes)"
    else
        COMMIT=unknown
    fi
    echo "commit:  $COMMIT"
    echo "config:  $CONFIG"
    echo "args:    $*"
} > "$OUT_DIR/run_info.txt"

# --out_dir and --wandb_run_name come before "$@" so they can still be overridden from the command line
uv run --frozen python train.py "$CONFIG" \
    --out_dir="$OUT_DIR" \
    --wandb_run_name="$SLURM_JOB_NAME" \
    "$@"

echo "finished at $(date)"
