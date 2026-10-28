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
# Checkpoints and logs go to out/<job name>/ unless you pass --out_dir=...
#
# To name a run, use jobs/submit.sh (or sbatch --job-name=<name> jobs/train.sbatch ...).
# The name is used for the log file, the out dir, the wandb run name and in squeue.
#
# Crash recovery: submit the same name with the same arguments again. If out/<name>/ckpt.pt
# exists, the run resumes from that checkpoint (otherwise it starts over). Reusing a name with
# different arguments is refused, so an old run can't be continued by mistake.

#SBATCH --job-name=train
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job id
# get an email when a job fails or hits the time limit (remove the first "# " to enable;
# --mail-user=you@example.com if the cluster doesn't know your address)
# #SBATCH --mail-type=FAIL,TIME_LIMIT

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

# dataset: --dataset=... on the command line, else dataset = '...' in the config (or in a file the
# config execs, like config/depth/common.py), else train.py's default
DATASET=$(printf '%s\n' "$@" | sed -n 's/^--dataset=//p' | tail -1)
CONFIG_FILES=("$CONFIG" $(sed -n "s/^exec(open(['\"]\([^'\"]*\)['\"]).*/\1/p" "$CONFIG"))
[[ -z "$DATASET" ]] && DATASET=$(sed -n "s/^dataset *= *['\"]\([^'\"]*\)['\"].*/\1/p" "${CONFIG_FILES[@]}" | tail -1)
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

OUT_DIR="out/${SLURM_JOB_NAME}"
mkdir -p "$OUT_DIR"

# resume if a previous attempt of this run left a checkpoint. the first attempt's command is
# recorded in run_info.txt, and a different command under the same name is refused
RUN_COMMAND="$CONFIG $*"
RESUME_ARGS=()
if [[ -f "$OUT_DIR/run_info.txt" ]]; then
    FIRST_COMMAND=$(sed -n 's/^command: //p' "$OUT_DIR/run_info.txt" | head -1)
    if [[ "$FIRST_COMMAND" != "$RUN_COMMAND" ]]; then
        echo "$OUT_DIR is from a different run:" >&2
        echo "  before: $FIRST_COMMAND" >&2
        echo "  now:    $RUN_COMMAND" >&2
        echo "pick another name, or delete $OUT_DIR to start over" >&2
        exit 1
    fi
    if [[ -f "$OUT_DIR/ckpt.pt" ]]; then
        RESUME_ARGS=(--init_from=resume)
        echo "resuming from $OUT_DIR/ckpt.pt"
    fi
fi

# a finished run is not trained again, and a run that would start over and overwrite its logs
# needs RESTART=1 (jobs/run_state.sh; jobs/submit.sh checks the same before queueing)
source jobs/run_state.sh
case "$(run_state "$OUT_DIR")" in
    finished)
        echo "$OUT_DIR already finished, nothing to do"
        exit 0
        ;;
    partial)
        if [[ -z "${RESTART:-}" ]]; then
            echo "$OUT_DIR has logs but no checkpoint, so this would start over and overwrite them." >&2
            echo "to do that: RESTART=1 jobs/submit.sh $SLURM_JOB_NAME $RUN_COMMAND" >&2
            exit 1
        fi
        ;;
esac

echo "job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) at $(date)"
echo "config: $CONFIG  extra args: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# record what this run was, one block per attempt, so out/ dirs can be told apart later
{
    echo "=== attempt: job $SLURM_JOB_ID, started $(date)${RESUME_ARGS:+, resumed from checkpoint}"
    echo "name:    $SLURM_JOB_NAME"
    if COMMIT=$(git rev-parse --short HEAD 2>/dev/null); then
        git diff --quiet HEAD || COMMIT="$COMMIT (+ uncommitted changes)"
    else
        COMMIT=unknown
    fi
    echo "commit:  $COMMIT"
    echo "command: $RUN_COMMAND"
} >> "$OUT_DIR/run_info.txt"

# --out_dir and --wandb_run_name come before "$@" so they can still be overridden from the command line
uv run --frozen python train.py "$CONFIG" \
    --out_dir="$OUT_DIR" \
    --wandb_run_name="$SLURM_JOB_NAME" \
    "$@" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}

echo "finished at $(date)"
