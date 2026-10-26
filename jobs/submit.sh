#!/bin/bash
# Submit a named training run. The name shows up in squeue, the log file (logs/<name>-<id>.out),
# the out dir (out/<name>/) and as the wandb run name. Submitting the same name with the same
# arguments again resumes a crashed run from its last checkpoint (see jobs/train.sbatch).
#
# Usage (from the repo root):
#   jobs/submit.sh NAME [config.py] [--key=value ...]
#
# Examples:
#   jobs/submit.sh test-run
#   jobs/submit.sh d6-w384 config/cluster_test.py
#   jobs/submit.sh d12-w384 config/cluster_test.py --n_layer=12
#   jobs/submit.sh d24-completep config/cluster_test.py --n_layer=24 --mup_enabled=True --depth_alpha_enabled=True --depth_multiplier=4.0
#
# To change SLURM resources, use sbatch directly:
#   sbatch --job-name=NAME --time=08:00:00 jobs/train.sbatch config/cluster_test.py --max_iters=10000

set -euo pipefail

if [[ $# -lt 1 || "$1" == --* || "$1" == *.py ]]; then
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

NAME="$1"
shift

# same rule train.sbatch enforces, checked here so a bad name fails before queueing
if [[ ! "$NAME" =~ ^[A-Za-z][A-Za-z0-9._-]*$ || "$NAME" =~ ^(True|False|None)$ ]]; then
    echo "name '$NAME' must start with a letter and use only letters, digits, . _ -" >&2
    exit 1
fi

# two jobs with the same name would write to the same out/<name>/ directory
if [[ -n "$(squeue -u "$USER" -h -n "$NAME" -o %i 2>/dev/null)" ]]; then
    echo "a job named '$NAME' is already queued or running (squeue -u $USER -n $NAME)" >&2
    exit 1
fi

mkdir -p logs
sbatch --job-name="$NAME" "$(dirname "$0")/train.sbatch" "$@"
