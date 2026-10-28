#!/bin/bash
# Submit a named training run. The name shows up in squeue, the log file (logs/<name>-<id>.out),
# the out dir (out/<name>/) and as the wandb run name. Submitting the same name with the same
# arguments again resumes a crashed run from its last checkpoint (see jobs/train.sbatch).
# A name that is queued, running or finished is skipped (exit 0), so scripts that submit many runs
# can be run again safely. A crashed run without a checkpoint is refused unless RESTART=1.
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
    echo "skipping $NAME: already queued or running (squeue -u $USER -n $NAME)"
    exit 0
fi

# same command check as train.sbatch, done here so a mismatch fails before queueing. it comes first,
# so a run that finished with other arguments (e.g. an old LR) is refused rather than skipped
OUT_DIR="out/$NAME"
if [[ -f "$OUT_DIR/run_info.txt" ]]; then
    if [[ $# -gt 0 && "$1" != --* ]]; then COMMAND="$*"; else COMMAND="config/cluster_test.py $*"; fi
    FIRST_COMMAND=$(sed -n 's/^command: //p' "$OUT_DIR/run_info.txt" | head -1)
    if [[ "$FIRST_COMMAND" != "$COMMAND" ]]; then
        echo "refusing $NAME: $OUT_DIR is from a different run" >&2
        echo "  before: $FIRST_COMMAND" >&2
        echo "  now:    $COMMAND" >&2
        exit 1
    fi
fi

# never train a finished run again, and never restart a run whose logs would be overwritten
# (jobs/train.sbatch checks the same when the job starts)
source "$(dirname "$0")/run_state.sh"
case "$(run_state "$OUT_DIR")" in
    finished)
        echo "skipping $NAME: already finished ($OUT_DIR/eval.csv)"
        exit 0
        ;;
    partial)
        if [[ -z "${RESTART:-}" ]]; then
            echo "refusing $NAME: $OUT_DIR has logs but no checkpoint, so it would start over from step 0" >&2
            echo "  and overwrite them. look at it first (tail logs/$NAME-*.out), then to rerun it from scratch:" >&2
            echo "  RESTART=1 jobs/submit.sh $NAME $*" >&2
            exit 1
        fi
        ;;
esac

mkdir -p logs
sbatch --job-name="$NAME" "$(dirname "$0")/train.sbatch" "$@"
