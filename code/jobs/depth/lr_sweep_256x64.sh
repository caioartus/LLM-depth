#!/bin/bash
# Full-length LR sweep at the deepest shape (256x64), added 2026-11-13, to bracket the full-length
# optimum of both parameterisations. Same 96GB H100s and settings as jobs/depth/final.sh; only the LR changes.
# Runs already done at 256x64, full length:
#   SP         6e-4 (sp-tuned-256x64), 1.2e-3 (sp-transfer-256x64), 2.4e-3 (check-sp-256x64-lr2.4e-3)
#              -> still improving at 2.4e-3, so not bracketed from above
#   CompleteP  2^-7 = 0.0078125 (completep-256x64) only
# Round 1 (default) adds:
#   check-sp-256x64-lr4.8e-3            one grid step above the best SP LR
#   check-completep-256x64-lr0.00390625 eta_base = 2^-8, one grid step below
#   check-completep-256x64-lr0.015625   eta_base = 2^-6, one grid step above
# Bracketed = the best LR has a worse (or diverged) neighbour on both sides. If an edge LR wins, extend
# by one grid step (x2 or /2) with round 2, e.g.:
#   jobs/depth/lr_sweep_256x64.sh sp 9.6e-3
#   jobs/depth/lr_sweep_256x64.sh completep 0.03125
# The names start with check-, so analysis/collect_results.py files them as post-hoc checks, not
# as conditions (a)-(d).
#
# Usage (from code/):
#   DRY_RUN=1 jobs/depth/lr_sweep_256x64.sh    # print the commands and each run's state, submit nothing
#   jobs/depth/lr_sweep_256x64.sh              # round 1
#   jobs/depth/lr_sweep_256x64.sh PARAM LR     # one extra run, PARAM = sp | completep
#
# Running it again only submits what is missing (jobs/submit.sh skips finished, queued and running
# runs and resumes a crashed one).

set -euo pipefail

# a full 256x64 run takes ~1.7 h on an H100; a 3 h limit leaves room for a resume and backfills
# sooner than final.sh's 8 h
export SBATCH_TIMELIMIT="${TIMELIMIT:-03:00:00}"
export SBATCH_GRES="${GPU:-gpu:h100-96:1}"
echo "GPU: $SBATCH_GRES   time limit: $SBATCH_TIMELIMIT"

source jobs/run_state.sh
FAILED=()
submit() {
    if [[ -n "${DRY_RUN:-}" ]]; then echo "jobs/submit.sh $*   [$(run_state "out/$1")]"; else jobs/submit.sh "$@" || FAILED+=("$1"); fi
}

run() {  # run PARAM LR
    [[ "$1" == sp || "$1" == completep ]] || { echo "PARAM must be sp or completep, got '$1'" >&2; exit 1; }
    submit "check-$1-256x64-lr$2" "config/depth/$1/256x64.py" --learning_rate="$2"
}

if [[ $# -eq 2 ]]; then
    run "$1" "$2"
elif [[ $# -eq 0 ]]; then
    run sp 4.8e-3
    run completep 0.00390625
    run completep 0.015625
else
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "not submitted (see the messages above): ${FAILED[*]}" >&2
    exit 1
fi
