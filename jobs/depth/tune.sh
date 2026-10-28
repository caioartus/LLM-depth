#!/bin/bash
# Submit the learning-rate sweeps of the depth/width study (CLAUDE.md "Tuning protocol").
# Every job is one short run on a fraction of the shape's tokens (tune_fraction in config/depth/tune.py).
#
# Usage (from the repo root):
#   jobs/depth/tune.sh sp                        # 5 shapes x 5 LRs = 25 jobs
#   jobs/depth/tune.sh completep                 # 5 shapes x 5 LRs = 25 jobs
#   jobs/depth/tune.sh sp 256x64                 # one shape, all 5 LRs
#   jobs/depth/tune.sh sp 256x64 4.8e-3          # edge rule: extra LR point(s) for one shape
#   jobs/depth/tune.sh completep 512x4           # optional diagnostic shape, not run for now (CLAUDE.md)
#   DRY_RUN=1 jobs/depth/tune.sh sp              # print the commands and each run's state, submit nothing
#
# Runs that are already finished, queued or running are skipped (jobs/submit.sh), so running this
# again only submits what is missing. Run names: tune-<sp|completep>-<shape>-lr<lr>.
# Results: out/<name>/eval.csv. When all have finished, run analysis/collect_results.py: it picks
# the optimum per shape -> results/best_lr.{csv,md}.

set -euo pipefail

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

PARAM="${1:-}"
case "$PARAM" in
    sp)
        SHAPES=(256x64 384x28 512x16 640x10 1024x4)
        LRS=(1.5e-4 3e-4 6e-4 1.2e-3 2.4e-3)
        ;;
    completep)
        SHAPES=(512x16 256x64 1024x4 384x28 640x10)                         # 512x16 sets the LR of (c), each shape's optimum is (d)
        LRS=(0.0009765625 0.001953125 0.00390625 0.0078125 0.015625)        # eta_base = 2^-10 ... 2^-6
        ;;
    *)
        usage
        ;;
esac
shift
if [[ $# -gt 0 ]]; then SHAPES=("$1"); shift; fi    # optional: a single shape
if [[ $# -gt 0 ]]; then LRS=("$@"); fi              # optional: custom LRs

source jobs/run_state.sh
FAILED=()
for SHAPE in "${SHAPES[@]}"; do
    CONFIG="config/depth/$PARAM/$SHAPE.py"
    [[ -f "$CONFIG" ]] || { echo "$CONFIG not found" >&2; exit 1; }
    for LR in "${LRS[@]}"; do
        NAME="tune-$PARAM-$SHAPE-lr$LR"
        CMD=(jobs/submit.sh "$NAME" "$CONFIG" config/depth/tune.py --learning_rate="$LR")
        if [[ -n "${DRY_RUN:-}" ]]; then echo "${CMD[*]}   [$(run_state "out/$NAME")]"; else "${CMD[@]}" || FAILED+=("$NAME"); fi
    done
done
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "not submitted (see the messages above): ${FAILED[*]}" >&2
    exit 1
fi
