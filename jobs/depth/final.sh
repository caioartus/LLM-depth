#!/bin/bash
# Submit the 14 final runs of the depth/width study (CLAUDE.md "Final runs"), each at the full
# compute budget. The learning rates come from results/best_lr.csv, written by
# analysis/collect_results.py from the tuning runs. The script refuses to submit if a needed
# LR is missing or its optimum is on the edge of the grid (edge rule not done yet).
#
# Usage (from the repo root):
#   uv run --frozen python analysis/collect_results.py   # refresh results/best_lr.csv first
#   DRY_RUN=1 jobs/depth/final.sh                         # print the sbatch commands without submitting
#   jobs/depth/final.sh                                   # on a 96GB H100
#   GPU=gpu:a100-80:1 jobs/depth/final.sh                 # on another GPU type (sinfo -o "%P %G")
#
# Run names: sp-512x16 (shared by (a) and (b)), sp-transfer-<shape>, sp-tuned-<shape>, completep-<shape>.
# A crashed run is resumed by running this script again (finished runs just stop at their last step).

set -euo pipefail

BEST=results/best_lr.csv
[[ -f "$BEST" ]] || { echo "$BEST not found, run: uv run --frozen python analysis/collect_results.py" >&2; exit 1; }

# best_lr <parameterization> <shape>: prints the tuned LR, or fails if it is missing or on the grid edge
best_lr() {
    local row
    row=$(awk -F, -v p="$1" -v s="$2" 'NR > 1 && $1 == p && $2 == s' "$BEST")
    [[ -n "$row" ]] || { echo "no tuned LR for $1 $2 in $BEST" >&2; return 1; }
    [[ "$(cut -d, -f7 <<< "$row")" == "False" ]] || { echo "$1 $2: optimum on the grid edge, see results/best_lr.md" >&2; return 1; }
    cut -d, -f3 <<< "$row"
}

OTHER_SHAPES=(256x64 384x28 640x10 1024x4)

# look every LR up first, so nothing is submitted unless all of them are usable
SP_LR=$(best_lr sp 512x16)                 # (a) SP-transfer: the SP optimum at 512x16, for every shape
COMPLETEP_LR=$(best_lr completep 512x16)   # (c) CompleteP: the eta_base optimum at 512x16, for every shape
SP_TUNED=()                                # (b) SP-tuned: the SP optimum of each shape
for SHAPE in "${OTHER_SHAPES[@]}"; do
    LR=$(best_lr sp "$SHAPE")              # a plain assignment, so set -e stops here if the lookup fails
    SP_TUNED+=("$LR")
done
echo "SP LR (512x16): $SP_LR   CompleteP eta_base (512x16): $COMPLETEP_LR"

# full runs take ~2-6 h on an A100 (deep shapes are slower), so raise train.sbatch's 2 h limit.
# sbatch reads these variables and they override the #SBATCH --time and --gres lines.
# all final runs go on the same GPU type, so their wall-clock times can be compared. the fastest in
# gpu-long is the 96GB H100 (h100-47 are MIG halves). another GPU: GPU=gpu:a100-80:1 jobs/depth/final.sh
# (on >= 75GB GPUs config/depth/common.py doubles the micro-batch; MFU uses the GPU's own peak)
export SBATCH_TIMELIMIT=08:00:00
export SBATCH_GRES="${GPU:-gpu:h100-96:1}"
echo "GPU: $SBATCH_GRES"

submit() {
    if [[ -n "${DRY_RUN:-}" ]]; then echo "jobs/submit.sh $*"; else jobs/submit.sh "$@"; fi
}

# SP 512x16: both SP-transfer and SP-tuned
submit sp-512x16 config/depth/sp/512x16.py --learning_rate="$SP_LR"

for i in "${!OTHER_SHAPES[@]}"; do
    SHAPE=${OTHER_SHAPES[$i]}
    submit "sp-transfer-$SHAPE" "config/depth/sp/$SHAPE.py" --learning_rate="$SP_LR"
    if [[ "${SP_TUNED[$i]}" == "$SP_LR" ]]; then
        echo "skipping sp-tuned-$SHAPE: its tuned LR equals the 512x16 one, so it is the same run as sp-transfer-$SHAPE"
    else
        submit "sp-tuned-$SHAPE" "config/depth/sp/$SHAPE.py" --learning_rate="${SP_TUNED[$i]}"
    fi
done

for SHAPE in 256x64 384x28 512x16 640x10 1024x4; do
    submit "completep-$SHAPE" "config/depth/completep/$SHAPE.py" --learning_rate="$COMPLETEP_LR"
done
