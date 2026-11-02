#!/bin/bash
# Submit the extra runs added after the final runs, on the same 96GB H100s.
# None of them is a condition: analysis/collect_results.py gives them their own kinds from the name
# prefix, so they never enter conditions (a)-(d) in runs.csv or results.ipynb.
#   check-sp-256x64-lr2.4e-3   post-hoc LR check: SP 256x64 at full length one grid step above its best
#                              full-length LR (1.2e-3), which is not bracketed from above
#   seed1-sp-512x16, seed2-..  seeds 1 and 2 of sp-512x16 and completep-512x16 (seed 0), same LR:
#   seed1-completep-512x16, .. the run-to-run noise. The seed only changes the init (the data order
#                              and the eval windows are fixed)
#
# Usage (from the repo root):
#   DRY_RUN=1 jobs/depth/extra.sh    # print the commands and each run's state, submit nothing
#   jobs/depth/extra.sh
#
# The names are new, so no existing run is touched. Running it again only submits what is missing
# (jobs/submit.sh skips finished, queued and running runs and resumes a crashed one).

set -euo pipefail

# the seed-0 runs' LRs, read from their own run_info.txt so the extra seeds match them exactly
first_lr() {
    local info="results/runs/$1/run_info.txt"
    [[ -f "$info" ]] || { echo "$info not found, run analysis/collect_results.py first" >&2; return 1; }
    sed -n 's/^command: .*--learning_rate=\([^ ]*\).*/\1/p' "$info" | head -1
}
SP_LR=$(first_lr sp-512x16)
COMPLETEP_LR=$(first_lr completep-512x16)
[[ -n "$SP_LR" && -n "$COMPLETEP_LR" ]] || { echo "could not read the 512x16 LRs" >&2; exit 1; }
echo "SP LR (512x16): $SP_LR   CompleteP eta_base (512x16): $COMPLETEP_LR"

# same GPU and time limit as jobs/depth/final.sh
export SBATCH_TIMELIMIT=08:00:00
export SBATCH_GRES="${GPU:-gpu:h100-96:1}"
echo "GPU: $SBATCH_GRES"

source jobs/run_state.sh
FAILED=()
submit() {
    if [[ -n "${DRY_RUN:-}" ]]; then echo "jobs/submit.sh $*   [$(run_state "out/$1")]"; else jobs/submit.sh "$@" || FAILED+=("$1"); fi
}

submit check-sp-256x64-lr2.4e-3 config/depth/sp/256x64.py --learning_rate=2.4e-3

for SEED in 1 2; do
    submit "seed$SEED-sp-512x16" config/depth/sp/512x16.py --learning_rate="$SP_LR" --seed="$SEED"
    submit "seed$SEED-completep-512x16" config/depth/completep/512x16.py --learning_rate="$COMPLETEP_LR" --seed="$SEED"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "not submitted (see the messages above): ${FAILED[*]}" >&2
    exit 1
fi
