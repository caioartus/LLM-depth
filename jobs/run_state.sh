# Sourced by jobs/submit.sh, jobs/train.sbatch and jobs/depth/coord_check.sbatch, so a run that
# already finished is never trained again and a run whose logs would be lost is not restarted.
#
# run_state OUT_DIR prints one of
#   finished  the last eval is at max_iters: nothing to do
#   resume    not finished, has ckpt.pt: train.sbatch continues from it
#   partial   not finished, no ckpt.pt, but has logged rows (a crashed or diverged tuning run):
#             running it again starts from step 0 and overwrites its train.csv / eval.csv / diag.csv
#   new       nothing logged yet

run_state() {
    local dir="$1" max last
    if [[ -f "$dir/config.json" && -f "$dir/eval.csv" ]]; then
        max=$(sed -n 's/.*"max_iters": *\([0-9][0-9]*\).*/\1/p' "$dir/config.json" | head -1)
        last=$(tail -n 1 "$dir/eval.csv" | cut -d, -f1)
        if [[ -n "$max" && "$last" == "$max" ]]; then echo finished; return; fi
    fi
    if [[ -f "$dir/ckpt.pt" ]]; then
        echo resume
    elif [[ ( -f "$dir/train.csv" && $(wc -l < "$dir/train.csv") -gt 1 ) || ( -f "$dir/eval.csv" && $(wc -l < "$dir/eval.csv") -gt 1 ) ]]; then
        echo partial
    else
        echo new
    fi
}
